import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio

# We need the EncodecModel from transformers
from transformers import EncodecModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnCodecDataset(Dataset):
    """Dataset for EnCodec fine-tuning.
    
    Loads raw 24kHz audio chunks for reconstruction learning.
    Focuses on the high-frequency and breathiness reconstruction.
    """
    def __init__(self, raw_dir: str | Path, chunk_size: int = 24000): # 1 second chunks
        self.raw_dir = Path(raw_dir)
        self.chunk_size = chunk_size
        self.audio_files = list(self.raw_dir.rglob("*.wav")) + list(self.raw_dir.rglob("*.flac"))
        
    def __len__(self) -> int:
        return len(self.audio_files)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        audio_path = self.audio_files[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 24k if necessary
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)
                
            # Mono conversion if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Random chunking
            if waveform.shape[1] > self.chunk_size:
                start = torch.randint(0, waveform.shape[1] - self.chunk_size, (1,)).item()
                waveform = waveform[:, start:start+self.chunk_size]
            else:
                # Pad if too short
                pad_len = self.chunk_size - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_len))
                
            return waveform # [1, T]
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # return silence on error
            return torch.zeros((1, self.chunk_size))

def multi_resolution_stft_loss(
    x: torch.Tensor, 
    y: torch.Tensor, 
    fft_sizes: list[int] = [1024, 2048, 512],
    hop_sizes: list[int] = [120, 240, 50],
    win_lengths: list[int] = [600, 1200, 240]
) -> torch.Tensor:
    """Spectral convergence and log magnitude STFT loss."""
    loss = 0.0
    for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
        window = torch.hann_window(wl).to(x.device)
        
        stft_x = torch.stft(x.squeeze(1), n_fft=fs, hop_length=hs, win_length=wl, window=window, return_complex=True)
        stft_y = torch.stft(y.squeeze(1), n_fft=fs, hop_length=hs, win_length=wl, window=window, return_complex=True)
        
        mag_x = torch.abs(stft_x) + 1e-7
        mag_y = torch.abs(stft_y) + 1e-7
        
        # Spectral convergence
        sc_loss = torch.norm(mag_x - mag_y, p="fro") / torch.norm(mag_x, p="fro")
        
        # Log STFT magnitude loss
        log_mag_loss = F.l1_loss(torch.log(mag_x), torch.log(mag_y))
        
        loss += sc_loss + log_mag_loss
        
    return loss / len(fft_sizes)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune EnCodec Decoder for high-freq/breathiness.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory with raw 24kHz audio (Expresso/Intimate)")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/codec"), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing dataset...")
    dataset = EnCodecDataset(args.raw_dir)
    if len(dataset) == 0:
        logger.error(f"No audio files found in {args.raw_dir}")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    logger.info("Loading pre-trained EnCodec Model...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(args.device)
    
    # Freeze encoder and quantizer, ONLY train the decoder
    model.eval() # Set entire model to eval to freeze batch norms if any
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.quantizer.parameters():
        param.requires_grad = False
        
    # Unfreeze decoder
    model.decoder.train()
    for param in model.decoder.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-5, betas=(0.8, 0.99))
    
    logger.info("Starting Fine-tuning...")
    step = 0
    while step < args.max_steps:
        for audio in loader:
            audio = audio.to(args.device) # [B, 1, T]
            
            optimizer.zero_grad()
            
            # Forward pass: Encode -> Quantize -> Decode
            with torch.no_grad():
                encoded = model.encode(audio, bandwidth=6.0)
                audio_codes = encoded.audio_codes
                audio_scales = encoded.audio_scales
                
            # Decode requires gradients
            decoded = model.decode(audio_codes, audio_scales).audio_values # [B, 1, T]
            
            # Ensure shape match
            min_len = min(audio.shape[-1], decoded.shape[-1])
            audio_c = audio[..., :min_len]
            decoded_c = decoded[..., :min_len]
            
            # Calculate Losses
            # 1. Multi-resolution STFT loss (focuses on frequencies)
            loss_stft = multi_resolution_stft_loss(audio_c, decoded_c)
            
            # 2. High-frequency emphasis loss (Optional, specific for breathiness)
            # We can compute STFT and weigh high-frequency bins more
            stft_real = torch.stft(audio_c.squeeze(1), n_fft=1024, hop_length=240, return_complex=True)
            stft_pred = torch.stft(decoded_c.squeeze(1), n_fft=1024, hop_length=240, return_complex=True)
            mag_real = torch.abs(stft_real)
            mag_pred = torch.abs(stft_pred)
            
            # Frequencies > 6kHz (approx bin 256+ out of 513)
            loss_hf = F.l1_loss(mag_pred[:, 256:, :], mag_real[:, 256:, :])
            
            total_loss = loss_stft + 2.0 * loss_hf
            
            total_loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                logger.info(f"Step {step}: Total Loss = {total_loss.item():.4f}, STFT = {loss_stft.item():.4f}, HF = {loss_hf.item():.4f}")
                
            if step > 0 and step % 1000 == 0:
                out_path = args.output_dir / f"encodec_decoder_step_{step}.pt"
                torch.save(model.decoder.state_dict(), out_path)
                logger.info(f"Saved checkpoint to {out_path}")
                
            step += 1
            if step >= args.max_steps:
                break
                
    # Final save
    final_path = args.output_dir / "encodec_decoder_finetuned.pt"
    torch.save(model.decoder.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
