import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from tmrvc_train.models import DisentangledUCLM
from tmrvc_train.dataset import DisentangledUCLMDataset
from tmrvc_train.trainer import UCLMTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Pad tensors to max length in batch."""
    # Find max length
    max_len = max(item["target_a"].shape[1] for item in batch)
    
    source_a_t = []
    target_a = []
    target_b = []
    explicit_state = []
    ssl_state = []
    speaker_embed = []
    
    for item in batch:
        T = item["target_a"].shape[1]
        pad_len = max_len - T
        
        # [8, T] -> [8, max_len]
        s_a = torch.nn.functional.pad(item["source_a_t"], (0, pad_len), value=-1)
        t_a = torch.nn.functional.pad(item["target_a"], (0, pad_len), value=-1)
        t_b = torch.nn.functional.pad(item["target_b"], (0, pad_len), value=-1)
        
        # [T, d] -> [max_len, d]
        e_s = torch.nn.functional.pad(item["explicit_state"].transpose(0, 1), (0, pad_len)).transpose(0, 1)
        s_s = torch.nn.functional.pad(item["ssl_state"].transpose(0, 1), (0, pad_len)).transpose(0, 1)
        
        source_a_t.append(s_a)
        target_a.append(t_a)
        target_b.append(t_b)
        explicit_state.append(e_s)
        ssl_state.append(s_s)
        speaker_embed.append(item["speaker_embed"])
        
    return {
        "source_a_t": torch.stack(source_a_t),
        "target_a": torch.stack(target_a),
        "target_b": torch.stack(target_b),
        "explicit_state": torch.stack(explicit_state),
        "ssl_state": torch.stack(ssl_state),
        "speaker_embed": torch.stack(speaker_embed)
    }

def main():
    parser = argparse.ArgumentParser(description="Train Disentangled UCLM")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    logger.info("Initializing Disentangled UCLM dataset...")
    dataset = DisentangledUCLMDataset(args.cache_dir, max_frames=args.max_frames)
    
    if len(dataset) == 0:
        logger.error("No valid dataset found in %s", args.cache_dir)
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    logger.info("Initializing Disentangled UCLM model...")
    model = DisentangledUCLM(
        d_model=512,
        n_heads=8,
        n_layers=12,
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_explicit=8,
        d_ssl=128,
        d_speaker=192,
        vq_bins=128
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = UCLMTrainer(model, optimizer, device=args.device)
    
    logger.info("Starting training loop...")
    step = 0
    while step < args.max_steps:
        for batch in loader:
            losses = trainer.train_step(batch)
            if step % 10 == 0:
                logger.info(f"Step {step}: Total Loss={losses['loss']:.4f}, VQ Loss={losses['loss_vq']:.4f}")
                
            step += 1
            if step >= args.max_steps:
                break
                
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
