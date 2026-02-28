# UCLM Implementation Roadmap

Kojiro Tanaka — UCLM Implementation Plan
Created: 2026-02-27 (Asia/Tokyo)

> **Goal:** UCLM (Unified Codec Language Model) の実装。
> **優先順位:** TTS 優先、その後 VC。**既存資産は完全置き換え。**

---

## 1. Overview

```
Phase 1: Data Pipeline (UCLM用拡張)
Phase 2: EnCodec Integration
Phase 3: UCLM Architecture (TTS mode)
Phase 4: Training (TTS)
Phase 5: UCLM Architecture (VC mode)
Phase 6: Training (VC + joint)
Phase 7: Streaming Inference
Phase 8: Evaluation & Paper
```

**Total: ~12 weeks**

---

## 2. Phase 1: Data Pipeline Extension

### 2.1 Current Pipeline (prepare_dataset.py)

```
raw_audio → normalize → extract_features → annotate → save
                ↓              ↓              ↓
            24kHz/mono    mel/f0/spk_embed  text/emotion
```

### 2.2 Extended Pipeline for UCLM

```
raw_audio → normalize → extract_features → annotate → encode_codec → save
                ↓              ↓              ↓             ↓
            24kHz/mono    mel/f0/spk_embed  text/emotion  codec_tokens
                                                ↓
                                        voice_state (estimated)
```

### 2.3 New UtteranceMeta Fields

```python
@dataclass
class UtteranceMeta:
    # Existing
    utterance_id: str
    speaker_id: str
    n_frames: int
    duration_sec: float
    text: str
    language_id: int = 0
    emotion_id: int = 6
    emotion_label: str = "neutral"
    emotion_confidence: float = 0.0
    vad: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.5])
    
    # NEW: UCLM specific
    phonemes: str = ""                    # G2P output
    phoneme_ids: list[int] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)  # frames per phoneme
    voice_state_mean: list[float] = field(default_factory=lambda: [0.5]*8)
    # voice_state per-frame is stored separately as voice_state.npy
```

### 2.4 New Files per Utterance

```
data/cache/{dataset}/train/{speaker}/{utt_id}/
├── mel.npy              # [80, T] - existing
├── f0.npy               # [1, T] - existing  
├── spk_embed.npy        # [192] - existing
├── meta.json            # UtteranceMeta
├── codec_tokens.npy     # [n_codebooks, T] - NEW
├── voice_state.npy      # [T, 8] - NEW
├── phoneme_ids.npy      # [L] - NEW
└── durations.npy        # [L] - NEW
```

### 2.5 Implementation Tasks

| Task | File | Description |
|---|---|---|
| Add EnCodec encoder | `tmrvc_data/codec.py` | Wrap EnCodec for token extraction |
| Add voice state estimator | `tmrvc_data/voice_state.py` | Frame-level acoustic params |
| Add G2P to pipeline | `prepare_dataset.py` | Call existing G2P module |
| Add MFA alignment | `prepare_dataset.py` | Call existing alignment module |
| Update UtteranceMeta | `tmrvc_data/cache.py` | Add new fields |

**Duration: 1 week**

---

## 3. Phase 2: EnCodec Integration

### 3.1 EnCodec Setup

```python
# tmrvc_data/codec.py
from transformers import EncodecModel

class EnCodecWrapper:
    """EnCodec wrapper for token extraction and decoding."""
    
    def __init__(self, model_name="facebook/encodec_24khz", device="cuda"):
        self.model = EncodecModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.sample_rate = 24000
        self.n_codebooks = 8
        self.vocab_size = 1024
        
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_samples] at 24kHz
        Returns:
            tokens: [B, n_codebooks, T_frames] at 100fps (10ms/frame)
        """
        with torch.no_grad():
            encoded = self.model.encode(waveform, bandwidth=6.0)
            # encoded.audio_codes: [B, n_codebooks, T, 1]
            return encoded.audio_codes.squeeze(-1)
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, n_codebooks, T_frames]
        Returns:
            waveform: [B, 1, T_samples]
        """
        with torch.no_grad():
            # Add back the dimension
            codes = tokens.unsqueeze(-1)
            decoded = self.model.decode(codes, None)
            return decoded.audio_values
```

### 3.2 Voice State Estimator

```python
# tmrvc_data/voice_state.py

class VoiceStateEstimator(nn.Module):
    """Estimate frame-level voice state parameters from audio."""
    
    # Voice state dimensions:
    # [0]: breathiness    [0, 1]
    # [1]: tension        [0, 1]
    # [2]: arousal        [0, 1]
    # [3]: valence        [-1, 1]
    # [4]: roughness      [0, 1]
    # [5]: voicing        [0, 1]
    # [6]: energy         [0, 1]
    # [7]: rate           [0.5, 2.0]
    
    def __init__(self):
        # Pre-trained on labeled data or use heuristics
        self.breathiness_detector = load_pretrained("breathiness_model.pt")
        self.tension_estimator = load_pretrained("tension_model.pt")
        # ... etc.
        
    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 80, T]
            f0: [B, 1, T]
        Returns:
            voice_state: [B, T, 8]
        """
        B, _, T = mel.shape
        
        # Energy from mel
        energy = mel.mean(dim=1, keepdim=True).squeeze(1)  # [B, T]
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        
        # Voicing from F0
        voicing = (f0.squeeze(1) > 50).float()  # [B, T]
        
        # Breathiness from spectral tilt
        breathiness = self.estimate_breathiness(mel)  # [B, T]
        
        # Tension from F0 variance
        tension = self.estimate_tension(f0)  # [B, T]
        
        # Roughness from F0 micro-perturbations
        roughness = self.estimate_roughness(f0)  # [B, T]
        
        # Arousal from energy + F0 range
        arousal = self.estimate_arousal(energy, f0)  # [B, T]
        
        # Valence from spectral features (heuristic)
        valence = torch.zeros(B, T, device=mel.device)  # Neutral default
        
        # Rate (speaking rate, local)
        rate = torch.ones(B, T, device=mel.device)  # 1.0 = normal
        
        return torch.stack([
            breathiness, tension, arousal, valence,
            roughness, voicing, energy, rate
        ], dim=-1)  # [B, T, 8]
```

### 3.3 Tasks

| Task | Description |
|---|---|
| Install EnCodec | `uv add transformers` |
| Implement `EnCodecWrapper` | Token extraction and decoding |
| Implement `VoiceStateEstimator` | Frame-level acoustic params |
| Test on sample audio | Verify token quality |

**Duration: 1 week**

---

## 4. Phase 3: UCLM Architecture (TTS Mode)

### 4.1 Model Components

```python
# tmrvc_train/models/uclm.py

class UCLM(nn.Module):
    """Unified Codec Language Model."""
    
    def __init__(
        self,
        vocab_size: int = 1024,
        n_codebooks: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_speaker: int = 192,
        d_voice_state: int = 8,
        d_text: int = 256,
    ):
        # Text encoder
        self.text_encoder = TextEncoder(d_text=d_text, d_model=d_model)
        
        # Voice state encoder
        self.voice_state_encoder = VoiceStateEncoder(
            d_state=d_voice_state, d_model=d_model
        )
        
        # Speaker conditioning
        self.speaker_proj = nn.Linear(d_speaker, d_model)
        
        # Mode embedding (TTS=0, VC=1)
        self.mode_embed = nn.Embedding(2, d_model)
        
        # Main transformer (causal)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4),
            num_layers=n_layers,
        )
        
        # Codebook embeddings
        self.codebook_embed = nn.ModuleList([
            nn.Embedding(vocab_size, d_model)
            for _ in range(n_codebooks)
        ])
        
        # Output heads
        self.ar_head = nn.Linear(d_model, vocab_size)  # First codebook (AR)
        self.parallel_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_codebooks - 1)
        ])
        
    def forward(
        self,
        text_features: torch.Tensor,       # [B, L, d_text]
        voice_state: torch.Tensor,         # [B, T, d_state]
        speaker_embed: torch.Tensor,       # [B, d_speaker]
        past_tokens: torch.Tensor | None,  # [B, n_cb, k]
        target_tokens: torch.Tensor,       # [B, n_cb, T] for training
        mode: int = 0,                     # 0=TTS, 1=VC
    ) -> dict[str, torch.Tensor]:
        """Training forward pass."""
        B, T = voice_state.shape[:2]
        
        # Encode conditions
        text_cond = self.text_encoder(text_features)  # [B, L, d_model]
        state_cond = self.voice_state_encoder(voice_state)  # [B, T, d_model]
        spk_cond = self.speaker_proj(speaker_embed).unsqueeze(1)  # [B, 1, d_model]
        mode_cond = self.mode_embed(torch.tensor([mode])).unsqueeze(0)  # [1, 1, d_model]
        
        # Fuse conditions with voice state (frame-level)
        cond = state_cond + spk_cond + mode_cond  # [B, T, d_model]
        
        # Cross-attend with text
        memory = text_cond.transpose(0, 1)  # [L, B, d_model]
        
        # Target embedding (teacher forcing)
        tgt_emb = torch.stack([
            self.codebook_embed[i](target_tokens[:, i, :])
            for i in range(self.n_codebooks)
        ], dim=2).mean(dim=2)  # [B, T, d_model]
        
        tgt = (tgt_emb + cond).transpose(0, 1)  # [T, B, d_model]
        
        # Causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T)
        
        # Transformer
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)  # [T, B, d_model]
        out = out.transpose(0, 1)  # [B, T, d_model]
        
        # Predict
        logits_ar = self.ar_head(out)  # [B, T, vocab_size]
        logits_parallel = torch.stack([
            head(out) for head in self.parallel_heads
        ], dim=1)  # [B, n_cb-1, T, vocab_size]
        
        return {
            "logits_ar": logits_ar,
            "logits_parallel": logits_parallel,
        }
```

### 4.2 Training Loss

```python
def uclm_loss(
    logits_ar: torch.Tensor,        # [B, T, vocab]
    logits_parallel: torch.Tensor,  # [B, n_cb-1, T, vocab]
    target_tokens: torch.Tensor,    # [B, n_cb, T]
) -> torch.Tensor:
    """Multi-codebook loss."""
    
    # AR loss (first codebook)
    loss_ar = F.cross_entropy(
        logits_ar.view(-1, logits_ar.size(-1)),
        target_tokens[:, 0, :].reshape(-1),
    )
    
    # Parallel loss (remaining codebooks)
    loss_parallel = 0
    for i in range(target_tokens.size(1) - 1):
        loss_parallel += F.cross_entropy(
            logits_parallel[:, i].reshape(-1, logits_parallel.size(-1)),
            target_tokens[:, i + 1, :].reshape(-1),
        )
    loss_parallel /= (target_tokens.size(1) - 1)
    
    return loss_ar + loss_parallel
```

### 4.3 Tasks

| Task | File |
|---|---|
| Implement `TextEncoder` | `models/text_encoder.py` (reuse existing) |
| Implement `VoiceStateEncoder` | `models/voice_state_encoder.py` (new) |
| Implement `UCLM` | `models/uclm.py` (new) |
| Implement loss function | `models/uclm.py` |
| Unit tests | `tests/python/test_uclm.py` |

**Duration: 2 weeks**

---

## 5. Phase 4: Training (TTS)

### 5.1 Dataset

```python
# tmrvc_data/uclm_dataset.py

class UCLMDataset(Dataset):
    """Dataset for UCLM training."""
    
    def __getitem__(self, idx):
        utt = self.utterances[idx]
        
        return {
            "phoneme_ids": torch.tensor(utt.phoneme_ids),
            "durations": torch.tensor(utt.durations),
            "codec_tokens": torch.from_numpy(
                np.load(utt.path / "codec_tokens.npy")
            ),
            "voice_state": torch.from_numpy(
                np.load(utt.path / "voice_state.npy")
            ),
            "spk_embed": torch.from_numpy(
                np.load(utt.path / "spk_embed.npy")
            ),
            "text": utt.text,
        }
```

### 5.2 Training Script

```bash
# TTS training
uv run tmrvc-train-uclm \
    --cache-dir data/cache \
    --datasets libritts_r,vctk,expresso \
    --mode tts \
    --batch-size 16 \
    --max-frames 400 \
    --device cuda \
    --max-steps 200000
```

### 5.3 Training Config

```yaml
# configs/train_uclm_tts.yaml
model:
  vocab_size: 1024
  n_codebooks: 8
  d_model: 512
  n_heads: 8
  n_layers: 12

training:
  batch_size: 16
  max_frames: 400
  lr: 1.0e-4
  warmup_steps: 10000
  max_steps: 200000
  gradient_accumulation: 2

data:
  datasets: [libritts_r, vctk, expresso]
  num_workers: 4
```

### 5.4 Tasks

| Task | Description |
|---|---|
| Implement `UCLMDataset` | Load codec tokens, voice state |
| Implement `UCLMTrainer` | Training loop |
| Implement CLI `tmrvc-train-uclm` | Entry point |
| Run initial training | LibriTTS-R subset |
| Evaluate with codec decode | Check audio quality |

**Duration: 2 weeks**

---

## 6. Phase 5: UCLM Architecture (VC Mode)

### 6.1 VC Mode Extension

```python
class UCLM(nn.Module):
    # ... existing code ...
    
    def forward_vc(
        self,
        source_tokens: torch.Tensor,      # [B, n_cb, T]
        voice_state: torch.Tensor,        # [B, T, d_state]
        speaker_embed: torch.Tensor,      # [B, d_speaker]
        past_tokens: torch.Tensor | None, # [B, n_cb, k]
        target_tokens: torch.Tensor,      # [B, n_cb, T] for training
    ) -> dict[str, torch.Tensor]:
        """VC training forward pass."""
        B, n_cb, T = source_tokens.shape
        
        # Encode source tokens
        src_emb = torch.stack([
            self.codebook_embed[i](source_tokens[:, i, :])
            for i in range(n_cb)
        ], dim=2).mean(dim=2)  # [B, T, d_model]
        
        # Encode conditions
        state_cond = self.voice_state_encoder(voice_state)
        spk_cond = self.speaker_proj(speaker_embed).unsqueeze(1)
        mode_cond = self.mode_embed(torch.tensor([1])).unsqueeze(0)  # VC mode
        
        # Fuse
        cond = src_emb + state_cond + spk_cond + mode_cond
        
        # ... rest same as TTS mode ...
```

### 6.2 Tasks

| Task | Description |
|---|---|
| Add `forward_vc` method | Source token conditioning |
| Implement `VCEncoder` (optional) | Alternative to direct token input |
| Unit tests | Test VC mode |

**Duration: 1 week**

---

## 7. Phase 6: Training (VC + Joint)

### 7.1 Multi-Task Training

```python
for batch in dataloader:
    mode = random.choice(['tts', 'vc'])
    
    if mode == 'tts':
        output = model.forward_tts(...)
        loss = uclm_loss(output, batch['target_tokens'])
    else:
        output = model.forward_vc(...)
        loss = uclm_loss(output, batch['target_tokens'])
    
    loss.backward()
```

### 7.2 Tasks

| Task | Description |
|---|---|
| Implement multi-task sampler | Balance TTS/VC batches |
| Run joint training | TTS + VC together |
| Fine-tune on expressive data | Expresso, JVNV, custom |

**Duration: 2 weeks**

---

## 8. Phase 7: Streaming Inference

### 8.1 Block-wise Generation

```python
class StreamingUCLM:
    """Real-time streaming inference."""
    
    def __init__(self, model, codec_decoder, block_size=40):
        self.model = model
        self.codec_decoder = codec_decoder
        self.block_size = block_size  # 400ms
        self.token_buffer = []
        
    def generate_block(
        self,
        text_block: torch.Tensor,
        voice_state_block: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one block of audio."""
        
        # Get past context
        past_tokens = torch.stack(self.token_buffer[-3:], dim=1) if self.token_buffer else None
        
        # Generate tokens
        with torch.no_grad():
            tokens = self.model.generate(
                text=text_block,
                voice_state=voice_state_block,
                speaker=speaker_embed,
                past_tokens=past_tokens,
            )
        
        # Update buffer
        self.token_buffer.append(tokens)
        
        # Decode
        audio = self.codec_decoder.decode(tokens)
        
        return audio
```

### 8.2 ONNX Export

```bash
uv run tmrvc-export-uclm \
    --checkpoint checkpoints/uclm_best.pt \
    --output-dir models/uclm \
    --export-codec-decoder
```

### 8.3 Tasks

| Task | Description |
|---|---|
| Implement `StreamingUCLM` | Block-wise generation |
| Implement ONNX export | UCLM + EnCodec decoder |
| Integrate with tmrvc-engine | C++ streaming inference |
| Latency measurement | Verify <50ms |

**Duration: 2 weeks**

---

## 9. Phase 8: Evaluation & Paper

### 9.1 Evaluation Metrics

| Task | Metrics |
|---|---|
| TTS quality | UTMOS, MOS |
| TTS expressiveness | Emo-SIM, NV-MOS |
| VC quality | SECS, MCD |
| VC speaker | Speaker similarity |
| Latency | E2E ms, RTF |

### 9.2 Paper Submission

Target: **Interspeech 2026** or **ICASSP 2026**

**Duration: 1 week**

---

## 10. Summary Timeline

| Phase | Duration | Key Deliverables |
|---|---|---|
| 1. Data Pipeline | 1 week | Extended `prepare_dataset.py` |
| 2. EnCodec | 1 week | `EnCodecWrapper`, `VoiceStateEstimator` |
| 3. UCLM TTS | 2 weeks | UCLM architecture |
| 4. Training TTS | 2 weeks | Trained TTS model |
| 5. UCLM VC | 1 week | VC mode extension |
| 6. Training Joint | 2 weeks | Unified TTS+VC model |
| 7. Streaming | 2 weeks | Real-time inference |
| 8. Evaluation | 1 week | Paper draft |

**Total: 12 weeks**

---

## 11. Dependencies

| Package | Purpose |
|---|---|
| `transformers` | EnCodec model |
| `torch` | Model implementation |
| `onnx`, `onnxruntime` | ONNX export & inference |

---

## 12. File Structure

```
TMRVC/
├── tmrvc-data/src/tmrvc_data/
│   ├── codec.py              # NEW: EnCodec wrapper
│   ├── voice_state.py        # NEW: Voice state estimator
│   ├── uclm_dataset.py       # NEW: UCLM dataset
│   └── prepare_dataset.py    # MODIFIED: Add codec tokens
│
├── tmrvc-train/src/tmrvc_train/
│   ├── models/
│   │   ├── uclm.py           # NEW: UCLM model
│   │   ├── voice_state_encoder.py  # NEW
│   │   └── text_encoder.py   # EXISTING
│   ├── uclm_trainer.py       # NEW
│   └── cli/train_uclm.py     # NEW
│
├── tmrvc-export/src/tmrvc_export/
│   └── export_uclm.py        # NEW: ONNX export
│
├── configs/
│   └── train_uclm.yaml       # NEW
│
└── docs/design/
    └── unified-codec-lm.md   # NEW: Design document
```
