# UCLM Implementation Roadmap

Kojiro Tanaka — UCLM Implementation Plan
Created: 2026-02-27 (Asia/Tokyo)
Updated: 2026-03-01 — Token Spec v2 (dual-stream) sync

> **Goal:** UCLM (Unified Codec Language Model) の実装。
> **優先順位:** TTS 優先、その後 VC。**既存資産は完全置き換え。**
> **v2 Contract:** `A_t` (acoustic RVQ) + `B_t` (control tuple) を 10ms 単位で同時生成。

---

## 1. Overview

```
Phase 1: Data Pipeline (UCLM用拡張)
Phase 2: EnCodec Fine-Tuning & SSL Voice State Integration
Phase 3: UCLM Architecture (TTS mode & Disentanglement)
Phase 4: Training (TTS)
Phase 5: UCLM Architecture (VC mode)
Phase 6: Training (VC + joint, dual-stream)
Phase 7: Streaming Inference (rolling A/B context + delta state)
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
raw_audio → normalize → extract_features → annotate → encode_codec/control → save
                ↓              ↓              ↓                   ↓
            24kHz/mono    mel/f0/spk_embed  text/emotion     A_t / B_t tokens
                                                ↓                   ↓
                                   voice_state (estimated)   delta_voice_state
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
├── acoustic_tokens.npy  # [8, T] - NEW (A_t)
├── control_tokens.npy   # [4, T] - NEW (B_t = [op,type,dur,int])
├── voice_state.npy      # [T, 8] - NEW
├── delta_voice_state.npy # [T, 8] - NEW
├── phoneme_ids.npy      # [L] - NEW
└── durations.npy        # [L] - NEW
```

### 2.5 Implementation Tasks

| Task | File | Description |
|---|---|---|
| Add codec encoder | `tmrvc_data/codec.py` | Acoustic token extraction (`A_t`) |
| Add control tokenizer | `tmrvc_data/control_tokens.py` | Event tuple extraction (`B_t`) |
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
        self.rvq_vocab_size = 1024
        
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_samples] at 24kHz
        Returns:
            tokens: [B, n_codebooks, T_frames] at 100fps (10ms/frame), IDs in 0..1023
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

### 3.2 SSL Voice State Estimator

```python
# tmrvc_data/voice_state.py

class SSLVoiceStateEstimator(nn.Module):
    """Extract explicit parameters + latent SSL style space."""
    
    def __init__(self):
        self.explicit_estimator = VoiceStateEstimator() # 8-dim heuristic
        self.wavlm_extractor = WavLMFeatureExtractor(d_output=128)
        
    def forward(self, audio_16k: torch.Tensor, audio_24k: torch.Tensor, mel: torch.Tensor, f0: torch.Tensor) -> dict:
        """
        Args:
            audio_16k: [B, T_16k] for WavLM
            audio_24k: [B, T_24k] for frame alignment
            mel: [B, 80, T]
            f0: [B, 1, T]
        Returns:
            explicit_state: [B, T, 8]
            ssl_state: [B, T, 128] (from WavLM)
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
| Install EnCodec & WavLM | `uv add transformers` |
| Fine-tune EnCodec Decoder | Focus on high-frequency / breathiness (Intimate/Expresso data) |
| Implement `EnCodecWrapper` | Token extraction and decoding |
| Implement `SSLVoiceStateEstimator`| Extract latent style space via WavLM |
| Implement `VoiceStateEstimator` | Frame-level acoustic params (8-dim) |
| Test on sample audio | Verify token quality |

**Duration: 1 week**

---

## 4. Phase 3: UCLM Architecture (TTS Mode)

### 4.0 v2 変更点 (必須)

- 単一 `target_tokens` ではなく `target_A`, `target_B` を使用
- 出力 head は `acoustic_heads (8 x 1024)` + `control_heads (4 x 64)`
- `delta_voice_state` を forward 入力に追加
- loss は `L = L_A + L_B + L_delta (+ optional disentangle losses)`

### 4.1 Model Components

```python
# tmrvc_train/models/uclm.py

class UCLM(nn.Module):
    """Unified Codec LM (dual-stream token prediction)."""

    def __init__(
        self,
        rvq_vocab_size: int = 1024,
        control_vocab_size: int = 64,
        n_codebooks: int = 8,
        d_model: int = 512,
        d_speaker: int = 192,
        d_voice_state: int = 8,
    ):
        self.backbone = CausalTransformer(d_model=d_model)
        self.voice_state_encoder = VoiceStateEncoder(d_state=d_voice_state, d_model=d_model)
        self.speaker_proj = nn.Linear(d_speaker, d_model)

        self.acoustic_heads = nn.ModuleList([
            nn.Linear(d_model, rvq_vocab_size) for _ in range(n_codebooks)
        ])
        self.control_heads = nn.ModuleList([
            nn.Linear(d_model, control_vocab_size) for _ in range(4)  # [op,type,dur,int]
        ])

    def forward(
        self,
        cond_features: torch.Tensor,       # [B, T, d_model] (text or VC condition)
        voice_state: torch.Tensor,         # [B, T, 8]
        delta_voice_state: torch.Tensor,   # [B, T, 8]
        speaker_embed: torch.Tensor,       # [B, 192]
        past_a: torch.Tensor | None,       # [B, 8, k]
        past_b: torch.Tensor | None,       # [B, 4, k]
    ) -> dict[str, torch.Tensor]:
        state_cond = self.voice_state_encoder(voice_state + delta_voice_state)
        spk_cond = self.speaker_proj(speaker_embed).unsqueeze(1)
        feat = self.backbone(cond_features + state_cond + spk_cond, past_a=past_a, past_b=past_b)

        logits_a = torch.stack([head(feat) for head in self.acoustic_heads], dim=1)
        logits_b = torch.stack([head(feat) for head in self.control_heads], dim=1)
        return {"logits_a": logits_a, "logits_b": logits_b}
```

### 4.2 Training Loss

```python
def uclm_loss(
    logits_a: torch.Tensor,         # [B, 8, T, 1024]
    logits_b: torch.Tensor,         # [B, 4, T, 64]
    target_a: torch.Tensor,         # [B, 8, T]
    target_b: torch.Tensor,         # [B, 4, T]
    delta_voice_state: torch.Tensor,# [B, T, 8]
    pred_delta: torch.Tensor,       # [B, T, 8]
) -> torch.Tensor:
    """Dual-stream loss for UCLM v2."""

    loss_a = sum(
        F.cross_entropy(logits_a[:, i].reshape(-1, logits_a.size(-1)), target_a[:, i, :].reshape(-1))
        for i in range(8)
    ) / 8.0

    loss_b = sum(
        F.cross_entropy(logits_b[:, i].reshape(-1, logits_b.size(-1)), target_b[:, i, :].reshape(-1))
        for i in range(4)
    ) / 4.0

    loss_delta = F.l1_loss(pred_delta, delta_voice_state)
    return loss_a + loss_b + 0.1 * loss_delta
```

### 4.3 Tasks

| Task | File |
|---|---|
| Implement `TextEncoder` | `models/text_encoder.py` (reuse existing) |
| Implement `VoiceStateEncoder` | `models/voice_state_encoder.py` (new) |
| Implement `ControlTokenizer` | `data/control_tokens.py` (new) |
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
            "acoustic_tokens": torch.from_numpy(
                np.load(utt.path / "acoustic_tokens.npy")
            ),
            "control_tokens": torch.from_numpy(
                np.load(utt.path / "control_tokens.npy")
            ),
            "voice_state": torch.from_numpy(
                np.load(utt.path / "voice_state.npy")
            ),
            "delta_voice_state": torch.from_numpy(
                np.load(utt.path / "delta_voice_state.npy")
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
    --datasets libritts_r vctk expresso \
    --batch-size 16 \
    --max-frames 400 \
    --device cuda \
    --max-steps 200000
```

### 5.3 Training Config

```yaml
# configs/train_uclm_tts.yaml
model:
  rvq_vocab_size: 1024
  control_vocab_size: 64
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
| Implement `UCLMDataset` | Load `acoustic_tokens`, `control_tokens`, `voice_state`, `delta_voice_state` |
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
        source_a: torch.Tensor,           # [B, n_cb, T]
        source_b: torch.Tensor,           # [B, 4, T]
        voice_state: torch.Tensor,        # [B, T, d_state]
        delta_voice_state: torch.Tensor,  # [B, T, d_state]
        speaker_embed: torch.Tensor,      # [B, d_speaker]
        past_a: torch.Tensor | None,      # [B, n_cb, k]
        past_b: torch.Tensor | None,      # [B, 4, k]
        target_a: torch.Tensor,           # [B, n_cb, T] for training
        target_b: torch.Tensor,           # [B, 4, T] for training
    ) -> dict[str, torch.Tensor]:
        """VC training forward pass."""
        B, n_cb, T = source_a.shape
        
        # Encode source tokens
        src_emb = torch.stack([
            self.codebook_embed[i](source_a[:, i, :])
            for i in range(n_cb)
        ], dim=2).mean(dim=2)  # [B, T, d_model]
        
        # Encode conditions
        state_cond = self.voice_state_encoder(voice_state + delta_voice_state)
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
        loss = uclm_loss(
            output['logits_a'], output['logits_b'],
            batch['target_a'], batch['target_b'],
            batch['delta_voice_state'], output['pred_delta']
        )
    else:
        output = model.forward_vc(...)
        loss = uclm_loss(
            output['logits_a'], output['logits_b'],
            batch['target_a'], batch['target_b'],
            batch['delta_voice_state'], output['pred_delta']
        )
    
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
        delta_voice_state_block: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one block of audio."""
        
        # Get past context (dual-stream)
        past_a, past_b = self.get_past_context()
        
        # Generate tokens
        with torch.no_grad():
            tokens_a, tokens_b = self.model.generate(
                text=text_block,
                voice_state=voice_state_block,
                delta_voice_state=delta_voice_state_block,
                speaker=speaker_embed,
                past_a=past_a,
                past_b=past_b,
            )
        
        # Update buffer
        self.push_context(tokens_a, tokens_b)
        
        # Decode
        audio = self.codec_decoder.decode(tokens_a, tokens_b, voice_state_block, delta_voice_state_block)
        
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
| Implement ONNX export | `uclm_core` dual-head + codec decoder |
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
│   ├── control_tokens.py     # NEW: Event tuple tokenizer
│   ├── voice_state.py        # NEW: Voice state estimator
│   ├── uclm_dataset.py       # NEW: UCLM dataset
│   └── prepare_dataset.py    # MODIFIED: Add A/B tokens
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
