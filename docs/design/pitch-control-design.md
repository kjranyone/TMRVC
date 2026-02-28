# Pitch Control Design for Singing Voice Conversion

## Overview

This document describes the pitch control architecture for TMRVC, designed to support both speaking and singing voice conversion while maintaining audio quality in the Codec-Latent paradigm.

## Problem Statement

In the Codec-Latent paradigm, applying pitch shift to the **output** audio causes quality degradation because:
1. The token model predicts tokens based on input characteristics
2. The decoder reconstructs audio from tokens with implicit pitch information
3. Post-hoc pitch modification breaks the learned acoustic coherence

For singing VC, we need:
- Melody preservation (vibrato, portamento, timing)
- Optional transposition (key change)
- Voice quality transfer independent of pitch

## Theoretical Foundation

### F0 as Explicit Conditioning

In traditional VC, F0 is implicit in features. For singing VC, we make F0 **explicit**:

```
Source Audio → F0 Extraction → Normalized F0 Contour
                    ↓
            Pitch Shift Parameter
                    ↓
            Target F0 = Source F0 × 2^(shift/12)
```

### Architectural Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    F0-Aware Codec-Latent Pipeline               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Audio In [24000 Hz]                                            │
│     │                                                           │
│     ├──────────────────┐                                        │
│     │                  │                                        │
│     ▼                  ▼                                        │
│  ┌────────────┐   ┌──────────────┐                              │
│  │ F0 Tracker │   │ Codec        │──► Tokens [T, N_CB]         │
│  │ (CREPE)    │   │ Encoder      │                              │
│  └────────────┘   └──────────────┘                              │
│     │                                                          │
│     ▼                                                           │
│  ┌────────────────────────────────────┐                         │
│  │ F0 Processing                       │                         │
│  │                                     │                         │
│  │  f0_raw [T]                         │                         │
│  │     ↓                               │                         │
│  │  f0_normalized = log2(f0 / f0_mean) │                         │
│  │     ↓                               │                         │
│  │  f0_shifted = f0_normalized + shift │◄── pitch_shift param   │
│  │     ↓                               │                         │
│  │  f0_target = f0_mean * 2^f0_shifted │                         │
│  └────────────────────────────────────┘                         │
│     │                                                          │
│     ▼                                                           │
│  ┌───────────────────────────────────────────────────┐          │
│  │ Token Model (F0-Conditioned Transformer)          │          │
│  │                                                   │          │
│  │  Inputs:                                          │          │
│  │  - tokens_in: [1, T, N_CB]   (from encoder)      │          │
│  │  - speaker_embed: [192]      (target speaker)    │          │
│  │  - f0_condition: [1, T, 2]   (normalized + shift)│          │
│  │  - state: [1, S]             (KV cache)          │          │
│  │                                                   │          │
│  │  Output:                                          │          │
│  │  - tokens_out: [1, T, N_CB]  (converted tokens)  │          │
│  │  - state_new: [1, S]         (updated cache)     │          │
│  └───────────────────────────────────────────────────┘          │
│     │                                                          │
│     ▼                                                           │
│  ┌──────────────┐                                               │
│  │ Codec        │──► Audio Out [24000 Hz]                       │
│  │ Decoder      │                                               │
│  └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## F0 Conditioning Details

### F0 Extraction

```python
# CREPE-based F0 extraction (already in preprocessing)
f0 = crepe.predict(audio, sr=24000, step_size=10)  # [T], 100Hz frame rate

# Handle unvoiced frames
f0[f0 == 0] = f0_mean  # or interpolate
```

### F0 Normalization

Normalization makes F0 speaker-invariant:

```python
def normalize_f0(f0: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize F0 to log-scale relative to speaker mean."""
    f0_voiced = f0[f0 > 0]
    f0_mean = np.exp(np.mean(np.log(f0_voiced))) if len(f0_voiced) > 0 else 220.0
    
    # Log-normalized: 0 = speaker mean, ±1 = octave up/down
    f0_norm = np.log2(f0 / f0_mean + 1e-8)
    
    return f0_norm, f0_mean
```

### Pitch Shift Application

```python
def apply_pitch_shift(f0_norm: np.ndarray, shift_semitones: float) -> np.ndarray:
    """Apply pitch shift in semitones to normalized F0."""
    # shift_semitones: +12 = one octave up, -12 = one octave down
    shift_octaves = shift_semitones / 12.0
    return f0_norm + shift_octaves
```

### F0 Conditioning Tensor

```python
def create_f0_condition(f0_norm: np.ndarray, pitch_shift: float) -> np.ndarray:
    """Create F0 conditioning tensor for token model.
    
    Returns:
        [1, T, 2] tensor containing:
        - [:, :, 0]: normalized F0 (original)
        - [:, :, 1]: pitch shift amount (constant per utterance)
    """
    T = len(f0_norm)
    condition = np.zeros((1, T, 2), dtype=np.float32)
    condition[0, :, 0] = f0_norm
    condition[0, :, 1] = pitch_shift  # broadcast to all frames
    return condition
```

## Token Model Architecture (Implementation)

### Current Architecture

`tmrvc-train/src/tmrvc_train/models/token_model.py` は現在:
- FiLM conditioner で speaker embedding を適用
- F0条件付けなし

### Modified Architecture

F0条件付けを追加:

```python
@dataclass
class TokenModelConfig:
    n_codebooks: int = 4
    codebook_size: int = 1024
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    context_length: int = 10
    d_spk: int = 192
    d_f0: int = 2           # [f0_normalized, pitch_shift]
    dropout: float = 0.1


class TokenModel(nn.Module):
    def __init__(self, config: Optional[TokenModelConfig] = None):
        super().__init__()
        self.config = config or TokenModelConfig()
        
        # Token embeddings (existing)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(self.config.codebook_size, self.config.d_model)
            for _ in range(self.config.n_codebooks)
        ])
        
        # Position embedding (existing)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.config.context_length, self.config.d_model) * 0.02
        )
        
        # Speaker projection (existing)
        self.spk_proj = nn.Linear(self.config.d_spk, self.config.d_model)
        
        # NEW: F0 conditioning projection
        self.f0_proj = nn.Linear(self.config.d_f0, self.config.d_model)
        
        # FiLM for speaker (existing)
        self.film = FiLMConditioner(self.config.d_model, self.config.d_model)
        
        # Transformer layers (existing)
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) 
            for _ in range(self.config.n_layers)
        ])
        
        # Output heads (existing)
        self.output_heads = nn.ModuleList([
            nn.Linear(self.config.d_model, self.config.codebook_size)
            for _ in range(self.config.n_codebooks)
        ])
    
    def forward(
        self,
        tokens: torch.Tensor,
        spk_embed: torch.Tensor,
        f0_condition: torch.Tensor,  # NEW: [B, L, 2]
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            tokens: [B, K, L] - context tokens (K=4 codebooks)
            spk_embed: [B, 192] - speaker embedding
            f0_condition: [B, L, 2] - [f0_normalized, pitch_shift] per frame
            kv_caches: optional KV-cache for streaming
        
        Returns:
            logits: [B, K, vocab_size]
            new_kv_caches: updated KV-cache
        """
        B, K, L = tokens.shape
        
        # Token embeddings (sum across codebooks)
        x = torch.zeros(B, L, self.config.d_model, device=tokens.device)
        for i, emb in enumerate(self.token_embeddings):
            x = x + emb(tokens[:, i, :])
        
        # Position embedding
        x = x + self.pos_embedding[:, :L, :]
        
        # Speaker conditioning (FiLM)
        spk_cond = self.spk_proj(spk_embed)
        x = self.film(x, spk_cond)
        
        # NEW: F0 conditioning (additive)
        f0_emb = self.f0_proj(f0_condition)  # [B, L, d_model]
        x = x + f0_emb
        
        # Transformer layers with KV-cache
        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, new_kv = layer(x, kv_caches[i])
            new_kv_caches.append(new_kv)
        
        # Output (last frame only for streaming)
        x_last = x[:, -1, :]
        
        logits = torch.stack([head(x_last) for head in self.output_heads], dim=1)
        
        return logits, new_kv_caches
```

### ONNX Export Changes

`tmrvc-export/src/tmrvc_export/export_onnx.py`:

```python
class TokenModelONNX(nn.Module):
    def __init__(self, model: TokenModel):
        self.model = model
    
    def forward(self, tokens, spk_embed, f0_condition, kv_cache_flat):
        kv_caches = self.model.flat_to_kv_cache(kv_cache_flat)
        logits, new_kv_caches = self.model(tokens, spk_embed, f0_condition, kv_caches)
        kv_cache_out_flat = self.model.kv_cache_to_flat(new_kv_caches)
        return logits, kv_cache_out_flat

# Export
dummy_tokens = torch.randint(0, 1024, (1, 4, 10))
dummy_spk = torch.randn(1, 192)
dummy_f0 = torch.randn(1, 10, 2)  # NEW input
dummy_kv = torch.zeros(12, 1, 4, 10, 64)

torch.onnx.export(
    TokenModelONNX(model),
    (dummy_tokens, dummy_spk, dummy_f0, dummy_kv),
    "token_model.onnx",
    input_names=["tokens_in", "spk_embed", "f0_condition", "kv_cache_in"],
    output_names=["logits", "kv_cache_out"],
    dynamic_axes={"f0_condition": {1: "L"}},
)
```

## Training Strategy

### Data Augmentation

Train with pitch-shifted versions to learn pitch invariance:

```python
def augment_with_pitch_shift(audio, f0, shift_range=(-12, 12)):
    """Augment audio with random pitch shift."""
    shift = np.random.uniform(*shift_range)
    
    # PSOLA-based pitch shift for training data
    audio_shifted = psola_pitch_shift(audio, shift)
    f0_shifted = f0 * (2 ** (shift / 12))
    
    return audio_shifted, f0_shifted, shift
```

### Training Objective

```python
def training_step(batch):
    # Unpack
    audio = batch['audio']
    f0 = batch['f0']
    speaker_embed = batch['speaker_embed']
    target_tokens = batch['target_tokens']
    pitch_shift = batch['pitch_shift']  # augmentation parameter
    
    # Forward
    tokens_enc = codec_encoder(audio)
    f0_norm, f0_mean = normalize_f0(f0)
    f0_condition = create_f0_condition(f0_norm, pitch_shift)
    
    tokens_pred, _ = token_model(
        tokens_enc, 
        speaker_embed, 
        f0_condition
    )
    
    # Loss: cross-entropy per codebook
    loss = sum(
        F.cross_entropy(pred.view(-1, vocab_size), target.view(-1))
        for pred, target in zip(tokens_pred, target_tokens.unbind(-1))
    )
    
    return loss
```

## Inference Pipeline

### Real-time Streaming

```rust
// Rust streaming processor with F0 conditioning
pub struct StreamingProcessorWithF0 {
    codec_encoder: OrtSession,
    token_model: OrtSession,
    codec_decoder: OrtSession,
    f0_tracker: CrepeTracker,  // Lightweight CREPE for streaming
    f0_buffer: Vec<f32>,
    f0_mean: f32,
    pitch_shift: f32,
    // ... other state
}

impl StreamingProcessorWithF0 {
    pub fn process_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        // 1. Track F0 (CREPE-lite on current frame)
        let f0 = self.f0_tracker.predict(frame);
        self.f0_buffer.push(f0);
        
        // 2. Encode audio
        let tokens = self.codec_encoder.run(frame)?;
        
        // 3. Create F0 conditioning
        let f0_norm = (f0 / self.f0_mean).log2();
        let f0_condition = [f0_norm, self.pitch_shift];
        
        // 4. Token model with F0 conditioning
        let tokens_out = self.token_model.run(tokens, &f0_condition)?;
        
        // 5. Decode
        let audio_out = self.codec_decoder.run(tokens_out)?;
        
        audio_out
    }
}
```

### Latency Budget

| Component | Latency |
|-----------|---------|
| F0 Tracker (CREPE-lite) | ~3ms |
| Codec Encoder | ~8ms |
| Token Model | ~12ms |
| Codec Decoder | ~10ms |
| **Total** | **~33ms** |

## MVP Strategy

### Phase 1: No Pitch Control (Current)
- Remove pitch_shift parameter from VST
- Document as future enhancement
- Focus on voice quality

### Phase 2: F0-Conditioned Token Model
- Implement F0 extraction in preprocessing
- Modify token model architecture
- Retrain with F0 conditioning
- Add pitch_shift parameter

### Phase 3: Singing VC Optimization
- Optimize F0 tracker for streaming (CREPE-lite)
- Add vibrato preservation
- Add formant adjustment (independent of pitch)

## Parameter Design

### VST Parameters

```rust
pub enum Param {
    // ... existing params
    PitchShift,      // -24 to +24 semitones, default 0
    FormantShift,    // -12 to +12 semitones, default 0 (Phase 3)
    PitchSmoothing,  // 0 to 100ms, default 20ms (for stability)
}
```

### API Parameters

```json
{
    "pitch_shift": 0.0,      // semitones, -24 to +24
    "formant_shift": 0.0,    // semitones, -12 to +12 (Phase 3)
    "pitch_smoothing_ms": 20 // smoothing window for F0
}
```

## Open Questions

1. **F0 Tracker Latency**: CREPE is accurate but slow. For streaming, consider:
   - CREPE-lite (reduced model size)
   - PYIN with lookahead buffering
   - Hybrid approach (CREPE for voiced detection, simple correlation for F0)

2. **Pitch Drift Handling**: For long singing phrases, F0 mean may drift:
   - Use rolling window for normalization
   - Or use fixed reference pitch from speaker profile

3. **Unvoiced Frames**: How to handle unvoiced frames in F0 conditioning:
   - Interpolate from neighboring voiced frames
   - Use special token for unvoiced
   - Let model learn to ignore F0 during unvoiced

## References

- CREPE: A Convolutional Representation for Pitch Estimation (Kim et al., 2018)
- Differentiable Digital Signal Processing (DDSP) (Engel et al., 2020)
- So-VITS-SVC F0 conditioning approach
- RVC (Retrieval-based Voice Conversion) pitch design
