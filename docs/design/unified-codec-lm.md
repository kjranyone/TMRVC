# Unified Codec Language Model for Real-Time TTS and VC

Kojiro Tanaka — Unified Architecture Design
Created: 2026-02-27 (Asia/Tokyo)
Updated: 2026-03-01 — Token Spec v2 dual-stream (`A_t`/`B_t`)

> **Core Insight:** テキスト生成（TTS）も音声変換（VC）も、**「条件付けされた codec token 変換」** として統一的に扱う。単一の CodecLM で両モードをカバーし、50ms 以下のストリーミング推論を実現。

---

## 1. Abstract

We propose **Unified Codec Language Model (UCLM)**, a single neural architecture that performs both text-to-speech synthesis and voice conversion through conditioned codec token transformation. Unlike pipeline-based approaches that separate TTS and VC into distinct systems, UCLM treats both as instances of the same fundamental task: generating acoustic tokens conditioned on input modality (text or source audio), speaker identity, and optional voice state parameters.

Key innovations:
1. **Unified TTS/VC**: Same model performs both tasks via mode conditioning.
2. **Disentangled Representations**: Vector Quantization (VQ) bottleneck and Gradient Reversal Layer (GRL) to perfectly separate content, style, and speaker.
3. **Learning-based Voice State**: Frame-level acoustic control powered by SSL (WavLM) latent space and explicit 8-dim parameters.
4. **Real-time streaming & CFG**: Block-wise generation (<50ms) with Classifier-Free Guidance (CFG) for amplified sensual and emotional expression.

Experiments demonstrate state-of-the-art performance on both TTS (including non-verbal vocalizations) and VC (including speaker adaptation), while enabling seamless transitions between speaking modes in a single generation stream.

---

## 1.1 Token Spec v2 Sync

This document follows the canonical spec in `emotion-aware-codec.md`.

- Acoustic stream `A_t`: `[B, 8]`, RVQ IDs `0..1023` (`RVQ_VOCAB_SIZE=1024`)
- Control stream `B_t`: `[B, 4]=[op, type, dur, int]` (`CONTROL_VOCAB_SIZE=64`)
- Frame unit: 10ms (`240 samples @ 24kHz`)
- Continuity: rolling context (`A/B` history) + `delta_voice_state`

---

## 2. Unified Formulation

### 2.1 Core Idea

Both TTS and VC can be formulated as:

```
P(A_t, B_t | input, speaker, voice_state_t, delta_voice_state_t, past_A, past_B)
```

| Mode | Input | Output |
|---|---|---|
| **TTS** | Text (phonemes) | `A_t` + `B_t` |
| **VC** | Source `A_t` + state | Target `A_t` + `B_t` |
| **TTS+VC** | Text + reference audio | Speech with unified token control |

The difference is only in the input modality; the core generation process is identical.

### 2.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Unified Codec Language Model (UCLM)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   TTS Encoder   │  │   VC Encoder    │  │  Voice State    │          │
│  │  (TextEncoder)  │  │ (+ VQ Bottleneck│  │  (+ SSL / GRL)  │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │                    │
│           └──────────────┬─────┴────────────────────┘                    │
│                          │                                               │
│                          ▼                                               │
│              ┌───────────────────────┐                                   │
│              │    Modality Fusion    │                                   │
│              │  (cross-attention)    │                                   │
│              └───────────┬───────────┘                                   │
│                          │                                               │
│         ┌────────────────┼────────────────┐                             │
│         │                │                │                              │
│         ▼                ▼                ▼                              │
│    ┌─────────┐    ┌─────────────┐   ┌──────────┐                        │
│    │  Text   │    │   Speaker   │   │  Past    │                        │
│    │  Cond   │    │   Embed     │   │ Context  │                        │
│    └────┬────┘    └──────┬──────┘   └────┬─────┘                        │
│         │                │               │                               │
│         └────────────────┼───────────────┘                               │
│                          │                                               │
│                          ▼                                               │
│              ┌───────────────────────┐                                   │
│              │   Codec Transformer   │                                   │
│              │  (causal, streaming)  │                                   │
│              └───────────┬───────────┘                                   │
│                          │                                               │
│                          ▼                                               │
│              ┌───────────────────────┐                                   │
│              │  Dual Token Heads     │                                   │
│              │  Acoustic A_t + B_t   │                                   │
│              └───────────┬───────────┘                                   │
│                          │                                               │
│                          ▼                                               │
│              ┌───────────────────────┐                                   │
│              │    Codec Decoder      │                                   │
│              │   (EnCodec / DAC)     │                                   │
│              └───────────┬───────────┘                                   │
│                          │                                               │
│                          ▼                                               │
│                      Waveform                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mode-Specific Components

### 3.1 TTS Mode

```python
class TTSEncoder(nn.Module):
    """Encode text (phonemes) for TTS mode."""
    
    def __init__(self, vocab_size, d_model=256, n_layers=6):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(...)
        
    def forward(self, phonemes, durations=None):
        """
        Args:
            phonemes: [B, L] phoneme IDs
            durations: [B, L] optional frame durations (for alignment)
        Returns:
            text_features: [B, L, d_model] or [B, T, d_model] if durations given
        """
        x = self.embedding(phonemes)
        x = self.transformer(x)
        
        if durations is not None:
            x = self.length_regulate(x, durations)
        
        return x
```

### 3.2 VC Mode

```python
class VCEncoder(nn.Module):
    """Encode source audio tokens for VC mode with Information Bottleneck."""
    
    def __init__(self, n_codebooks=8, vocab_size=1024, d_model=256, vq_bins=128):
        self.codebook_embed = nn.ModuleList([
            nn.Embedding(vocab_size, d_model // n_codebooks)
            for _ in range(n_codebooks)
        ])
        self.source_transformer = nn.TransformerEncoder(...)
        # NEW: Information Bottleneck to remove speaker/style from source
        self.vq_bottleneck = VectorQuantizer(vq_bins, d_model)
        
    def forward(self, source_tokens):
        """
        Args:
            source_tokens: [B, n_codebooks, T] from EnCodec
        Returns:
            source_features: [B, T, d_model]
            vq_loss: Commitment loss for bottleneck
        """
        embeddings = [
            embed(source_tokens[:, i, :])
            for i, embed in enumerate(self.codebook_embed)
        ]
        x = torch.cat(embeddings, dim=-1)
        x = self.source_transformer(x)
        
        # Apply Information Bottleneck
        x, vq_loss, _ = self.vq_bottleneck(x)
        return x, vq_loss
```

### 3.3 F0 Conditioning (Singing VC)

For singing voice conversion, F0 is provided as explicit conditioning:

```python
class F0Conditioner(nn.Module):
    """F0 conditioning for pitch control in singing VC."""
    
    def __init__(self, d_f0=2, d_model=256):
        self.d_f0 = d_f0
        self.proj = nn.Linear(d_f0, d_model)
    
    def forward(self, f0_condition):
        """
        Args:
            f0_condition: [B, T, 2] tensor containing:
                - [:, :, 0]: f0_normalized = log2(f0 / f0_mean)
                - [:, :, 1]: pitch_shift in semitones
        Returns:
            f0_features: [B, T, d_model]
        """
        return self.proj(f0_condition)

def normalize_f0(f0: np.ndarray, f0_mean: float) -> np.ndarray:
    """Normalize F0 relative to speaker mean."""
    return np.log2(f0 / f0_mean + 1e-8)

def apply_pitch_shift(f0_norm: np.ndarray, shift_semitones: float) -> np.ndarray:
    """Apply pitch shift in semitones."""
    return f0_norm + shift_semitones / 12.0
```

**F0 Extraction (Rust side):**
- Autocorrelation-based F0 detection in `f0_tracker.rs`
- F0 mean loaded from `.tmrvc_speaker` file
- Real-time pitch shift parameter from VST/RT GUI

**Integration in Token Model:**
```python
# In TokenModel.forward()
x = token_embeddings(tokens) + pos_embedding
x = film(x, spk_cond)  # Speaker conditioning
x = x + f0_proj(f0_condition)  # F0 conditioning (additive)
```

### 3.4 Voice State Encoder (with SSL & GRL)

```python
class VoiceStateEncoder(nn.Module):
    """Encode style using SSL representations and explicit continuous parameters.
       Includes Gradient Reversal Layer (GRL) for perfect disentanglement."""
    
    # Voice state explicit dimensions (8-dim):
    # breathiness, tension, arousal, valence, roughness, voicing, energy, rate
    
    def __init__(self, d_state=8, d_ssl=128, d_model=256):
        self.explicit_proj = nn.Linear(d_state, d_model // 2)
        self.ssl_proj = nn.Linear(d_ssl, d_model // 2)
        self.fusion = nn.Linear(d_model, d_model)
        self.temporal_conv = CausalConv1d(d_model, d_model, kernel_size=5)
        
        # GRL Adversarial Classifier (predicts speaker/text to unlearn them)
        self.adversarial_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers + num_phonemes)
        )
        
    def forward(self, voice_state_explicit, voice_state_ssl):
        """
        Args:
            voice_state_explicit: [B, T, 8] manual/heuristic parameters
            voice_state_ssl: [B, T, 128] from WavLM VAE latent space
        Returns:
            state_features: [B, T, d_model]
            adv_logits: For adversarial loss
        """
        x_exp = self.explicit_proj(voice_state_explicit)
        x_ssl = self.ssl_proj(voice_state_ssl)
        x = torch.cat([x_exp, x_ssl], dim=-1)
        x = self.fusion(x)
        
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        
        adv_logits = self.adversarial_classifier(x)
        return x, adv_logits
```

### 3.5 Modality Fusion

```python
class ModalityFusion(nn.Module):
    """Fuse TTS and VC modalities with mode conditioning."""
    
    def __init__(self, d_model=256):
        self.mode_embed = nn.Embedding(2, d_model)  # 0=TTS, 1=VC
        self.fusion = nn.MultiheadAttention(d_model, num_heads=4)
        
    def forward(self, text_features, source_features, mode):
        """
        Args:
            text_features: [B, L_text, d] or None
            source_features: [B, T, d] or None
            mode: 0 for TTS, 1 for VC
        Returns:
            fused: [B, T, d]
        """
        mode_token = self.mode_embed(torch.tensor([mode]))
        
        if mode == 0:  # TTS
            # Cross-attend: source_features is actually past_context
            # text_features drives generation
            query = text_features
            key = source_features if source_features is not None else text_features
        else:  # VC
            # Source features are primary
            query = source_features
            key = source_features
            
        fused, _ = self.fusion(query + mode_token, key, key)
        return fused
```

---

## 4. Codec Language Model

### 4.1 Dual-Stream Token Prediction

```python
class CodecTransformer(nn.Module):
    """Predict A_t (acoustic) and B_t (control) each frame."""
    
    def __init__(self, n_codebooks=8, rvq_vocab=1024, ctrl_vocab=64, d_model=512):
        self.n_codebooks = n_codebooks
        self.rvq_vocab = rvq_vocab
        self.ctrl_vocab = ctrl_vocab
        
        self.backbone = CausalTransformer(...)
        self.acoustic_heads = nn.ModuleList([
            nn.Linear(d_model, rvq_vocab) for _ in range(n_codebooks)
        ])
        self.control_heads = nn.ModuleList([
            nn.Linear(d_model, ctrl_vocab) for _ in range(4)  # [op, type, dur, int]
        ])
        
    def forward(self, cond, past_a=None, past_b=None):
        """
        Args:
            cond: [B, T, d_model]
            past_a: [B, 8, k] acoustic context
            past_b: [B, 4, k] control context
        Returns:
            logits_a: [B, 8, T, 1024]
            logits_b: [B, 4, T, 64]
        """
        feat = self.backbone(cond, past_a=past_a, past_b=past_b)

        logits_a = torch.stack([head(feat) for head in self.acoustic_heads], dim=1)
        logits_b = torch.stack([head(feat) for head in self.control_heads], dim=1)
        return logits_a, logits_b
```

### 4.2 Streaming Generation

```python
class StreamingCodecLM(nn.Module):
    """Block-wise streaming generation."""
    
    def __init__(self, block_size=40, context_blocks=2):
        self.block_size = block_size  # frames per block (400ms at 10ms/frame)
        self.context_blocks = context_blocks  # how many past blocks to keep
        
        self.codec_lm = CodecTransformer(...)
        self.codec_decoder = EnCodecDecoder(...)  # or DAC
        
    def generate_block(self, cond, past_a, past_b, voice_state_block, delta_state_block):
        """
        Generate one block of audio.
        
        Args:
            cond: [B, L, d] text or source conditioning
            voice_state_block: [B, block_size, d_state] voice state for this block
            past_a: [B, 8, k]
            past_b: [B, 4, k]
        Returns:
            audio_block: [B, 1, block_size * hop_length] waveform
            new_a: [B, 8, block_size]
            new_b: [B, 4, block_size]
        """
        logits_a, logits_b = self.codec_lm(cond, past_a, past_b)
        new_a = sample_a(logits_a)
        new_b = sample_b(logits_b)
        
        audio = self.codec_decoder(new_a, new_b, voice_state_block, delta_state_block)
        
        return audio, new_a, new_b
```

---

## 5. Real-Time VC Pipeline

### 5.1 Streaming VC Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Real-Time VC with UCLM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Audio (24kHz)                                                 │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐                                                     │
│  │  EnCodec    │  Encode in 10ms frames                              │
│  │  Encoder    │  audio[B,1,240] → tokens[B,8,1]                     │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Token Buffer                              │    │
│  │  Store last K tokens for context                            │    │
│  │  [token_{t-K}, ..., token_{t-1}]                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              UCLM (VC Mode)                                  │    │
│  │                                                              │    │
│  │  Input:                                                      │    │
│  │    • source_A_t: current acoustic tokens                     │    │
│  │    • target_spk_embed: from .tmrvc_speaker file             │    │
│  │    • voice_state: optional (from IR estimator or manual)    │    │
│  │    • delta_voice_state: frame delta                          │    │
│  │    • past_context: A/B tokens from rolling buffer            │    │
│  │                                                              │    │
│  │  Output: target_A_t[B, 8, 1], target_B_t[B, 4, 1]           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐                                                     │
│  │  EnCodec    │  Decode tokens to waveform                        │
│  │  Decoder    │  tokens[B,8,1] → audio[B,1,240]                   │
│  └──────┬──────┘                                                     │
│         │                                                            │
│         ▼                                                            │
│  Output Audio (24kHz)                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Latency Budget (VC)

| Component | Time (ms) |
|---|---|
| EnCodec encode | ~3 |
| Token buffer update | ~0.5 |
| UCLM inference (1 frame) | ~8 |
| EnCodec decode | ~5 |
| Overlap-add buffer | ~10 |
| **Total** | **~26.5** |

### 5.3 Frame-Level vs Block-Level VC

**Option A: Frame-Level (10ms)**
- Lowest latency (~27ms)
- Each frame processed independently
- May have discontinuities

**Option B: Block-Level (40ms)**
- Slightly higher latency (~35ms)
- 4 frames processed together
- Better temporal coherence

```python
class RealTimeVC:
    """Real-time voice conversion with UCLM."""
    
    def __init__(self, block_size=4):  # 4 frames = 40ms
        self.block_size = block_size
        self.encoder = EnCodecEncoder()
        self.decoder = EnCodecDecoder()
        self.uclm = UCLM()
        
        self.token_buffer_a = TokenBuffer(max_blocks=6)
        self.token_buffer_b = TokenBuffer(max_blocks=6)
        self.audio_buffer = AudioBuffer(hop_length=240)
        
    def process_frame(self, audio_frame, target_spk_embed, voice_state=None):
        """
        Process one 10ms frame.
        
        Args:
            audio_frame: [1, 240] 16-bit PCM at 24kHz
            target_spk_embed: [192] from .tmrvc_speaker
            voice_state: [8] optional voice state params
        Returns:
            output_frame: [1, 240] converted audio
        """
        source_a = self.encoder(audio_frame)
        
        past_a = self.token_buffer_a.get_context()
        past_b = self.token_buffer_b.get_context()
        delta_state = voice_state - self.prev_voice_state
        
        target_a, target_b = self.uclm(
            source_a=source_a,
            target_spk=target_spk_embed,
            voice_state=voice_state,
            delta_voice_state=delta_state,
            past_a=past_a,
            past_b=past_b,
            mode='vc'
        )
        
        self.token_buffer_a.append(target_a)
        self.token_buffer_b.append(target_b)
        self.prev_voice_state = voice_state

        output_audio = self.decoder(target_a, target_b, voice_state, delta_state)
        
        return output_audio
```

---

## 6. Training Strategy

### 6.1 Multi-Task Training

```python
# Training loop pseudocode
for batch in dataloader:
    mode = random.choice(['tts', 'vc'])
    
    if mode == 'tts':
        # TTS task
        text = batch['phonemes']
        audio = batch['audio']
        speaker = batch['speaker_embed']
        voice_state = batch['voice_state']  # extracted or manual
        
        target_a = codec_encoder(audio)
        target_b = control_tokenizer(audio, voice_state)
        
        # Predict
        pred_a, pred_b = uclm(
            text=text,
            speaker=speaker,
            voice_state=voice_state,
            delta_voice_state=compute_delta(voice_state),
            past_a=None,
            past_b=None,
            mode='tts'
        )
        
    else:  # vc
        # VC task
        source_audio = batch['source_audio']
        target_audio = batch['target_audio']
        target_speaker = batch['target_speaker_embed']
        voice_state = batch['voice_state']
        
        # Encode both
        source_a = codec_encoder(source_audio)
        target_a = codec_encoder(target_audio)
        target_b = control_tokenizer(target_audio, voice_state)
        
        # Predict
        pred_a, pred_b = uclm(
            source_a=source_a,
            speaker=target_speaker,
            voice_state=voice_state,
            delta_voice_state=compute_delta(voice_state),
            past_a=None,
            past_b=None,
            mode='vc'
        )
    
    # Classifier-Free Guidance (CFG) Dropout
    if random.random() < 0.15:
        voice_state = torch.zeros_like(voice_state)
        target_speaker = torch.zeros_like(target_speaker)

    # Loss (dual-stream + VQ + Adversarial)
    loss_uclm = ce_acoustic(pred_a, target_a) + ce_control(pred_b, target_b)
    loss_vq = vq_bottleneck_loss(batch)
    loss_adv = adversarial_disentanglement_loss(pred_adv_logits, batch['speaker'], batch['text'])
    
    loss = loss_uclm + loss_vq + loss_adv
    loss.backward()
```

### 6.2 Curriculum

| Phase | Tasks | Data |
|---|---|---|
| **1** | TTS only (clean) | LibriTTS-R, JSUT |
| **2** | VC only | VCTK, JVS |
| **3** | TTS + VC joint | Combined |
| **4** | + Voice state | Expresso, JVNV |
| **5** | + Non-verbal | Custom intimate data |
| **6** | Streaming fine-tune | All (causal masking) |

---

## 7. Integration with TMRVC

### 7.1 Replacing Current Architecture

**Before (5 models):**
```
ContentEncoder → Converter → Vocoder
SpeakerEncoder (offline)
IREstimator (100ms)
```

**After (3 models + offline speaker):**
```
CodecEncoder → UCLM Core (dual-head) → CodecDecoder
SpeakerEncoder (offline, reused)
```

### 7.2 Compatibility

| Feature | Current | UCLM |
|---|---|---|
| `.tmrvc_speaker` format | ✓ (spk_embed only) | ✓ (same) |
| IR robustness | ✓ (IR-robust training) | ✓ (data augmentation) |
| Few-shot adaptation | ✓ (LoRA) | ✓ (in-context or LoRA) |
| Causal streaming | ✓ (50ms) | ✓ (~30ms) |
| Emotion control | △ (style encoder) | ✓ (voice_state) |

### 7.3 Migration Path

1. **Phase 1**: UCLM for TTS only (keep current VC)
2. **Phase 2**: UCLM for VC (parallel run with current VC)
3. **Phase 3**: Deprecate ContentEncoder/Converter/Vocoder

---

## 8. Novelty Summary

| Contribution | Prior Art | Innovation |
|---|---|---|
| **Unified TTS/VC** | Separate systems | Single model, mode conditioning |
| **Codec LM for VC** | Mel-spectrogram based | Direct token transformation |
| **Voice state in VC** | Style vectors only | Frame-level acoustic control |
| **Streaming codec LM** | Full-sequence | Real-time block-wise |
| **TTS+VC training** | Task-specific | Multi-task joint training |

### Key Differentiators vs Prior Art

| Method | TTS | VC | Unified | Streaming | Voice State |
|---|---|---|---|---|---|
| VALL-E | ✓ | ✗ | ✗ | ✗ | ✗ |
| SoundStorm | ✓ | ✗ | ✗ | ✓ | ✗ |
| FreeVC | ✗ | ✓ | ✗ | ✗ | ✗ |
| StyleStream | ✗ | ✓ | ✗ | ✓ | ✗ |
| **UCLM (Ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## 9. Experimental Plan

### 9.1 Datasets

| Dataset | Hours | TTS | VC | NV |
|---|---|---|---|---|
| LibriTTS-R | 585 | ✓ | — | — |
| VCTK | 44 | ✓ | ✓ | — |
| JVS | 30 | ✓ | ✓ | — |
| Expresso | 40 | ✓ | — | ✓ |
| JVNV | 4 | ✓ | — | ✓ |
| IntimateDialogue | 5 | ✓ | — | ✓ |

### 9.2 Evaluation

| Task | Metrics |
|---|---|
| **TTS** | UTMOS, MOS, Emo-SIM |
| **VC** | SECS, MCD, speaker similarity |
| **NV** | NV-MOS, naturalness |
| **Streaming** | Latency, overrun rate |
| **Unified** | Cross-task consistency |

### 9.3 Ablations

1. **Mode conditioning**: Unified vs separate models
2. **Voice state**: w/ vs w/o in VC mode
3. **Context length**: 1 vs 2 vs 3 blocks
4. **Block size**: 10ms vs 40ms frames
5. **AR vs PAR**: Speed vs quality tradeoff

---

## 10. Paper Title Candidates

1. **"UCLM: Unified Codec Language Model for Real-Time TTS and Voice Conversion"**
2. **"One Model, Two Tasks: Unified Neural Codec Generation for Speech Synthesis and Conversion"**
3. **"CodecLM-VC: Real-Time Voice Conversion via Codec Language Modeling"**
4. **"Beyond Pipelines: Unified Token-Based Generation for TTS and VC"**

---

## 11. Implementation Roadmap

| Phase | Duration | Tasks |
|---|---|---|
| **A** | 1 week | EnCodec integration, token extraction |
| **B** | 2 weeks | UCLM architecture (TTS only) |
| **C** | 1 week | VC mode + modality fusion |
| **D** | 2 weeks | Multi-task training pipeline |
| **E** | 1 week | Streaming inference |
| **F** | 1 week | Voice state estimation |
| **G** | 1 week | Evaluation + ablations |

**Total: ~9 weeks**

---

## 12. Consistency Checklist

- [x] EnCodec frame rate = 100 fps (10ms) matches TMRVC hop_length
- [x] Speaker embedding dimension (192) matches current `.tmrvc_speaker`
- [x] Voice state dimension (8) compatible with voice source params
- [x] `A_t` (`[B,8]`) and `B_t` (`[B,4]`) の dual-stream が全モードで一致
- [x] `delta_voice_state` が学習/推論の両方で使用される
- [x] Streaming block size divisible by hop_length
- [x] Causal masking in transformer for real-time
- [x] Control vocabulary size (64) が ONNX/Rust/Python で一致

---

## 13. References

1. Wang et al., "VALL-E: Neural Codec Language Models are Zero-Shot TTS", 2023
2. Borsos et al., "SoundStorm: Efficient Parallel Audio Generation", 2023
3. Défossez et al., "High Fidelity Neural Audio Compression" (EnCodec), 2022
4. Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN" (DAC), 2023
5. Chen et al., "FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion", 2023
6. Huang et al., "StyleStream: Streaming Real-Time Voice Conversion", 2026
7. Le et al., "Voicebox: Text-Guided Multilingual Universal Speech Generation", 2023
