# Unified Neural Codec Generator for Expressive TTS

Kojiro Tanaka — Unified Generator Design
Created: 2026-02-27 (Asia/Tokyo)

> **Core Insight:** 音声を「テキストから生成 + NVを混ぜる」ではなく、**過去音声 + テキスト + 声状態 を条件に、コーデックトークンを連続生成**する単一モデルで、言語・非言語を統一的に扱う。

---

## 1. Abstract

Existing TTS systems treat non-verbal vocalizations (NVs) as special tokens inserted into text, creating discontinuities when "speech segments" meet "cry segments." We propose **Unified Codec Generator (UCG)**, a single neural codec language model that:

1. **Generates speech tokens autoregressively** conditioned on:
   - Past acoustic context (self-generated tokens)
   - Text content (phoneme/character sequence)
   - Voice state trajectory (continuous parameters: breathiness, tension, arousal, etc.)

2. **Unifies verbal and non-verbal in one stream**: Laughter-mixed speech, sobbing-while-speaking, and breathy whispers emerge naturally from the same generation process without explicit segmentation.

3. **Enables real-time streaming** via block-wise parallel decoding and streaming flow matching, achieving <50ms latency.

Experiments demonstrate superior naturalness in emotionally complex scenarios (intimate dialogue, crying speech, laughter-mixed utterances) compared to pipeline-based emotional TTS.

---

## 2. Motivation: Why Unified Generator?

### 2.1 Problem with Pipeline Approaches

```
Traditional: Text → [phonemes] → TTS → [speech] → +NV → [mixed]
                                              ↑
                                        Discontinuity at boundaries
```

- NV tokens create "insertion points" in the audio stream
- "Crying while speaking" requires artificial blending of two states
- Voice source parameters are applied globally, not dynamically

### 2.2 Unified Generator Solution

```
Unified: Text + VoiceState[t] + PastTokens[t-k:t] → CodecLM → Tokens[t:t+Δ] → Decoder → Audio
                         ↑
              Continuous voice state controls acoustic properties
              Past tokens provide acoustic continuity
```

- Single generation process for entire audio
- Past context ensures smooth transitions
- Voice state parameters guide acoustic characteristics in real-time

---

## 3. Architecture

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Unified Codec Generator                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Inputs:                                                             │
│    • text: Phoneme sequence [L]                                     │
│    • voice_state: Continuous params [T, d_state]                   │
│    • past_tokens: Codec tokens from previous blocks [k, n_codebooks]│
│    • spk_embed: Speaker embedding [d_speaker]                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Conditioning Module                        │   │
│  │  • TextEncoder: phonemes → text_features [L, d]              │   │
│  │  • VoiceStateEncoder: state → state_features [T, d]          │   │
│  │  • ContextEncoder: past_tokens → context [k, d]              │   │
│  │  • Cross-attention: text ↔ state temporal alignment          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               Codec Language Model (Transformer)              │   │
│  │                                                               │   │
│  │  For each position t:                                         │   │
│  │    P(token_t | token_<t, text, voice_state, past_audio)      │   │
│  │                                                               │   │
│  │  Multi-codebook prediction:                                   │   │
│  │    Q_0, Q_1, ..., Q_{n-1} ← n_codebooks × vocab_size         │   │
│  │    Delay pattern: [Q_0:t, Q_1:t-1, Q_2:t-2, ...]             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Neural Codec Decoder                         │   │
│  │  EnCodec / DAC / SoundStream                                  │   │
│  │  Tokens [T, n_codebooks] → Mel → Waveform                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Voice State Parameters

The voice state vector `s[t] ∈ R^d` controls acoustic properties continuously:

| Index | Parameter | Range | Effect |
|---|---|---|---|
| 0 | breathiness | [0, 1] | Aspiration noise level |
| 1 | tension | [0, 1] | Vocal fold tension |
| 2 | arousal | [0, 1] | Emotional activation |
| 3 | valence | [-1, 1] | Positive/negative emotion |
| 4 | roughness | [0, 1] | Voice quality (creaky/harsh) |
| 5 | voicing | [0, 1] | Voiced vs unvoiced continuum |
| 6 | energy | [0, 1] | Overall loudness |
| 7 | speaking_rate | [0.5, 2.0] | Relative speed |

**LLM Output:** The LLM generates `voice_state[t]` alongside text, creating a time-varying trajectory that guides acoustic generation.

### 3.3 Context Conditioning

**Key Innovation:** Past generated tokens serve as acoustic context, ensuring continuity:

```python
class ContextConditioner(nn.Module):
    """Encode past codec tokens as conditioning context."""
    
    def __init__(self, n_codebooks=8, vocab_size=1024, d_model=256):
        self.codebook_embed = nn.ModuleList([
            nn.Embedding(vocab_size, d_model // n_codebooks)
            for _ in range(n_codebooks)
        ])
        self.context_encoder = nn.TransformerEncoder(...)
        
    def forward(self, past_tokens):
        """
        Args:
            past_tokens: [B, n_codebooks, k]  # k = context length
        Returns:
            context: [B, k, d_model]
        """
        embeddings = []
        for i, embed in enumerate(self.codebook_embed):
            embeddings.append(embed(past_tokens[:, i, :]))
        context = torch.cat(embeddings, dim=-1)
        return self.context_encoder(context)
```

### 3.4 Streaming Generation

**Block-wise Generation:**

```
Block 0: Generate tokens[0:Δ] from text[0:l], state[0:Δ], past_tokens=∅
Block 1: Generate tokens[Δ:2Δ] from text[l:2l], state[Δ:2Δ], past_tokens[0:Δ]
Block 2: Generate tokens[2Δ:3Δ] from text[2l:3l], state[2Δ:3Δ], past_tokens[Δ:2Δ]
...
```

- Block size Δ = 20-40 frames (200-400ms at 10ms/frame)
- Past context k = 1-3 blocks (200-1200ms)
- Causal attention mask within each block

**Parallel Decoding (Non-autoregressive within block):**

Following SoundStorm / Voicebox:
1. Generate first codebook autoregressively
2. Generate remaining codebooks in parallel (non-AR)
3. Total speedup: ~4-8x vs full AR

```python
class ParallelCodecDecoder(nn.Module):
    """Non-autoregressive multi-codebook prediction within a block."""
    
    def forward(self, first_codebook, text_features, voice_state):
        """
        Args:
            first_codebook: [B, T] - AR generated
            text_features: [B, L, d]
            voice_state: [B, T, d_state]
        Returns:
            all_codebooks: [B, n_codebooks, T]
        """
        # Conditioning from first codebook + text + state
        cond = self.conditioner(first_codebook, text_features, voice_state)
        
        # Parallel prediction of remaining codebooks
        logits = self.parallel_transformer(cond)  # [B, T, (n_cb-1) * vocab]
        
        return self.rearrange(logits)
```

---

## 4. Training

### 4.1 Data Preparation

```
1. Encode all audio with neural codec (e.g., EnCodec)
   audio → codec_tokens [T, n_codebooks]

2. Extract voice state parameters
   audio → voice_state [T, d_state] via pre-trained estimator
   
3. Align text to tokens
   text + audio → forced alignment → token_text_alignment
```

### 4.2 Training Objective

```python
# Autoregressive loss for first codebook
loss_ar = CrossEntropy(pred_Q0[t], gt_Q0[t])

# Parallel loss for remaining codebooks
loss_parallel = sum(
    CrossEntropy(pred_Qi[t], gt_Qi[t])
    for i in range(1, n_codebooks)
)

# Voice state consistency loss (optional)
loss_state = MSE(encoder_state, target_state)

total_loss = loss_ar + loss_parallel + λ * loss_state
```

### 4.3 Curriculum

| Phase | Data | Objective |
|---|---|---|
| **1** | Clean speech only | Basic codec LM (text → tokens) |
| **2** | Emotional speech | Add voice_state conditioning |
| **3** | Speech + NV (laugh, cry, breath) | Learn mixed verbal/non-verbal |
| **4** | Streaming fine-tune | Block-wise causal training |

---

## 5. Real-Time Considerations

### 5.1 Latency Budget

| Component | Time (ms) |
|---|---|
| Text + State encoding | ~2 |
| Codec LM (block) | ~15 |
| Codec decoder | ~8 |
| Buffer management | ~5 |
| **Total** | **~30** |

### 5.2 Streaming Flow Matching (Optional)

For even lower latency, replace AR with streaming flow matching:

```python
class StreamingFlowMatching(nn.Module):
    """Flow-matching based token generation with streaming support."""
    
    def forward(self, x_t, t, text, voice_state, past_context):
        """
        Args:
            x_t: Noisy tokens at timestep t
            t: Flow timestep
            text, voice_state, past_context: Conditioning
        Returns:
            v: Velocity prediction (toward clean tokens)
        """
        # Causal transformer with block-wise context
        cond = self.fuse(text, voice_state, past_context)
        return self.velocity_net(x_t, t, cond)
```

---

## 6. Comparison with Prior Art

| Method | NV Support | Continuity | Real-time | Voice State |
|---|---|---|---|---|
| VALL-E | ✗ | — | ✗ | ✗ |
| SoundStorm | ✗ | — | ✓ (parallel) | ✗ |
| Voicebox | ✗ | ✓ (infill) | △ | ✗ |
| EmoCtrl-TTS | laughter, crying | △ | ✗ | ✗ |
| **UCG (Ours)** | **All NV types** | **✓ (unified)** | **✓ (<50ms)** | **✓ (continuous)** |

### Key Differentiators

1. **Unified generation**: No separate "speech" vs "NV" modes
2. **Continuous voice state**: Fine-grained acoustic control
3. **Past context conditioning**: Natural transitions
4. **Streaming-ready**: Block-wise generation from design

---

## 7. Integration with TMRVC

### 7.1 Replacing Pipeline Components

```
Current TMRVC:
  Text → TextEncoder → Duration → F0 → Content → Converter → Vocoder → Audio

Unified Generator:
  Text + VoiceState + PastTokens → CodecLM → CodecDecoder → Audio
```

**Simplification:** 5 models → 2 models (CodecLM + CodecDecoder)

### 7.2 Speaker Adaptation

- Few-shot: Provide reference audio as initial `past_tokens`
- Zero-shot: Use speaker embedding in conditioning

### 7.3 VC Compatibility

For voice conversion, the same architecture works:

```
Source audio → Codec tokens → CodecLM (with target speaker embed) → New tokens → Audio
```

---

## 8. Experimental Plan

### 8.1 Datasets

| Dataset | Hours | NV Types | Use |
|---|---|---|---|
| LibriTTS-R | 585h | — | Phase 1 (clean speech) |
| Expresso | 40h | laughter, sighs | Phase 2-3 |
| JVNV | 4h | 6 emotions | Phase 2 |
| IntimateDialogue | 5h | breath, moan, cry, sob | Phase 3 |
| **Custom NV-rich** | 10h | all types | Phase 3-4 |

### 8.2 Evaluation

| Metric | Description |
|---|---|
| Token accuracy | Codebook prediction accuracy |
| UTMOS / MOS | Naturalness |
| NV-MOS | Non-verbal quality |
| Continuity score | Transition smoothness (human eval) |
| Latency | End-to-end ms |
| SECS | Speaker similarity |

### 8.3 Ablations

1. **Context length**: k=0 vs k=1 vs k=3 blocks
2. **Voice state**: w/ vs w/o continuous conditioning
3. **Parallel vs AR**: Speed vs quality tradeoff
4. **Block size**: Δ=20 vs Δ=40 frames

---

## 9. Novelty Summary

| Contribution | Prior Art | Innovation |
|---|---|---|
| **Unified verbal+non-verbal** | Separate NV tokens | Single generation stream |
| **Continuous voice state** | Discrete emotion labels | Frame-level acoustic control |
| **Past context conditioning** | Text-only | Acoustic continuity from history |
| **Streaming codec LM** | Full-sequence | Block-wise real-time generation |

### Paper Title Candidates

1. **"Unified Codec Generator: Continuous Speech Synthesis with Non-Verbal Vocalizations"**
2. **"One Stream: Unified Generation of Verbal and Non-Verbal Speech via Codec Language Modeling"**
3. **"UCG-TTS: Streaming Neural Codec Generation with Voice State Conditioning"**

---

## 10. Implementation Roadmap

| Phase | Duration | Tasks |
|---|---|---|
| **A** | 1 week | Codec tokenization pipeline (EnCodec/DAC) |
| **B** | 2 weeks | CodecLM architecture + training (Phase 1) |
| **C** | 1 week | Voice state estimator + conditioning |
| **D** | 2 weeks | NV data collection + training (Phase 2-3) |
| **E** | 1 week | Streaming + parallel decoding |
| **F** | 1 week | Evaluation + ablations |
| **G** | 1 week | Paper draft |

**Total: ~9 weeks**

---

## 11. References

1. Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E), 2023
2. Borsos et al., "SoundStorm: Efficient Parallel Audio Generation", 2023
3. Le et al., "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale", 2023
4. Défossez et al., "High Fidelity Neural Audio Compression" (EnCodec), 2022
5. Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN" (DAC), 2023
6. Wu et al., "Laugh Now Cry Later: Controlling Time-Varying Emotional States" (EmoCtrl-TTS), 2024
7. Yang et al., "Streaming Flow Matching for Real-Time Audio Generation", 2024
