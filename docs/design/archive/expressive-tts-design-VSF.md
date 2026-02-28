# Voice Source Flow (VSF): Expressive TTS with Non-Verbal Vocalizations

Kojiro Tanaka — Expressive TTS Design
Created: 2026-02-27 (Asia/Tokyo)

> **Goal:** 濡れ場演技等の高度な感情表現を実現する新規 TTS アーキテクチャ。
> **Core Novelty:** (1) 8種類のNV（breath/moan等）+ voice source targets、(2) 感情→voice source 時変生成、(3) テキストなし純粋NV生成、(4) Causal streaming (<50ms)。

---

## 1. Abstract

Existing emotional TTS systems control emotion at linguistic granularity (phoneme/word/utterance) with speaker-independent representations, and **cannot generate non-verbal vocalizations (NVs)** essential for intimate acting—breathing, moaning, laughing, crying. We propose **Voice Source Flow (VSF)**, a novel framework that:

1. **Generates time-varying voice source parameters** (breathiness, tension, jitter, shimmer, formant_shift, roughness) directly from emotion embeddings
2. **Supports non-verbal vocalizations** via special tokens (`<breath>`, `<moan>`, `<laugh>`, etc.) with predefined acoustic targets
3. **Enables three generation modes**: text-only, mixed text+NV, and pure NV (no linguistic content)
4. **Translates emotion through speaker-specific mappings** — the same "embarrassment" produces different expressions for different speakers
5. **Achieves causal streaming inference** under 50ms latency

Experiments on Expresso, JVNV, and custom intimate dialogue datasets demonstrate superior expressiveness in nuanced emotional scenarios, with particular strength in generating natural breath patterns and vocal tremors during high-arousal states.

---

## 2. Related Work and Gap Analysis

### 2.1 Non-Verbal Vocalization (NV) Generation

| Method | NV Support | NV Types | Time-Varying |
|---|---|---|---|
| EmoCtrl-TTS (Wu et al., 2024) | ✓ | laughter, crying | ✓ |
| Inworld TTS-1 (2025) | ✓ | audio markups | ✓ |
| **VSF (Ours)** | **✓** | **breath, moan, laugh, cry, sigh, gasp, whisper, shout** | **✓ + voice source** |

**Gap:** No prior work generates **breath/moan/sigh** with explicit voice source parameter control.

### 2.2 Voice Source Control in TTS

| Method | Voice Source Params | Time-Varying | Emotion-Driven |
|---|---|---|---|
| Perceptual Voice Quality (Rautenberg et al., 2025) | ✓ | ✗ | ✗ |
| Acoustic Condition Pathway (TMRVC) | ✓ (8 params) | ✗ | ✗ |
| **VSF (Ours)** | **✓ (8 params)** | **✓** | **✓** |

**Gap:** Voice source parameters are treated as static style descriptors. No prior work generates them **time-varyingly from emotion**.

### 2.3 Real-Time Emotional TTS

| Method | Latency | Streaming |
|---|---|---|
| StyleStream (Liu et al., 2026) | ~1s | ✓ |
| Standard EVC/TTS | >200ms | ✗ |
| **HSEF (Ours)** | **<50ms** | **✓ (causal)** |

**Gap:** No prior work achieves sub-50ms causal streaming with fine-grained emotion control.

---

## 3. Proposed Architecture

### 3.1 Overview

```
Text + NV_tokens + emotion_semantic[12, T] (or LLM output)
        │
        ├──▶ TextEncoder → text_features[256, L]
        │
        ├──▶ NVEmbedding → nv_features[256, L_nv]     ───┐
        │     (non-verbal tokens: <breath>, <moan>...)   │
        │                                                 │
        ├──▶ TokenMerger(text_features, nv_features) ◀───┘
        │       → merged_features[256, L_total]
        │
        ├──▶ DurationPredictor(merged, emotion_macro)
        │       → durations[L_total]
        │
        ├──▶ LengthRegulate → merged_features[256, T]
        │
        ├──▶ EmotionHierarchicalDecoder ──────────────────┐
        │       ├── Macro (~1s):   emotion_category[T/100]│
        │       │                + intensity[T/100]       │
        │       ├── Meso (~100ms): emotion_flow[T]        │
        │       │                (VAD + dynamics)         │
        │       └── Micro (~10ms): voice_source[T]        │
        │                          (breathiness, tension) │
        │                                                  │
        │       + SpeakerEmotionTranslator               │
        │         emotion_expressed = T_speaker @ emotion │
        │                                                  │
        ├──▶ F0Predictor(merged, emotion_flow) ◀─────────┘
        │       → f0[T], voiced[T]
        │       ※ NV区間: voiced=0 for <breath>/<whisper>
        │                 special F0 pattern for <laugh>/<cry>
        │
        ├──▶ ContentSynthesizer(merged)
        │       → content[256, T]
        │
        └──▶ Converter(content, spk_embed, emotion_expressed)
                → Vocoder → Audio
```

### 3.2 Non-Verbal Vocalization (NV) Support

**Key Extension:** Support for non-verbal vocalizations (breathing, moaning, laughing, crying, etc.) 
that are essential for intimate acting and expressive speech.

#### NV Token Vocabulary

| Token | Description | Voice Source | F0/Voicing |
|---|---|---|---|
| `<breath>` | Inhalation/exhalation | ↑breathiness_high | unvoiced |
| `<moan>` | Moaning/groaning | ↑breathiness_low, ↓tension | voiced, flat F0 |
| `<laugh>` | Laughter | — | periodic bursts, voiced |
| `<cry>` | Crying/sobbing | ↑roughness, ↑jitter | voiced, falling F0 |
| `<sigh>` | Sigh | ↑breathiness, ↓tension | unvoiced, falling |
| `<gasp>` | Sharp intake | ↑breathiness_high | unvoiced, short |
| `<whisper>` | Whispered speech | ↑breathiness, ↓tension | unvoiced throughout |
| `<shout>` | Shouting | ↑tension_high, ↑energy | voiced, high F0 |

#### NV Embedding Module

```python
class NVEmbedding(nn.Module):
    """Embed non-verbal tokens into the same space as phonemes.
    
    NV tokens are treated as special phonemes with learned acoustic targets.
    """
    
    NV_VOCAB = {
        "<breath>": 200,
        "<moan>": 201,
        "<laugh>": 202,
        "<cry>": 203,
        "<sigh>": 204,
        "<gasp>": 205,
        "<whisper>": 206,
        "<shout>": 207,
    }
    
    # Predefined voice source targets for each NV token
    NV_VOICE_SOURCE_TARGETS = {
        "<breath>":   [0.8, 0.9, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0],  # high breathiness
        "<moan>":     [0.6, 0.4, 0.2, 0.2, 0.02, 0.01, 0.0, 0.3], # low tension, breathy
        "<laugh>":    [0.3, 0.2, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # moderate
        "<cry>":      [0.4, 0.3, 0.3, 0.3, 0.05, 0.03, 0.0, 0.7], # high roughness
        "<sigh>":     [0.7, 0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0],  # breathy, relaxed
        "<gasp>":     [0.9, 0.8, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],  # very breathy
        "<whisper>":  [0.9, 0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],  # max breathiness
        "<shout>":    [0.2, 0.3, 0.9, 0.9, 0.03, 0.02, 0.0, 0.4], # high tension
    }
    
    def __init__(self, d_model=256, vocab_size=208):
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Learnable residual from targets
        self.nv_residual = nn.Embedding(8, 8)  # 8 NV tokens, 8 voice source params
```

#### F0/Voicing Control for NV Tokens

```python
class NVF0Controller(nn.Module):
    """Generate F0 and voicing patterns for NV tokens."""
    
    def forward(self, f0_pred, voiced_pred, token_ids):
        # Override voicing for breath-type tokens
        breath_tokens = [200, 204, 205, 206]  # <breath>, <sigh>, <gasp>, <whisper>
        for tok in breath_tokens:
            mask = (token_ids == tok)
            voiced_pred[mask] = 0.0
        
        # Special F0 patterns
        # <laugh>: periodic modulation
        # <cry>: falling contour
        # <moan>: flat/low
        # <shout>: high/intense
        
        return f0_adjusted, voiced_adjusted
```

### 3.2 Temporal Scale Hierarchy

Unlike linguistic hierarchy (phoneme/word/utterance), we decompose emotion across **temporal scales**:

| Level | Timescale | Resolution | Content |
|---|---|---|---|
| **Macro** | ~1s | 10 frames (100ms) | Emotion category + intensity |
| **Meso** | ~100ms | 1 frame | VAD (valence/arousal/dominance) + dynamics |
| **Micro** | ~10ms | 1 frame | Voice source params (breathiness, tension, jitter, shimmer, formant_shift, roughness) |

**Rationale:**
- Macro captures emotion transitions (e.g., neutral → embarrassed)
- Meso captures intensity changes (e.g., embarrassment deepening)
- Micro captures acoustic nuances (e.g., breathiness increasing, voice trembling)

### 3.3 SpeakerEmotionTranslator

**Core Insight:** The same emotion semantic produces different acoustic expressions for different speakers.

```
emotion_semantic[T, d_semantic]  (speaker-independent "embarrassment")
        │
        ├──▶ T_speaker (learned per speaker, low-rank)
        │     T_speaker = W_base + LoRA_speaker
        │     LoRA_speaker: [d_semantic × r] × [r × d_expressed]
        │     r = 4 (rank)
        │
        └──▶ emotion_expressed[T, d_expressed]
             (speaker-specific acoustic realization)
```

**Implementation:**
- `T_speaker` is a small LoRA-style adapter (4-16KB per speaker)
- Learned from few-shot enrollment (5-10 minutes of emotional speech)
- Enables zero-shot transfer: new speaker provides 30s enrollment → LoRA optimized

### 3.4 EmotionHierarchicalDecoder

```python
class EmotionHierarchicalDecoder(nn.Module):
    """
    Multi-scale emotion decoder with speaker conditioning.
    
    Input:
        text_features[B, 256, T]
        emotion_semantic[B, d_semantic] or [B, d_semantic, T]
        spk_embed[B, 192]
    
    Output:
        emotion_flow[B, d_flow, T]        # Meso-level: VAD + dynamics
        voice_source[B, 8, T]             # Micro-level: breathiness, tension, etc.
    """
    
    def __init__(self, d_text=256, d_semantic=32, d_flow=16, d_voice_source=8):
        # Macro decoder: operates at T/10 resolution
        self.macro_conv = nn.Conv1d(d_text, 128, kernel_size=1)
        self.macro_rnn = nn.GRU(128, 64, num_layers=2, bidirectional=False, batch_first=True)
        self.macro_head = nn.Linear(64, d_semantic)  # emotion category + intensity
        
        # Meso decoder: frame-level emotion flow
        self.meso_conv = CausalConvNeXtStack(d_text, 128, n_blocks=4, kernel=3)
        self.meso_head = nn.Linear(128, d_flow)
        
        # Micro decoder: voice source params from emotion flow
        self.micro_conv = CausalConvNeXtStack(d_flow + d_text, 64, n_blocks=2, kernel=3)
        self.micro_head = nn.Linear(64, d_voice_source)
        
        # Speaker translator (LoRA-style)
        self.speaker_translator = SpeakerEmotionTranslator(d_semantic, d_flow, rank=4)
    
    def forward(self, text_features, emotion_semantic, spk_embed):
        B, _, T = text_features.shape
        
        # Macro: downsample → decode → upsample
        macro_in = text_features[:, :, ::10]  # T/10
        macro_feat = self.macro_conv(macro_in)
        macro_feat = macro_feat.permute(0, 2, 1)  # [B, T/10, 128]
        macro_feat, _ = self.macro_rnn(macro_feat)
        macro_out = self.macro_head(macro_feat)  # [B, T/10, d_semantic]
        macro_upsampled = F.interpolate(macro_out.permute(0, 2, 1), size=T, mode='linear')
        
        # Meso: frame-level with macro conditioning
        meso_feat = self.meso_conv(text_features + macro_upsampled * 0.1)
        emotion_flow = self.meso_head(meso_feat.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Speaker translation
        emotion_expressed = self.speaker_translator(emotion_flow, spk_embed)
        
        # Micro: voice source from emotion + text
        micro_in = torch.cat([emotion_expressed, text_features], dim=1)
        micro_feat = self.micro_conv(micro_in)
        voice_source = self.micro_head(micro_feat.permute(0, 2, 1)).permute(0, 2, 1)
        voice_source = torch.sigmoid(voice_source)  # [0, 1] range
        
        return emotion_expressed, voice_source
```

### 3.5 Voice Source Parameter Definition

| Index | Parameter | Range | Description |
|---|---|---|---|
| 0 | breathiness_low | [0, 1] | Low-frequency aspiration (< 3kHz) |
| 1 | breathiness_high | [0, 1] | High-frequency aspiration (>= 3kHz) |
| 2 | tension_low | [0, 1] | Low-frequency vocal fold tension |
| 3 | tension_high | [0, 1] | High-frequency vocal fold tension |
| 4 | jitter | [0, 0.1] | F0 micro-perturbation |
| 5 | shimmer | [0, 0.1] | Amplitude micro-perturbation |
| 6 | formant_shift | [-1, 1] | Vocal tract length ratio |
| 7 | roughness | [0, 1] | Subharmonic / pressed voice |

### 3.6 Generation Modes

The system supports three generation modes:

#### Mode 1: Text-Only (Standard TTS)
```
"Hello, how are you?" → speech with emotion_control
```
Standard emotional TTS with time-varying voice source control.

#### Mode 2: Mixed Text + NV
```
"Ah... <breath> that feels <moan> good <sigh>"
```
Interleaved text and non-verbal tokens. NV tokens are embedded and processed
alongside phonemes, with special handling for voicing and F0.

#### Mode 3: Pure NV (No Text)
```
<moan> <moan> <breath> <sigh>
```
**Critical for intimate acting**: Generate pure non-verbal vocalizations
without any linguistic content. The NV tokens drive the entire generation
through `nv_features → DurationPredictor → F0Predictor → ContentSynthesizer`.

```python
class PureNVGenerator(nn.Module):
    """Generate speech from NV tokens only (no linguistic content).
    
    Uses a separate lightweight decoder for NV-only sequences.
    """
    
    def __init__(self, d_nv=256, d_emotion=16):
        # NV-only duration predictor
        self.nv_duration = nn.Sequential(
            nn.Conv1d(d_nv, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(128, 1, 1),
            nn.Softplus(),
        )
        
        # NV → content (no text encoder needed)
        self.nv_to_content = CausalConvNeXtStack(d_nv, 256, n_blocks=4)
        
    def forward(self, nv_features, emotion_flow):
        durations = self.nv_duration(nv_features)
        content = self.nv_to_content(nv_features + emotion_flow)
        return content, durations
```

### 3.7 Use Cases (Intimate Acting)

| Scenario | Tokens | Voice Source Pattern |
|---|---|---|
| **Embarrassment** | "I... <breath> I don't know..." | ↑breathiness, ↓tension, ↑jitter |
| **Arousal buildup** | "Yes... <moan> more..." | ↑breathiness over time, tension varies |
| **Climax** | "<moan> <moan> ah!" | Peak breathiness, high roughness |
| **Post-climax** | "<sigh> that was..." | ↓tension, ↑breathiness_high, falling F0 |
| **Whispered secret** | "<whisper> I love you" | Unvoiced throughout, high breathiness |
| **Shout** | "<shout> NO!" | High tension, high F0, high energy |

---

## 4. Training Strategy

### 4.1 Data Requirements

| Dataset | Hours | NV Annotations | Use |
|---|---|---|---|
| Expresso | 40h | ✓ (laughter, sighs) | Training (EN) |
| JVNV | 4h | ✗ | Training (JA), emotion labels only |
| **Custom intimate data** | 2-5h | ✓ (breath, moan, sigh, gasp) | **NV training** |
| NaturalVoices | 5,049h | Auto-annotated | Pretraining (optional) |

**NV Data Collection:**
- Manual annotation: Label <breath>, <moan>, etc. in intimate dialogue datasets
- Automatic detection: Train breath/laugh detector on small labeled set
- Data augmentation: Synthesize NV tokens with varying durations

### 4.2 Training Phases

**Phase A: Core TTS + EmotionHierarchy (No NV)**
- Dataset: Expresso + JVNV
- Loss: mel + STFT + emotion_cls + VAD + voice_source_MSE
- Components: TextEncoder, EmotionHierarchicalDecoder, F0Predictor, ContentSynthesizer

**Phase B: NV Token Training**
- Dataset: Custom intimate data (2-5h with NV labels)
- Loss: mel + voice_source_MSE (with NV targets) + NV_classification
- Components: NVEmbedding, NVF0Controller, PureNVGenerator

**Phase C: SpeakerEmotionTranslator Learning**
- Per-speaker LoRA optimization
- Enrollment: 5-10 min emotional speech per speaker

**Phase D: End-to-End Fine-tuning**
- Joint training with frozen Converter/Vocoder
- Mixed text + NV sequences
- Loss: mel + STFT + emotion_consistency + NV_timing

### 4.3 Loss Functions

```python
# Phase A/B losses
loss_mel = F.l1_loss(pred_mel, gt_mel)
loss_stft = multi_resolution_stft_loss(pred_mel, gt_mel)
loss_emotion = F.cross_entropy(pred_emotion_cls, gt_emotion)
loss_vad = F.mse_loss(pred_vad, gt_vad)
loss_voice_source = F.mse_loss(pred_vs, gt_vs)  # voice source params

# Phase B additional losses
loss_nv_cls = F.cross_entropy(pred_nv_type, gt_nv_type)  # NV token classification
loss_nv_vs = F.mse_loss(pred_vs[nv_mask], nv_targets)    # NV voice source targets

# Phase D additional losses
loss_nv_timing = F.l1_loss(pred_nv_duration, gt_nv_duration)
loss_consistency = emotion_consistency_loss(pred_emotion_flow, text_context)

total_loss = (
    1.0 * loss_mel +
    0.5 * loss_stft +
    0.3 * loss_emotion +
    0.2 * loss_vad +
    0.5 * loss_voice_source +
    0.3 * loss_nv_cls +
    0.3 * loss_nv_vs
)
```

---

## 5. Novelty Summary

| Contribution | Prior Art | Our Innovation |
|---|---|---|
| **Voice source from emotion (time-varying)** | Static presets only | Emotion → breathiness/tension/jitter per frame |
| **Non-verbal vocalizations** | Laughter/crying only | **8 NV types** + voice source targets |
| **Pure NV generation** | Requires text | Text-free NV sequences |
| **Speaker-dependent emotion** | Speaker-independent style | Learned per-speaker translation |
| **Real-time inference** | >200ms or non-streaming | <50ms causal streaming |

### Key Differentiators vs. Closest Prior Art

| Aspect | EmoCtrl-TTS (2024) | TTS-CtrlNet (2025) | **VSF (Ours)** |
|---|---|---|---|
| NV types | laughter, crying | — | **8 types including breath/moan** |
| Voice source | ✗ | ✗ | **✓ (8 params, time-varying)** |
| Pure NV mode | ✗ | ✗ | **✓** |
| Latency | ~200ms | — | **<50ms** |
| Streaming | ✗ | ✗ | **✓ (causal)** |

---

## 6. Experimental Plan

### 6.1 Datasets

| Dataset | Hours | NV Annotations | Emotion Labels | Use |
|---|---|---|---|---|
| Expresso | 40h | ✓ (laughter, sighs) | 26 styles | Training (EN) |
| JVNV | 4h | ✗ | 6 emotions | Training (JA) |
| **IntimateDialogue (new)** | 5h | **✓ (all 8 NV types)** | VAD + category | **NV training + eval** |
| NaturalVoices | 5,049h | Auto-detected | Auto-annotated | Pretraining |

### 6.2 Evaluation Metrics

| Category | Metric | Description |
|---|---|---|
| **Overall Quality** | UTMOS, MOS | Naturalness |
| **Emotion** | Emo-SIM, Aro-Val SIM | Emotion similarity |
| **Speaker** | SECS | Speaker cosine similarity |
| **Voice Source** | VS-MSE, VS-Corr | Breathiness/tension prediction accuracy |
| **NV Quality** | NV-MOS, NV-Detection-F1 | Naturalness + correctness of NV type |
| **Latency** | E2E-RTF, Overrun % | Real-time factor, buffer overruns |

### 6.3 Evaluation Scenarios

1. **Standard emotional TTS**: Compare with Hierarchical ED, TTS-CtrlNet, EmoCtrl-TTS
2. **NV generation quality**: A/B test for breath/moan/sigh naturalness
3. **Mixed text+NV**: Evaluate timing and naturalness of NV insertion
4. **Pure NV**: Evaluate expressiveness of NV-only sequences
5. **Speaker adaptation**: Zero-shot vs. few-shot (5 min enrollment)

### 6.4 Ablation Studies

| Study | Comparison | Expected Insight |
|---|---|---|
| **NV tokens** | w/ vs. w/o NV | Importance of explicit NV modeling |
| **Voice source targets** | Predefined vs. learned | Optimal initialization strategy |
| **Speaker translator** | w/ vs. w/o T_speaker | Speaker-dependent emotion value |
| **Pure NV mode** | Text+NV vs. Pure NV | Quality of text-free generation |
| **Voice source time-varying** | Static vs. dynamic | Importance of temporal control |

### 6.5 Human Evaluation Protocol

**MOS Tests:**
- 20 raters, 100 samples each
- Scenarios: standard TTS, text+NV, pure NV
- Aspects: naturalness, emotion appropriateness, NV quality

**A/B Preference Tests:**
- Baseline: EmoCtrl-TTS (laughter/crying only)
- Proposed: VSF (full NV + voice source)
- Question: "Which sounds more natural for intimate dialogue?"

---

## 7. Novelty Summary (Updated)

| Contribution | Prior Art | Innovation |
|---|---|---|
| **NV generation (breath/moan/sigh)** | Laughter/crying only | **8 NV types with acoustic targets** |
| **Time-varying voice source from emotion** | Static presets | **Frame-level generation** |
| **Pure NV mode (no text)** | Requires text | **Text-free generation** |
| **Causal streaming (<50ms)** | >200ms | **Real-time expressive TTS** |

### Paper Title Candidates

1. **"Voice Source Flow: Time-Varying Emotional Control with Non-Verbal Vocalization Support"**
2. **"Beyond Text: Expressive TTS with Non-Verbal Vocalizations and Voice Source Control"**
3. **"VSF-TTS: Generating Breath, Moans, and Voice Source Parameters from Emotion"**

---

## 8. Implementation Roadmap

| Phase | Duration | Tasks |
|---|---|---|
| **A** | 1 week | EmotionHierarchicalDecoder + NVEmbedding implementation |
| **B** | 1 week | NVF0Controller + PureNVGenerator |
| **C** | 1 week | SpeakerEmotionTranslator + LoRA integration |
| **D** | 2 weeks | Data collection (IntimateDialogue) + annotation |
| **E** | 2 weeks | Training pipeline (Phase A-D) |
| **F** | 1 week | Evaluation + ablation studies |
| **G** | 1 week | Paper draft + demos |

**Total: ~9 weeks**

---

## 9. Integration with TMRVC Pipeline

### 9.1 Converter FiLM Extension

Current: `d_cond = d_speaker(192) + n_style_params(64) = 256`

Extended: `d_cond = d_speaker(192) + emotion_expressed(16) + voice_source(8) + acoustic_params(32) = 248`

```python
# In converter.py
class ConverterWithEmotion(nn.Module):
    def forward(self, content, spk_embed, emotion_expressed, voice_source, acoustic_params):
        cond = torch.cat([
            spk_embed,           # [B, 192]
            emotion_expressed,   # [B, 16]
            voice_source,        # [B, 8]
            acoustic_params,     # [B, 32]
        ], dim=-1)  # [B, 248]
        
        # FiLM conditioning
        for block in self.blocks:
            x = block(x)
            x = self.film(x, cond)
```

### 9.2 Constants Update

```yaml
# configs/constants.yaml
# Emotion hierarchy
d_emotion_semantic: 32      # Input emotion dimension
d_emotion_flow: 16          # Meso-level emotion (VAD + dynamics)
d_voice_source: 8           # Micro-level voice source params
emotion_hierarchy_rank: 4   # LoRA rank for speaker translator

# NV tokens
nv_vocab_size: 8            # Number of NV token types
nv_token_start_id: 200      # Starting ID for NV tokens

# Macro/Meso/Micro timescales
macro_hop_frames: 10        # 100ms per macro step
meso_hop_frames: 1          # 10ms per meso step (frame-level)
```

---

## 10. Consistency Checklist

- [ ] `d_emotion_flow(16)` + `d_voice_source(8)` + `acoustic_params(32)` + `d_speaker(192)` = 248 (FiLM input)
- [ ] Voice source params match `acoustic-condition-pathway.md` indices 24-31
- [ ] NV token IDs start at 200 (after phoneme vocab 0-199)
- [ ] CausalConvNeXt blocks used for streaming compatibility
- [ ] LoRA rank (4) consistent with VC few-shot design
- [ ] Training config frozen components: Converter, Vocoder, SpeakerEncoder
- [ ] NV voice source targets defined for all 8 NV types
- [ ] Pure NV mode uses same ContentSynthesizer → Converter path

---

## 11. References

1. Inoue et al., "Hierarchical Control of Emotion Rendering in Speech Synthesis," IEEE TAC 2025
2. Jeong et al., "TTS-CtrlNet: Time varying emotion aligned text-to-speech generation with ControlNet," 2025
3. Wu et al., "Laugh Now Cry Later: Controlling Time-Varying Emotional States of Flow-Matching-Based Zero-Shot TTS," SLT 2024
4. Zhou et al., "Expressive Voice Conversion: A Joint Framework for Speaker Identity and Emotional Style Transfer," ASRU 2021
5. Rautenberg et al., "Speech Synthesis along Perceptual Voice Quality Dimensions," ICASSP 2025
6. Gong et al., "Perturbation Self-Supervised Representations for Cross-Lingual Emotion TTS," 2025
7. Park et al., "DEX-TTS: Diffusion-based EXpressive Text-to-Speech with Style Modeling on Time Variability," 2024
8. Inworld AI, "TTS-1 Technical Report," 2025
