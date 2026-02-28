# TMRVC Rust Engine Design

Kojiro Tanaka — Rust engine design
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-28 (Rust implementation)

> **Scope:** tmrvc-engine-rs (nih-plug 非依存の Rust ストリーミング推論ライブラリ) と
> tmrvc-vst (nih-plug VST3 ラッパー) の設計。

---

## 1. クラス図

```
┌───────────────────────────────────────────────────────────────────┐
│  tmrvc-vst (nih-plug VST3)                                        │
│                                                                   │
│  ┌─────────────────────┐    ┌─────────────────────┐              │
│  │  TMRVCPlugin        │    │  TMRVCParams        │              │
│  │  (Plugin)           │◄──▶│  (Params)           │              │
│  │                     │    │                     │              │
│  │  - initialize()     │    │  - dry_wet          │              │
│  │  - process()        │    │  - output_gain      │              │
│  │  - reset()          │    │  - alpha_timbre     │              │
│  │                     │    │  - latency_quality_q│              │
│  └────────┬────────────┘    └─────────────────────┘              │
│           │ owns                                                  │
│           ▼                                                       │
└───────────┼───────────────────────────────────────────────────────┘
            │
            │ uses
            ▼
┌───────────────────────────────────────────────────────────────────┐
│  tmrvc-engine-rs (nih-plug-free Rust crate)                       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  StreamingEngine                                         │     │
│  │  (central orchestrator)                                  │     │
│  │                                                          │     │
│  │  - new(status: Option<Arc<SharedStatus>>)                │     │
│  │  - load_models(dir: &Path)                               │     │
│  │  - load_speaker(path: &Path)                             │     │
│  │  - load_style(path: &Path)                               │     │
│  │  - process_one_frame(input, output, params)              │     │
│  │  - reset()                                               │     │
│  │  - is_ready() -> bool                                    │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  OrtBundle          │  │  TensorPool           │     │     │
│  │  │                     │  │                       │     │     │
│  │  │  - content_encoder  │  │  - mel_frame          │     │     │
│  │  │  - ir_estimator     │  │  - content            │     │     │
│  │  │  - converter        │  │  - acoustic_params    │     │     │
│  │  │  - converter_hq     │  │  - pred_features      │     │     │
│  │  │  - vocoder          │  │  - stft_mag/phase     │     │     │
│  │  │  - (TTS models)     │  │  - (contiguous alloc) │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  ModelStates        │  │  ContentBuffer        │     │     │
│  │  │                     │  │ (HQ mode lookahead)   │     │     │
│  │  │  - content_encoder  │  │                       │     │     │
│  │  │  - ir_estimator     │  │  - push(content)      │     │     │
│  │  │  - converter        │  │  - fill_flat_tensor() │     │     │
│  │  │  - converter_hq     │  │  - is_full()          │     │     │
│  │  │  - vocoder          │  │                       │     │     │
│  │  │  (PingPongState)    │  │                       │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐  ┌──────────────────────┐     │     │
│  │  │  SpeakerFile        │  │  StyleFile            │     │     │
│  │  │                     │  │                       │     │     │
│  │  │  - spk_embed[192]   │  │  - target_log_f0      │     │     │
│  │  │  - lora_delta[15872]│  │  - target_articulation│     │     │
│  │  │  - voice_source_    │  │  - metadata           │     │     │
│  │  │    preset[8]        │  │                       │     │     │
│  │  └─────────────────────┘  └──────────────────────┘     │     │
│  │                                                          │     │
│  │  ┌─────────────────────┐                                │     │
│  │  │  SharedStatus       │  Atomic stats for GUI         │     │
│  │  │  (Arc<SharedStatus>)│                               │     │
│  │  │                     │                                │     │
│  │  │  - input_level_db   │                                │     │
│  │  │  - output_level_db  │                                │     │
│  │  │  - inference_ms     │                                │     │
│  │  │  - latency_quality_q│                                │     │
│  │  └─────────────────────┘                                │     │
│  └─────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. メモリレイアウト: TensorPool

### 2.1 設計方針

すべてのテンソルバッファを **単一の contiguous allocation** で確保。
Audio thread 内での動的メモリ確保を完全に排除する。

### 2.2 Contiguous Allocation 図

```
TensorPool: single Vec<f32> (total ~50k floats = ~200 KB)
┌────────────────────────────────────────────────────────────┐
│ Offset   Size     Name                    Shape            │
│ ──────   ─────    ──────────────────────  ──────────────── │
│ 0        80       mel_frame               [80]             │
│ 80       1        f0_frame                [1]              │
│ 81       256      content                 [256]            │
│ 337      32       acoustic_params         [32]             │
│ 369      192      spk_embed               [192]            │
│ 561      513      pred_features           [513]            │
│ 1074     513      stft_mag                [513]            │
│ 1587     513      stft_phase              [513]            │
│ 2100     800      mel_chunk (IR est.)     [80 × 10]        │
│ 2900     1024     fft_real                [1024]           │
│ 3924     1024     fft_imag                [1024]           │
│ 4948     960      ola_buffer              [960]            │
│ 5908     960      hann_window             [960] (pre-comp) │
│ 6868     960      context_buffer          [960]            │
│ 7828     960      windowed                [960]            │
│ 8788     1024     padded                  [1024]           │
│ 9812     41040    mel_filterbank          [80 × 513]       │
│ ──────                                                      │
│ Total:   50852 floats ≈ 198 KB                              │
└────────────────────────────────────────────────────────────┘
```

> **Note:** State tensors (ping-pong buffers) are allocated separately via `PingPongState`
> structs, not in the TensorPool. See §3 for state management.

### 2.3 TensorPool API (Rust)

```rust
pub struct TensorPool {
    data: Vec<f32>,
}

impl TensorPool {
    pub fn new() -> Self;
    pub fn total_floats(&self) -> usize;
    
    // Typed sub-slice accessors
    pub fn mel_frame(&self) -> &[f32];
    pub fn mel_frame_mut(&mut self) -> &mut [f32];
    pub fn f0_frame(&self) -> &[f32];
    pub fn f0_frame_mut(&mut self) -> &mut [f32];
    pub fn content(&self) -> &[f32];
    pub fn content_mut(&mut self) -> &mut [f32];
    pub fn acoustic_params(&self) -> &[f32];
    pub fn acoustic_params_mut(&mut self) -> &mut [f32];
    pub fn spk_embed(&self) -> &[f32];
    pub fn spk_embed_mut(&mut self) -> &mut [f32];
    pub fn pred_features(&self) -> &[f32];
    pub fn pred_features_mut(&mut self) -> &mut [f32];
    pub fn stft_mag(&self) -> &[f32];
    pub fn stft_mag_mut(&mut self) -> &mut [f32];
    pub fn stft_phase(&self) -> &[f32];
    pub fn stft_phase_mut(&mut self) -> &mut [f32];
    pub fn mel_chunk(&self) -> &[f32];
    pub fn mel_chunk_mut(&mut self) -> &mut [f32];
    
    // FFT/work buffers
    pub fn fft_real(&self) -> &[f32];
    pub fn fft_imag(&self) -> &[f32];
    pub fn ola_buffer(&self) -> &[f32];
    pub fn ola_buffer_mut(&mut self) -> &mut [f32];
    pub fn hann_window(&self) -> &[f32];
    pub fn context_buffer(&self) -> &[f32];
    pub fn context_buffer_mut(&mut self) -> &mut [f32];
    
    pub fn reset(&mut self);
}
```

---

## 3. State Tensor 管理: PingPongState

### 3.1 Ping-Pong Double Buffering

各 streaming model は state tensor を 2 つ保持し、フレームごとに交互に使用する。

```rust
pub struct PingPongState {
    buffer_a: Vec<f32>,
    buffer_b: Vec<f32>,
    current: bool,  // false = A is input, true = B is input
}

impl PingPongState {
    pub fn new(shape: [usize; 3]) -> Self {
        let size = shape[0] * shape[1] * shape[2];
        Self {
            buffer_a: vec![0.0; size],
            buffer_b: vec![0.0; size],
            current: false,
        }
    }
    
    pub fn input(&self) -> &[f32] {
        if self.current { &self.buffer_b } else { &self.buffer_a }
    }
    
    pub fn output(&mut self) -> &mut [f32] {
        if self.current { &mut self.buffer_a } else { &mut self.buffer_b }
    }
    
    pub fn swap(&mut self) {
        self.current = !self.current;
    }
    
    pub fn reset(&mut self) {
        self.buffer_a.fill(0.0);
        self.buffer_b.fill(0.0);
        self.current = false;
    }
}
```

### 3.2 ModelStates

```rust
struct ModelStates {
    content_encoder: PingPongState,  // [1, 256, 28]
    ir_estimator: PingPongState,     // [1, 128, 6]
    converter: PingPongState,        // [1, 384, 52]
    converter_hq: PingPongState,     // [1, 384, 46]
    vocoder: PingPongState,          // [1, 256, 14]
}
```

### 3.3 State Sizes (from constants.yaml)

| Model | State Shape | Elements | Memory (f32) |
|---|---|---|---|
| content_encoder | [1, 256, 28] | 7,168 | 28 KB |
| ir_estimator | [1, 128, 6] | 768 | 3 KB |
| converter | [1, 384, 52] | 19,968 | 78 KB |
| converter_hq | [1, 384, 46] | 17,664 | 69 KB |
| vocoder | [1, 256, 14] | 3,584 | 14 KB |
| **合計 (Live)** | | **31,488** | **~123 KB** |
| **合計 (HQ)** | | **29,184** | **~114 KB** |

---

## 4. OrtBundle: ONNX Runtime セッション管理

### 4.1 API (Rust)

```rust
pub struct OrtBundle {
    // VC streaming models (required)
    content_encoder: Session,
    ir_estimator: Session,
    converter: Session,
    converter_hq: Option<Session>,
    vocoder: Session,
    
    // TTS front-end models (optional)
    text_encoder: Option<Session>,
    duration_predictor: Option<Session>,
    f0_predictor: Option<Session>,
    content_synthesizer: Option<Session>,
}

impl OrtBundle {
    pub fn load(model_dir: &Path) -> Result<Self>;
    
    // Getters for sessions
    pub fn content_encoder(&self) -> &Session;
    pub fn ir_estimator(&self) -> &Session;
    pub fn converter(&self) -> &Session;
    pub fn converter_hq(&self) -> Option<&Session>;
    pub fn vocoder(&self) -> &Session;
    pub fn has_hq(&self) -> bool;
}
```

### 4.2 Session Options

```rust
fn build_session(model_path: impl AsRef<Path>) -> Result<Session> {
    Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path.as_ref())
        .map_err(Into::into)
}
```

---

## 5. Speaker 管理

### 5.1 SpeakerFile

```rust
pub struct SpeakerFile {
    pub spk_embed: [f32; D_SPEAKER],       // [192]
    pub lora_delta: Vec<f32>,               // [15872]
    pub metadata: SpeakerMetadata,
}

pub struct SpeakerMetadata {
    pub profile_name: String,
    pub author_name: String,
    pub co_author_name: String,
    pub licence_url: String,
    pub thumbnail_b64: String,
    pub created_at: String,
    pub description: String,
    pub source_audio_files: Vec<String>,
    pub source_sample_count: u64,
    pub training_mode: String,             // "embedding" | "finetune"
    pub checkpoint_name: String,
    pub voice_source_preset: Option<Vec<f32>>,  // [8]
}
```

### 5.2 .tmrvc_speaker v2 Format

```
Offset   Size (bytes)    Field
──────   ────────────    ──────────────────────────
0x0000   4               Magic: "TMSP" (0x544D5350)
0x0004   4               Version: uint32_le = 2
0x0008   4               embed_size: uint32_le = 192
0x000C   4               lora_size: uint32_le = 15872
0x0010   4               metadata_size: uint32_le
0x0014   4               thumbnail_size: uint32_le (always 0)
0x0018   768             spk_embed: float32_le[192]
0x0318   63488           lora_delta: float32_le[15872]
0x10118  metadata_size   metadata_json: UTF-8 JSON
         32              checksum: SHA-256
```

### 5.3 LoRA Delta Size

```
Per layer (FiLM projection LoRA):
  d_cond = d_speaker + n_acoustic_params = 192 + 32 = 224
  d_model_x2 = d_converter_hidden × 2 = 384 × 2 = 768
  lora_A: d_cond × lora_rank = 224 × 4 = 896
  lora_B: lora_rank × d_model_x2 = 4 × 768 = 3,072
  Per layer total: 896 + 3,072 = 3,968

Total: n_lora_layers × 3,968 = 4 × 3,968 = 15,872 floats
Memory: 15,872 × 4 bytes ≈ 62 KB
```

---

## 6. FrameParams: Per-Frame Parameters

```rust
pub struct FrameParams {
    pub dry_wet: f32,              // 0.0 = dry, 1.0 = wet
    pub output_gain: f32,          // Linear gain
    pub alpha_timbre: f32,         // Target timbre strength
    pub beta_prosody: f32,         // Target prosody strength
    pub gamma_articulation: f32,   // Target articulation strength
    pub latency_quality_q: f32,    // 0.0 = Live, 1.0 = Quality
    pub voice_source_alpha: f32,   // Voice source preset blend
}
```

### 6.1 Latency-Quality Trade-off

| q value | Mode | Converter | Latency | State |
|---------|------|-----------|---------|-------|
| q ≤ 0.3 | Live | converter (T=1) | ~20ms | 52 frames |
| q > 0.3 | HQ | converter_hq (T=7→1) | ~80ms | 46 frames |

Mode switching uses 100ms crossfade for glitch-free transition.

---

## 7. SharedStatus: Atomic Stats for GUI

```rust
pub struct SharedStatus {
    pub input_level_db: AtomicF32,
    pub output_level_db: AtomicF32,
    pub inference_ms: AtomicF32,
    pub inference_p50_ms: AtomicF32,
    pub inference_p95_ms: AtomicF32,
    pub frame_count: AtomicU64,
    pub overrun_count: AtomicU64,
    pub underrun_count: AtomicU64,
    pub latency_quality_q: AtomicF32,
    pub alpha_timbre: AtomicF32,
    pub beta_prosody: AtomicF32,
    pub gamma_articulation: AtomicF32,
    pub estimated_log_f0: AtomicF32,
    pub style_target_log_f0: AtomicF32,
    pub style_target_articulation: AtomicF32,
    pub style_loaded: AtomicBool,
    pub is_running: AtomicBool,
}
```

---

## 8. StreamingEngine: Central Orchestrator

### 8.1 API

```rust
pub struct StreamingEngine {
    tensor_pool: TensorPool,
    ort_bundle: Option<OrtBundle>,
    states: ModelStates,
    spk_embed: [f32; D_SPEAKER],
    lora_delta: Vec<f32>,
    style: Option<StyleFile>,
    acoustic_params_cached: [f32; N_ACOUSTIC_PARAMS],
    frame_counter: usize,
    // ... (additional fields)
    status: Option<Arc<SharedStatus>>,
    models_loaded: bool,
    speaker_loaded: bool,
    content_buffer: ContentBuffer,
    hq_mode: bool,
    voice_source_preset: Option<[f32; N_VOICE_SOURCE_PARAMS]>,
}

impl StreamingEngine {
    pub fn new(status: Option<Arc<SharedStatus>>) -> Self;
    
    // Model/speaker loading
    pub fn load_models(&mut self, dir: &Path) -> Result<()>;
    pub fn load_speaker(&mut self, path: &Path) -> Result<()>;
    pub fn load_style(&mut self, path: &Path) -> Result<()>;
    pub fn clear_style(&mut self);
    pub fn has_style(&self) -> bool;
    pub fn is_ready(&self) -> bool;
    
    // Processing
    pub fn process_one_frame(&mut self, input: &[f32], output: &mut [f32], params: &FrameParams);
    pub fn reset(&mut self);
}
```

### 8.2 process_one_frame 概要

```rust
pub fn process_one_frame(&mut self, input: &[f32], output: &mut [f32], params: &FrameParams) {
    // 0. Check readiness
    if !self.is_ready() {
        output.copy_from_slice(input);  // Bypass
        return;
    }
    
    // 1. Update context buffer for causal STFT
    self.update_context_buffer(input);
    
    // 2. Compute causal STFT → mel_frame
    self.compute_mel_frame();
    
    // 3. Estimate F0 (YIN-based, causal)
    let f0 = self.estimate_f0();
    
    // 4. Content encoder
    self.run_content_encoder();
    
    // 5. IR estimator (every 10 frames)
    if self.frame_counter % IR_UPDATE_INTERVAL == 0 {
        self.run_ir_estimator();
        // Voice source preset blend
        if let Some(preset) = &self.voice_source_preset {
            self.blend_voice_source_preset(params.voice_source_alpha);
        }
    }
    
    // 6. Style processing (F0 modification, etc.)
    self.apply_style_modifications(params);
    
    // 7. Converter (Live or HQ)
    let use_hq = params.latency_quality_q > HQ_THRESHOLD_Q && self.ort_bundle.has_hq();
    if use_hq {
        self.content_buffer.push(self.tensor_pool.content());
        if self.content_buffer.is_full() {
            self.run_converter_hq();
        } else {
            // Use causal converter during HQ buffer fill
            self.run_converter();
        }
    } else {
        self.run_converter();
    }
    
    // 8. Vocoder
    self.run_vocoder();
    
    // 9. iSTFT + Overlap-Add
    self.istft_and_ola(output);
    
    // 10. Dry/wet mix + output gain
    self.apply_mix_and_gain(input, output, params);
    
    // 11. Update status
    self.update_status();
    
    // 12. Increment frame counter, swap ping-pong states
    self.frame_counter += 1;
}
```

---

## 9. VST3 統合 (tmrvc-vst)

### 9.1 Cargo.toml

```toml
[package]
name = "tmrvc-vst"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
nih-plug = { git = "https://github.com/robbert-vdh/nih-plug" }
tmrvc-engine-rs = { path = "../tmrvc-engine-rs" }
```

### 9.2 Plugin Parameters

| Parameter | ID | Range | Default | Type |
|---|---|---|---|---|
| **Dry/Wet** | 0 | 0.0 - 1.0 | 1.0 | Float |
| **Output Gain** | 1 | -60.0 - +12.0 dB | 0.0 | Float |
| **Alpha Timbre** | 2 | 0.0 - 1.0 | 1.0 | Float |
| **Beta Prosody** | 3 | 0.0 - 1.0 | 0.0 | Float |
| **Gamma Articulation** | 4 | 0.0 - 1.0 | 0.0 | Float |
| **Latency/Quality** | 5 | 0.0 - 1.0 | 0.0 | Float |
| **Voice Source Blend** | 6 | 0.0 - 1.0 | 0.0 | Float |

### 9.3 Plugin Implementation

```rust
struct TMRVCPlugin {
    params: Arc<TMRVCParams>,
    engine: StreamingEngine,
}

impl Plugin for TMRVCPlugin {
    fn initialize(&mut self, _audio_io_layout: &AudioIOLayout, _config: &PluginConfig) -> bool {
        // Load models from TMRVC_MODEL_DIR env var or default
        let model_dir = std::env::var("TMRVC_MODEL_DIR")
            .unwrap_or_else(|_| "models/fp32".to_string());
        let _ = self.engine.load_models(Path::new(&model_dir));
        
        // Load speaker from TMRVC_SPEAKER_PATH env var or default
        let speaker_path = std::env::var("TMRVC_SPEAKER_PATH")
            .unwrap_or_else(|_| "models/test_speaker.tmrvc_speaker".to_string());
        let _ = self.engine.load_speaker(Path::new(&speaker_path));
        
        true
    }
    
    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers, _context: &mut ProcessContext) {
        for channel_samples in buffer.iter_samples() {
            let (input, output) = channel_samples.split_at_mut(1);
            let input = &input[0];
            let output = &mut output[0];
            
            let params = FrameParams {
                dry_wet: self.params.dry_wet.value(),
                output_gain: db_to_linear(self.params.output_gain.value()),
                alpha_timbre: self.params.alpha_timbre.value(),
                beta_prosody: self.params.beta_prosody.value(),
                gamma_articulation: self.params.gamma_articulation.value(),
                latency_quality_q: self.params.latency_quality.value(),
                voice_source_alpha: self.params.voice_source_blend.value(),
            };
            
            // Resample input 48kHz → 24kHz, process, resample output 24kHz → 48kHz
            // (handled internally by StreamingEngine)
            self.engine.process_one_frame(input, output, &params);
        }
    }
}
```

---

## 10. Real-time GUI (tmrvc-rt)

tmrvc-rt は egui を使用したスタンドアロン GUI アプリケーション。
StreamingEngine と SharedStatus を共有し、リアルタイムで統計を表示。

```rust
// tmrvc-rt/src/app.rs
pub struct TMRVCApp {
    engine: StreamingEngine,
    status: Arc<SharedStatus>,
    // egui state
}

impl eframe::App for TMRVCApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Display status from SharedStatus atomics
        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.label(format!("Input: {:.1} dB", self.status.input_level_db.load(Ordering::Relaxed)));
            ui.label(format!("Inference: {:.2} ms", self.status.inference_ms.load(Ordering::Relaxed)));
            // ...
        });
    }
}
```

---

## 11. Edge Cases

### 11.1 Inference Overrun

```
process_one_frame() が 10ms を超過した場合:
  - Output Ring Buffer への書き込みが遅延
  - 対処:
    1. consecutive_overruns を increment
    2. 連続 3 回超過で adaptive に q を降格
    3. 連続 10 回超過で dry bypass
  - 復帰:
    1. inference_ms < hop_time_ms × 0.8 で復帰
```

### 11.2 Speaker/Style Not Loaded

```
- is_ready() == false の場合、入力をそのまま出力 (bypass)
- GUI に "No speaker loaded" / "No models loaded" を表示
```

### 11.3 HQ Mode Transition

```
Live → HQ 遷移:
  1. content_buffer がいっぱいになるまで Live converter を使用
  2. いっぱいになったら HQ converter に切り替え
  3. 100ms crossfade で滑らかな遷移

HQ → Live 遷移:
  1. 直ちに Live converter に切り替え
  2. content_buffer をリセット
  3. 100ms crossfade
```

---

## 12. Build System

### 12.1 Cargo.toml (workspace)

```toml
[workspace]
members = ["tmrvc-engine-rs", "tmrvc-rt", "tmrvc-vst", "xtask"]
resolver = "2"

[profile.release]
opt-level = 3
lto = "thin"
```

### 12.2 tmrvc-engine-rs/Cargo.toml

```toml
[package]
name = "tmrvc-engine-rs"
version = "0.1.0"
edition = "2021"

[dependencies]
ort = { version = "2.0", features = ["load-dynamic"] }
anyhow = "1.0"
atomic_float = "1.0"
rustfft = "6.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
log = "0.4"
```

---

## 13. 整合性チェックリスト

- [x] StreamingEngine は nih-plug に非依存 (architecture.md §5.2)
- [x] ONNX Runtime は `ort` crate 経由 (CPU EP only)
- [x] TensorPool の shapes が onnx-contract.md §2-3 と一致
- [x] State tensor shapes が constants.yaml と一致
- [x] lora_delta_size = 15872 (constants.yaml, onnx-contract.md と一致)
- [x] Audio thread は RT-safe (Vec::new() は初期化時のみ)
- [x] Speaker format v2 が onnx-contract.md §6 と一致
- [x] Voice Source Preset: speaker ファイルから読み込み、ブレンド (RT-safe)
- [x] HQ mode: converter_hq.onnx が optional、lookahead=6
- [x] Latency-Quality spectrum: q パラメータで Live/HQ 切り替え
- [x] Style file: F0 modification, articulation control
- [x] SharedStatus: atomic stats for GUI
