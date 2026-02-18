use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use atomic_float::AtomicF32;
use rustfft::FftPlanner;

use crate::constants::*;
use crate::dsp;
use crate::ort_bundle::OrtBundle;
use crate::ping_pong::PingPongState;
use crate::speaker::SpeakerFile;
use crate::tensor_pool::TensorPool;

/// Circular buffer for content vectors, used by HQ mode.
///
/// Stores the last `MAX_LOOKAHEAD_HOPS + 1` content frames (7 × 256 floats).
struct ContentBuffer {
    data: Vec<f32>,
    write_pos: usize,
    count: usize,
}

impl ContentBuffer {
    fn new() -> Self {
        let capacity = MAX_LOOKAHEAD_HOPS + 1; // 7
        Self {
            data: vec![0.0; capacity * D_CONTENT],
            write_pos: 0,
            count: 0,
        }
    }

    fn push(&mut self, content: &[f32]) {
        let start = self.write_pos * D_CONTENT;
        self.data[start..start + D_CONTENT].copy_from_slice(content);
        self.write_pos = (self.write_pos + 1) % (MAX_LOOKAHEAD_HOPS + 1);
        if self.count < MAX_LOOKAHEAD_HOPS + 1 {
            self.count += 1;
        }
    }

    fn is_full(&self) -> bool {
        self.count >= MAX_LOOKAHEAD_HOPS + 1
    }

    /// Return flattened content tensor [D_CONTENT × 7] in time order.
    fn as_flat_tensor(&self) -> Vec<f32> {
        let capacity = MAX_LOOKAHEAD_HOPS + 1;
        let mut result = vec![0.0f32; capacity * D_CONTENT];
        for i in 0..capacity {
            // Read in chronological order starting from oldest
            let read_pos = (self.write_pos + i) % capacity;
            let src_start = read_pos * D_CONTENT;
            let dst_start = i * D_CONTENT;
            result[dst_start..dst_start + D_CONTENT]
                .copy_from_slice(&self.data[src_start..src_start + D_CONTENT]);
        }
        result
    }

    fn reset(&mut self) {
        self.data.fill(0.0);
        self.write_pos = 0;
        self.count = 0;
    }
}

/// Model hidden states (4+1 models × 2 ping-pong buffers).
struct ModelStates {
    content_encoder: PingPongState,
    ir_estimator: PingPongState,
    converter: PingPongState,
    converter_hq: PingPongState,
    vocoder: PingPongState,
}

impl ModelStates {
    fn new() -> Self {
        Self {
            content_encoder: PingPongState::new([1, D_CONTENT, CONTENT_ENC_STATE_FRAMES]),
            ir_estimator: PingPongState::new([1, 128, IR_EST_STATE_FRAMES]),
            converter: PingPongState::new([1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES]),
            converter_hq: PingPongState::new([1, D_CONVERTER_HIDDEN, CONVERTER_HQ_STATE_FRAMES]),
            vocoder: PingPongState::new([1, D_CONTENT, VOCODER_STATE_FRAMES]),
        }
    }

    fn reset(&mut self) {
        self.content_encoder.reset();
        self.ir_estimator.reset();
        self.converter.reset();
        self.converter_hq.reset();
        self.vocoder.reset();
    }
}

/// Per-frame parameters passed by the caller.
pub struct FrameParams {
    pub dry_wet: f32,
    pub output_gain: f32,
    /// Latency-Quality trade-off: 0.0 = Live (low latency), 1.0 = Quality (HQ mode).
    /// When `q > HQ_THRESHOLD_Q` and a HQ converter model is loaded, the engine
    /// switches to semi-causal HQ mode with higher latency but better quality.
    pub latency_quality_q: f32,
}

/// Shared atomic status values read by the GUI.
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
    pub is_running: AtomicBool,
}

impl SharedStatus {
    pub fn new() -> Self {
        Self {
            input_level_db: AtomicF32::new(-100.0),
            output_level_db: AtomicF32::new(-100.0),
            inference_ms: AtomicF32::new(0.0),
            inference_p50_ms: AtomicF32::new(0.0),
            inference_p95_ms: AtomicF32::new(0.0),
            frame_count: AtomicU64::new(0),
            overrun_count: AtomicU64::new(0),
            underrun_count: AtomicU64::new(0),
            latency_quality_q: AtomicF32::new(0.0),
            is_running: AtomicBool::new(false),
        }
    }
}

/// StreamingEngine: per-frame ONNX inference + DSP pipeline.
///
/// GUI-independent — can be reused for VST3 via nih-plug.
/// Supports both Live (causal, 20ms) and HQ (semi-causal, 80ms) modes.
pub struct StreamingEngine {
    tensor_pool: TensorPool,
    ort_bundle: Option<OrtBundle>,
    states: ModelStates,
    spk_embed: [f32; D_SPEAKER],
    ir_params_cached: [f32; N_IR_PARAMS],
    frame_counter: usize,
    fft_planner: FftPlanner<f32>,
    // Scratch buffers owned by the engine (avoids borrow-splitting issues with tensor_pool)
    context_copy: Vec<f32>,
    windowed_scratch: Vec<f32>,
    padded_scratch: Vec<f32>,
    fft_real_scratch: Vec<f32>,
    fft_imag_scratch: Vec<f32>,
    status: Option<Arc<SharedStatus>>,
    models_loaded: bool,
    speaker_loaded: bool,
    // HQ mode state
    content_buffer: ContentBuffer,
    hq_mode: bool,
    crossfade_counter: usize,
    crossfade_direction: bool, // true = transitioning TO HQ
    prev_features: Vec<f32>,
    consecutive_overruns: usize,
}

impl StreamingEngine {
    /// Create a new engine. `status` is optional — pass `None` for VST3 use.
    pub fn new(status: Option<Arc<SharedStatus>>) -> Self {
        let mut pool = TensorPool::new();
        dsp::init_mel_filterbank(pool.mel_filterbank_mut());

        Self {
            tensor_pool: pool,
            ort_bundle: None,
            states: ModelStates::new(),
            spk_embed: [0.0; D_SPEAKER],
            ir_params_cached: [0.0; N_IR_PARAMS],
            frame_counter: 0,
            fft_planner: FftPlanner::new(),
            context_copy: vec![0.0; WINDOW_LENGTH],
            windowed_scratch: vec![0.0; WINDOW_LENGTH],
            padded_scratch: vec![0.0; N_FFT],
            fft_real_scratch: vec![0.0; N_FFT],
            fft_imag_scratch: vec![0.0; N_FFT],
            status,
            models_loaded: false,
            speaker_loaded: false,
            content_buffer: ContentBuffer::new(),
            hq_mode: false,
            crossfade_counter: 0,
            crossfade_direction: false,
            prev_features: vec![0.0; N_FREQ_BINS],
            consecutive_overruns: 0,
        }
    }

    /// Load ONNX models from directory.
    pub fn load_models(&mut self, dir: &Path) -> Result<()> {
        let bundle = OrtBundle::load(dir)?;
        self.ort_bundle = Some(bundle);
        self.models_loaded = true;
        self.reset();
        log::info!("Models loaded from {:?}", dir);
        Ok(())
    }

    /// Load speaker embedding from .tmrvc_speaker file.
    pub fn load_speaker(&mut self, path: &Path) -> Result<()> {
        let spk = SpeakerFile::load(path)?;
        self.spk_embed = spk.spk_embed;
        self.speaker_loaded = true;
        log::info!("Speaker loaded from {:?}", path);
        Ok(())
    }

    /// Returns true if both models and speaker are loaded.
    pub fn is_ready(&self) -> bool {
        self.models_loaded && self.speaker_loaded
    }

    /// Process one hop frame (HOP_LENGTH samples in, HOP_LENGTH samples out).
    ///
    /// `params` provides dry/wet mix and output gain.
    /// If models/speaker aren't loaded, passes through dry signal.
    pub fn process_one_frame(&mut self, input: &[f32], output: &mut [f32], params: &FrameParams) {
        assert_eq!(input.len(), HOP_LENGTH);
        assert_eq!(output.len(), HOP_LENGTH);

        // Compute input level
        let input_rms = rms_db(input);
        if let Some(ref status) = self.status {
            status.input_level_db.store(input_rms, Ordering::Relaxed);
        }

        if !self.is_ready() {
            // Bypass: copy dry input to output
            output.copy_from_slice(input);
            if let Some(ref status) = self.status {
                status.output_level_db.store(input_rms, Ordering::Relaxed);
            }
            return;
        }

        let start = std::time::Instant::now();

        // 1. Update context buffer (shift left, append new hop)
        dsp::update_context_buffer(self.tensor_pool.context_buffer_mut(), input);

        // 2. Copy context and hann window to scratch buffers to avoid borrow conflicts
        self.context_copy
            .copy_from_slice(self.tensor_pool.context_buffer());
        let mut hann_copy = vec![0.0f32; WINDOW_LENGTH];
        hann_copy.copy_from_slice(self.tensor_pool.hann_window());

        // 3. Causal STFT → frequency domain (uses scratch buffers)
        dsp::causal_stft(
            &self.context_copy,
            &hann_copy,
            &mut self.windowed_scratch,
            &mut self.padded_scratch,
            &mut self.fft_real_scratch,
            &mut self.fft_imag_scratch,
            &mut self.fft_planner,
        );

        // 4. Compute log-mel from scratch buffers
        {
            let filterbank = self.tensor_pool.mel_filterbank().to_vec();
            dsp::compute_log_mel(
                &self.fft_real_scratch,
                &self.fft_imag_scratch,
                &filterbank,
                self.tensor_pool.mel_frame_mut(),
            );
        }

        // 5. Accumulate mel for IR estimator
        self.accumulate_mel_chunk();

        // 6. F0 = 0 (placeholder — real F0 estimation would go here)
        self.tensor_pool.f0_frame_mut()[0] = 0.0;

        // Run ONNX inference
        if let Some(ref mut bundle) = self.ort_bundle {
            // 7. Content encoder (always runs T=1)
            let mut content_out = vec![0.0f32; D_CONTENT];
            let mut state_out = vec![0.0f32; self.states.content_encoder.len()];
            let _ = bundle.run_content_encoder(
                self.tensor_pool.mel_frame(),
                self.tensor_pool.f0_frame(),
                self.states.content_encoder.input(),
                &mut content_out,
                &mut state_out,
            );
            self.states
                .content_encoder
                .output()
                .copy_from_slice(&state_out);
            self.states.content_encoder.swap();
            self.tensor_pool.content_mut().copy_from_slice(&content_out);

            // 7b. Push content to buffer (for HQ mode)
            self.content_buffer.push(&content_out);

            // 8. IR estimator (every IR_UPDATE_INTERVAL frames)
            if self.frame_counter % IR_UPDATE_INTERVAL == 0 {
                let mut ir_out = [0.0f32; N_IR_PARAMS];
                let mut ir_state_out = vec![0.0f32; self.states.ir_estimator.len()];
                let _ = bundle.run_ir_estimator(
                    self.tensor_pool.mel_chunk(),
                    self.states.ir_estimator.input(),
                    &mut ir_out,
                    &mut ir_state_out,
                );
                self.states
                    .ir_estimator
                    .output()
                    .copy_from_slice(&ir_state_out);
                self.states.ir_estimator.swap();
                self.ir_params_cached = ir_out;
                self.tensor_pool.ir_params_mut().copy_from_slice(&ir_out);
            }

            // 9. Determine target mode from latency-quality q
            let q = params.latency_quality_q;
            let target_hq = q > HQ_THRESHOLD_Q as f32 && bundle.has_hq_converter();

            // Initiate mode switch with crossfade
            if target_hq != self.hq_mode && self.crossfade_counter == 0 {
                self.crossfade_counter = CROSSFADE_FRAMES;
                self.crossfade_direction = target_hq;
                // Save current features for crossfade blending
                self.prev_features
                    .copy_from_slice(self.tensor_pool.pred_features());
            }

            // 9b. Run converter (live or HQ)
            let mut features_out = vec![0.0f32; N_FREQ_BINS];
            if self.hq_mode && self.content_buffer.is_full() {
                // HQ mode: run semi-causal converter with 7-frame content window
                let content_7 = self.content_buffer.as_flat_tensor();
                let mut conv_hq_state_out =
                    vec![0.0f32; self.states.converter_hq.len()];
                let _ = bundle.run_converter_hq(
                    &content_7,
                    &self.spk_embed,
                    &self.ir_params_cached,
                    self.states.converter_hq.input(),
                    &mut features_out,
                    &mut conv_hq_state_out,
                );
                self.states
                    .converter_hq
                    .output()
                    .copy_from_slice(&conv_hq_state_out);
                self.states.converter_hq.swap();
            } else {
                // Live mode: run causal converter with T=1
                let mut conv_state_out = vec![0.0f32; self.states.converter.len()];
                let _ = bundle.run_converter(
                    self.tensor_pool.content(),
                    &self.spk_embed,
                    &self.ir_params_cached,
                    self.states.converter.input(),
                    &mut features_out,
                    &mut conv_state_out,
                );
                self.states
                    .converter
                    .output()
                    .copy_from_slice(&conv_state_out);
                self.states.converter.swap();
            }

            // 9c. Crossfade blending between old and new features
            if self.crossfade_counter > 0 {
                let alpha =
                    1.0 - (self.crossfade_counter as f32 / CROSSFADE_FRAMES as f32);
                for i in 0..N_FREQ_BINS {
                    features_out[i] =
                        (1.0 - alpha) * self.prev_features[i] + alpha * features_out[i];
                }
                self.prev_features.copy_from_slice(&features_out);
                self.crossfade_counter -= 1;
                if self.crossfade_counter == 0 {
                    self.hq_mode = self.crossfade_direction;
                }
            }

            self.tensor_pool
                .pred_features_mut()
                .copy_from_slice(&features_out);

            // 10. Vocoder
            let mut mag_out = vec![0.0f32; N_FREQ_BINS];
            let mut phase_out = vec![0.0f32; N_FREQ_BINS];
            let mut voc_state_out = vec![0.0f32; self.states.vocoder.len()];
            let _ = bundle.run_vocoder(
                self.tensor_pool.pred_features(),
                self.states.vocoder.input(),
                &mut mag_out,
                &mut phase_out,
                &mut voc_state_out,
            );
            self.states.vocoder.output().copy_from_slice(&voc_state_out);
            self.states.vocoder.swap();
            self.tensor_pool.stft_mag_mut().copy_from_slice(&mag_out);
            self.tensor_pool
                .stft_phase_mut()
                .copy_from_slice(&phase_out);
        }

        // 11. iSTFT → time domain
        let mag = self.tensor_pool.stft_mag().to_vec();
        let phase = self.tensor_pool.stft_phase().to_vec();
        let mut time_signal = vec![0.0f32; WINDOW_LENGTH];
        dsp::istft(
            &mag,
            &phase,
            &hann_copy,
            &mut time_signal,
            &mut self.fft_planner,
        );

        // 12. Overlap-Add
        let mut hop_output = [0.0f32; HOP_LENGTH];
        dsp::overlap_add(
            &time_signal,
            self.tensor_pool.ola_buffer_mut(),
            &mut hop_output,
        );

        // 13. Dry/Wet mix
        let dw = params.dry_wet;
        let gain = params.output_gain;
        for i in 0..HOP_LENGTH {
            output[i] = (dw * hop_output[i] + (1.0 - dw) * input[i]) * gain;
        }

        // Update status
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        let hop_ms = HOP_LENGTH as f32 / SAMPLE_RATE as f32 * 1000.0;
        if elapsed > hop_ms {
            self.consecutive_overruns += 1;
        } else {
            self.consecutive_overruns = 0;
        }

        // Adaptive degradation: force switch to live mode on sustained overruns
        if self.consecutive_overruns > 3 && self.hq_mode {
            self.crossfade_counter = CROSSFADE_FRAMES;
            self.crossfade_direction = false; // to live
            log::warn!("Overrun detected, degrading to live mode");
        }

        if let Some(ref status) = self.status {
            status.inference_ms.store(elapsed, Ordering::Relaxed);
            status
                .output_level_db
                .store(rms_db(output), Ordering::Relaxed);
            status.frame_count.fetch_add(1, Ordering::Relaxed);

            if elapsed > hop_ms {
                status.overrun_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.frame_counter += 1;
    }

    /// Returns true if the engine is currently in HQ (semi-causal) mode.
    pub fn is_hq_mode(&self) -> bool {
        self.hq_mode
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.states.reset();
        self.tensor_pool.reset();
        self.ir_params_cached = [0.0; N_IR_PARAMS];
        self.frame_counter = 0;
        self.context_copy.fill(0.0);
        self.content_buffer.reset();
        self.hq_mode = false;
        self.crossfade_counter = 0;
        self.crossfade_direction = false;
        self.prev_features.fill(0.0);
        self.consecutive_overruns = 0;
        dsp::init_mel_filterbank(self.tensor_pool.mel_filterbank_mut());
    }

    /// Accumulate current mel frame into mel_chunk for IR estimator.
    fn accumulate_mel_chunk(&mut self) {
        let slot = self.frame_counter % IR_UPDATE_INTERVAL;
        let mel: Vec<f32> = self.tensor_pool.mel_frame().to_vec();
        let chunk = self.tensor_pool.mel_chunk_mut();
        // mel_chunk layout: [N_MELS x IR_UPDATE_INTERVAL], column-major
        for m in 0..N_MELS {
            chunk[m * IR_UPDATE_INTERVAL + slot] = mel[m];
        }
    }
}

/// Compute RMS level in dB.
pub fn rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -100.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    if rms < 1e-10 {
        -100.0
    } else {
        20.0 * rms.log10()
    }
}
