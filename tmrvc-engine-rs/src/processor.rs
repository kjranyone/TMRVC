use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use atomic_float::AtomicF32;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::constants::*;
use crate::dsp;
use crate::ort_bundle::OrtBundle;
use crate::ping_pong::PingPongState;
use crate::speaker::SpeakerFile;
use crate::style::StyleFile;
use crate::tensor_pool::TensorPool;

/// Circular buffer for content vectors, used by HQ mode.
///
/// Stores the last `MAX_LOOKAHEAD_HOPS + 1` content frames (7 x 256 floats).
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

    /// Fill flattened content tensor [D_CONTENT x 7] in time order.
    fn fill_flat_tensor(&self, dst: &mut [f32]) {
        let capacity = MAX_LOOKAHEAD_HOPS + 1;
        assert_eq!(dst.len(), capacity * D_CONTENT);
        for i in 0..capacity {
            // Read in chronological order starting from oldest
            let read_pos = (self.write_pos + i) % capacity;
            let src_start = read_pos * D_CONTENT;
            let dst_start = i * D_CONTENT;
            dst[dst_start..dst_start + D_CONTENT]
                .copy_from_slice(&self.data[src_start..src_start + D_CONTENT]);
        }
    }
    fn reset(&mut self) {
        self.data.fill(0.0);
        self.write_pos = 0;
        self.count = 0;
    }
}

/// Model hidden states (4+1 models x 2 ping-pong buffers).
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
    /// Target timbre strength (0.0 = no target timbre, 1.0 = full target timbre).
    pub alpha_timbre: f32,
    /// Target prosody strength (0.0 = keep source F0, 1.0 = follow style F0).
    pub beta_prosody: f32,
    /// Target articulation strength (0.0 = source articulation, 1.0 = target style).
    pub gamma_articulation: f32,
    /// Latency-Quality trade-off: 0.0 = Live (low latency), 1.0 = Quality (HQ mode).
    /// When `q > HQ_THRESHOLD_Q` and a HQ converter model is loaded, the engine
    /// switches to semi-causal HQ mode with higher latency but better quality.
    pub latency_quality_q: f32,
    /// Voice source preset blend strength: 0.0 = estimated (no blend), 1.0 = full preset.
    pub voice_source_alpha: f32,
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
    pub alpha_timbre: AtomicF32,
    pub beta_prosody: AtomicF32,
    pub gamma_articulation: AtomicF32,
    pub estimated_log_f0: AtomicF32,
    pub style_target_log_f0: AtomicF32,
    pub style_target_articulation: AtomicF32,
    pub style_loaded: AtomicBool,
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
            alpha_timbre: AtomicF32::new(1.0),
            beta_prosody: AtomicF32::new(0.0),
            gamma_articulation: AtomicF32::new(0.0),
            estimated_log_f0: AtomicF32::new(0.0),
            style_target_log_f0: AtomicF32::new(0.0),
            style_target_articulation: AtomicF32::new(0.0),
            style_loaded: AtomicBool::new(false),
            is_running: AtomicBool::new(false),
        }
    }
}
/// StreamingEngine: per-frame ONNX inference + DSP pipeline.
///
/// GUI-independent - can be reused for VST3 via nih-plug.
/// Supports both Live (causal, 20ms) and HQ (semi-causal, 80ms) modes.
pub struct StreamingEngine {
    tensor_pool: TensorPool,
    ort_bundle: Option<OrtBundle>,
    states: ModelStates,
    spk_embed: [f32; D_SPEAKER],
    lora_delta: Vec<f32>,
    spk_embed_effective: [f32; D_SPEAKER],
    lora_delta_effective: Vec<f32>,
    style: Option<StyleFile>,
    smoothed_log_f0: f32,
    acoustic_params_cached: [f32; N_ACOUSTIC_PARAMS],
    frame_counter: usize,
    fft_planner: FftPlanner<f32>,
    // Scratch buffers owned by the engine (avoids borrow-splitting issues with tensor_pool)
    context_copy: Vec<f32>,
    mel_filterbank_copy: Vec<f32>,
    windowed_scratch: Vec<f32>,
    padded_scratch: Vec<f32>,
    fft_real_scratch: Vec<f32>,
    fft_imag_scratch: Vec<f32>,
    content_out: Vec<f32>,
    content_state_out: Vec<f32>,
    ir_state_out: Vec<f32>,
    features_out: Vec<f32>,
    content_7_scratch: Vec<f32>,
    conv_hq_state_out: Vec<f32>,
    conv_state_out: Vec<f32>,
    mag_out: Vec<f32>,
    phase_out: Vec<f32>,
    voc_state_out: Vec<f32>,
    time_signal: Vec<f32>,
    stft_complex_scratch: Vec<Complex<f32>>,
    istft_complex_scratch: Vec<Complex<f32>>,
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
    effective_q: f32,
    dry_bypass_frames: usize,
    last_hop_output: [f32; HOP_LENGTH],
    voice_source_preset: Option<[f32; N_VOICE_SOURCE_PARAMS]>,
}

impl StreamingEngine {
    /// Create a new engine. `status` is optional - pass `None` for VST3 use.
    pub fn new(status: Option<Arc<SharedStatus>>) -> Self {
        let mut pool = TensorPool::new();
        dsp::init_mel_filterbank(pool.mel_filterbank_mut());
        let mel_filterbank_copy = pool.mel_filterbank().to_vec();
        Self {
            tensor_pool: pool,
            ort_bundle: None,
            states: ModelStates::new(),
            spk_embed: [0.0; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            spk_embed_effective: [0.0; D_SPEAKER],
            lora_delta_effective: vec![0.0; LORA_DELTA_SIZE],
            style: None,
            smoothed_log_f0: 0.0,
            acoustic_params_cached: [0.0; N_ACOUSTIC_PARAMS],
            frame_counter: 0,
            fft_planner: FftPlanner::new(),
            context_copy: vec![0.0; WINDOW_LENGTH],
            mel_filterbank_copy,
            windowed_scratch: vec![0.0; WINDOW_LENGTH],
            padded_scratch: vec![0.0; N_FFT],
            fft_real_scratch: vec![0.0; N_FFT],
            fft_imag_scratch: vec![0.0; N_FFT],
            content_out: vec![0.0; D_CONTENT],
            content_state_out: vec![0.0; D_CONTENT * CONTENT_ENC_STATE_FRAMES],
            ir_state_out: vec![0.0; 128 * IR_EST_STATE_FRAMES],
            features_out: vec![0.0; N_FREQ_BINS],
            content_7_scratch: vec![0.0; D_CONTENT * (MAX_LOOKAHEAD_HOPS + 1)],
            conv_hq_state_out: vec![0.0; D_CONVERTER_HIDDEN * CONVERTER_HQ_STATE_FRAMES],
            conv_state_out: vec![0.0; D_CONVERTER_HIDDEN * CONVERTER_STATE_FRAMES],
            mag_out: vec![0.0; N_FREQ_BINS],
            phase_out: vec![0.0; N_FREQ_BINS],
            voc_state_out: vec![0.0; D_CONTENT * VOCODER_STATE_FRAMES],
            time_signal: vec![0.0; WINDOW_LENGTH],
            stft_complex_scratch: vec![Complex::new(0.0, 0.0); N_FFT],
            istft_complex_scratch: vec![Complex::new(0.0, 0.0); N_FFT],
            status,
            models_loaded: false,
            speaker_loaded: false,
            content_buffer: ContentBuffer::new(),
            hq_mode: false,
            crossfade_counter: 0,
            crossfade_direction: false,
            prev_features: vec![0.0; N_FREQ_BINS],
            consecutive_overruns: 0,
            effective_q: 0.0,
            dry_bypass_frames: 0,
            last_hop_output: [0.0; HOP_LENGTH],
            voice_source_preset: None,
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
        self.voice_source_preset = spk.voice_source_preset();
        self.lora_delta = spk.lora_delta;
        self.speaker_loaded = true;

        log::info!("Speaker loaded from {:?}", path);
        Ok(())
    }
    /// Load utterance style from .tmrvc_style file.
    pub fn load_style(&mut self, path: &Path) -> Result<()> {
        let style = StyleFile::load(path)?;
        if let Some(ref status) = self.status {
            status
                .style_target_log_f0
                .store(style.target_log_f0, Ordering::Relaxed);
            status
                .style_target_articulation
                .store(style.target_articulation, Ordering::Relaxed);
            status.style_loaded.store(true, Ordering::Relaxed);
        }
        self.style = Some(style);
        log::info!("Style loaded from {:?}", path);
        Ok(())
    }

    /// Clear currently loaded style.
    pub fn clear_style(&mut self) {
        self.style = None;
        if let Some(ref status) = self.status {
            status.style_target_log_f0.store(0.0, Ordering::Relaxed);
            status
                .style_target_articulation
                .store(0.0, Ordering::Relaxed);
            status.style_loaded.store(false, Ordering::Relaxed);
        }
    }

    /// Returns true if a style file is loaded.
    pub fn has_style(&self) -> bool {
        self.style.is_some()
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
            output.copy_from_slice(input);
            self.last_hop_output.copy_from_slice(output);
            if let Some(ref status) = self.status {
                status.output_level_db.store(input_rms, Ordering::Relaxed);
            }
            return;
        }

        if self.dry_bypass_frames > 0 {
            self.dry_bypass_frames -= 1;
            output.copy_from_slice(input);
            self.last_hop_output.copy_from_slice(output);
            if let Some(ref status) = self.status {
                status.output_level_db.store(input_rms, Ordering::Relaxed);
                status.latency_quality_q.store(self.effective_q, Ordering::Relaxed);
                status.frame_count.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        let target_q = params.latency_quality_q.clamp(0.0, 1.0);
        if self.effective_q < target_q {
            self.effective_q = (self.effective_q + 0.02).min(target_q);
        } else if self.effective_q > target_q {
            self.effective_q = (self.effective_q - 0.02).max(target_q);
        }

        let start = std::time::Instant::now();

        // 1. Update context buffer (shift left, append new hop)
        dsp::update_context_buffer(self.tensor_pool.context_buffer_mut(), input);

        // 2. Copy context to scratch buffer to avoid borrow conflicts
        self.context_copy
            .copy_from_slice(self.tensor_pool.context_buffer());

        // 3. Causal STFT -> frequency domain (uses scratch buffers)
        dsp::causal_stft(
            &self.context_copy,
            self.tensor_pool.hann_window(),
            &mut self.windowed_scratch,
            &mut self.padded_scratch,
            &mut self.fft_real_scratch,
            &mut self.fft_imag_scratch,
            &mut self.stft_complex_scratch,
            &mut self.fft_planner,
        );

        // 4. Compute log-mel from scratch buffers
        dsp::compute_log_mel(
            &self.fft_real_scratch,
            &self.fft_imag_scratch,
            &self.mel_filterbank_copy,
            self.tensor_pool.mel_frame_mut(),
        );

        // 5. Accumulate mel for IR estimator
        self.accumulate_mel_chunk();

        // 6. Streaming F0 estimation (log-F0). Smooth more in quality mode.
        let estimated_log_f0 = dsp::estimate_log_f0_autocorr(&self.context_copy);
        let smooth = 0.75 + 0.2 * self.effective_q;
        self.smoothed_log_f0 = if estimated_log_f0 > 0.0 {
            if self.smoothed_log_f0 > 0.0 {
                smooth * self.smoothed_log_f0 + (1.0 - smooth) * estimated_log_f0
            } else {
                estimated_log_f0
            }
        } else {
            0.0
        };

        let mut f0_for_model = self.smoothed_log_f0;
        if let Some(style) = &self.style {
            let beta = params.beta_prosody.clamp(0.0, 1.0);
            if f0_for_model > 0.0 && style.target_log_f0 > 0.0 {
                f0_for_model = (1.0 - beta) * f0_for_model + beta * style.target_log_f0;
            }
        }
        self.tensor_pool.f0_frame_mut()[0] = f0_for_model;
        if let Some(ref status) = self.status {
            status
                .estimated_log_f0
                .store(self.smoothed_log_f0, Ordering::Relaxed);
        }

        // Run ONNX inference
        if let Some(ref mut bundle) = self.ort_bundle {
            // 7. Content encoder (always runs T=1)
            self.content_out.fill(0.0);
            self.content_state_out.fill(0.0);
            if let Err(e) = bundle.run_content_encoder(
                self.tensor_pool.mel_frame(),
                self.tensor_pool.f0_frame(),
                self.states.content_encoder.input(),
                &mut self.content_out,
                &mut self.content_state_out,
            ) {
                log::warn!("content_encoder failed: {}", e);
                fallback_output(&self.last_hop_output, input, output);
                return;
            }
            self.states
                .content_encoder
                .output()
                .copy_from_slice(&self.content_state_out);
            self.states.content_encoder.swap();
            self.tensor_pool.content_mut().copy_from_slice(&self.content_out);

            // 7b. Push content to buffer (for HQ mode)
            self.content_buffer.push(&self.content_out);

            // 8. IR estimator (dynamic interval: live=10, quality=5). Skip when overloaded.
            let ir_update_interval = if self.effective_q > HQ_THRESHOLD_Q {
                (IR_UPDATE_INTERVAL / 2).max(1)
            } else {
                IR_UPDATE_INTERVAL
            };
            let skip_ir_update = self.consecutive_overruns > 3;
            if !skip_ir_update && self.frame_counter % ir_update_interval == 0 {
                let mut acoustic_out = [0.0f32; N_ACOUSTIC_PARAMS];
                self.ir_state_out.fill(0.0);
                if let Err(e) = bundle.run_ir_estimator(
                    self.tensor_pool.mel_chunk(),
                    self.states.ir_estimator.input(),
                    &mut acoustic_out,
                    &mut self.ir_state_out,
                ) {
                    log::warn!("ir_estimator failed: {}", e);
                } else {
                    self.states
                        .ir_estimator
                        .output()
                        .copy_from_slice(&self.ir_state_out);
                    self.states.ir_estimator.swap();
                    self.acoustic_params_cached = acoustic_out;
                    self.tensor_pool.acoustic_params_mut().copy_from_slice(&acoustic_out);
                }
            }

            // 9. Determine target mode from effective latency-quality q
            let target_hq = self.effective_q > HQ_THRESHOLD_Q && bundle.has_hq_converter();

            // Initiate mode switch with crossfade
            if target_hq != self.hq_mode && self.crossfade_counter == 0 {
                self.crossfade_counter = CROSSFADE_FRAMES;
                self.crossfade_direction = target_hq;
                self.prev_features
                    .copy_from_slice(self.tensor_pool.pred_features());
            }

            // 9a. Apply timbre strength to speaker conditioning.
            let alpha = params.alpha_timbre.clamp(0.0, 1.0);
            for i in 0..D_SPEAKER {
                self.spk_embed_effective[i] = self.spk_embed[i] * alpha;
            }
            for i in 0..LORA_DELTA_SIZE {
                self.lora_delta_effective[i] = self.lora_delta[i] * alpha;
            }

            // 9b. Blend voice source preset into acoustic params (stack copy)
            let acoustic_for_converter = {
                let mut p = self.acoustic_params_cached;
                if let Some(ref preset) = self.voice_source_preset {
                    let alpha = params.voice_source_alpha.clamp(0.0, 1.0);
                    if alpha > 0.0 {
                        for i in 0..N_VOICE_SOURCE_PARAMS {
                            p[N_IR_PARAMS + i] =
                                (1.0 - alpha) * p[N_IR_PARAMS + i] + alpha * preset[i];
                        }
                    }
                }
                p
            };

            // 9c. Run converter (live or HQ)
            self.features_out.fill(0.0);
            if self.hq_mode && self.content_buffer.is_full() {
                self.content_buffer
                    .fill_flat_tensor(&mut self.content_7_scratch);
                self.conv_hq_state_out.fill(0.0);
                if let Err(e) = bundle.run_converter_hq(
                    &self.content_7_scratch,
                    &self.spk_embed_effective,
                    &self.lora_delta_effective,
                    &acoustic_for_converter,
                    self.states.converter_hq.input(),
                    &mut self.features_out,
                    &mut self.conv_hq_state_out,
                ) {
                    log::warn!("converter_hq failed: {}", e);
                    fallback_output(&self.last_hop_output, input, output);
                    return;
                }
                self.states
                    .converter_hq
                    .output()
                    .copy_from_slice(&self.conv_hq_state_out);
                self.states.converter_hq.swap();
            } else {
                self.conv_state_out.fill(0.0);
                if let Err(e) = bundle.run_converter(
                    self.tensor_pool.content(),
                    &self.spk_embed_effective,
                    &self.lora_delta_effective,
                    &acoustic_for_converter,
                    self.states.converter.input(),
                    &mut self.features_out,
                    &mut self.conv_state_out,
                ) {
                    log::warn!("converter failed: {}", e);
                    fallback_output(&self.last_hop_output, input, output);
                    return;
                }
                self.states
                    .converter
                    .output()
                    .copy_from_slice(&self.conv_state_out);
                self.states.converter.swap();
            }

            if !all_finite(&self.features_out) {
                log::warn!("non-finite converter output detected");
                fallback_output(&self.last_hop_output, input, output);
                return;
            }

            // 9d. Crossfade blending between old and new features
            if self.crossfade_counter > 0 {
                let alpha = 1.0 - (self.crossfade_counter as f32 / CROSSFADE_FRAMES as f32);
                for i in 0..N_FREQ_BINS {
                    self.features_out[i] =
                        (1.0 - alpha) * self.prev_features[i] + alpha * self.features_out[i];
                }
                self.prev_features.copy_from_slice(&self.features_out);
                self.crossfade_counter -= 1;
                if self.crossfade_counter == 0 {
                    self.hq_mode = self.crossfade_direction;
                }
            }

            // 9e. Articulation styling on converter output.
            if let Some(style) = &self.style {
                let gamma = params.gamma_articulation.clamp(0.0, 1.0);
                if gamma > 0.0 {
                    let src_articulation =
                        dsp::mel_articulation_proxy(self.tensor_pool.mel_frame());
                    let delta = (style.target_articulation - src_articulation) * gamma;
                    apply_articulation_tilt(&mut self.features_out, delta);
                }
            }

            self.tensor_pool
                .pred_features_mut()
                .copy_from_slice(&self.features_out);

            // 10. Vocoder
            self.mag_out.fill(0.0);
            self.phase_out.fill(0.0);
            self.voc_state_out.fill(0.0);
            if let Err(e) = bundle.run_vocoder(
                self.tensor_pool.pred_features(),
                self.states.vocoder.input(),
                &mut self.mag_out,
                &mut self.phase_out,
                &mut self.voc_state_out,
            ) {
                log::warn!("vocoder failed: {}", e);
                fallback_output(&self.last_hop_output, input, output);
                return;
            }
            self.states.vocoder.output().copy_from_slice(&self.voc_state_out);
            self.states.vocoder.swap();
            self.tensor_pool.stft_mag_mut().copy_from_slice(&self.mag_out);
            self.tensor_pool
                .stft_phase_mut()
                .copy_from_slice(&self.phase_out);

            if !all_finite(self.tensor_pool.stft_mag()) || !all_finite(self.tensor_pool.stft_phase()) {
                log::warn!("non-finite vocoder output detected");
                fallback_output(&self.last_hop_output, input, output);
                return;
            }
        }

        // 11. iSTFT -> time domain
        dsp::istft(
            self.tensor_pool.stft_mag(),
            self.tensor_pool.stft_phase(),
            self.tensor_pool.hann_window(),
            &mut self.time_signal,
            &mut self.istft_complex_scratch,
            &mut self.fft_planner,
        );

        if !all_finite(&self.time_signal) {
            log::warn!("non-finite time-domain frame detected");
            fallback_output(&self.last_hop_output, input, output);
            return;
        }

        // 12. Overlap-Add
        let mut hop_output = [0.0f32; HOP_LENGTH];
        dsp::overlap_add(
            &self.time_signal,
            self.tensor_pool.ola_buffer_mut(),
            &mut hop_output,
        );

        if !all_finite(&hop_output) {
            log::warn!("non-finite overlap-add output detected");
            fallback_output(&self.last_hop_output, input, output);
            return;
        }

        // 13. Dry/Wet mix
        let dw = params.dry_wet;
        let gain = params.output_gain;
        for i in 0..HOP_LENGTH {
            output[i] = (dw * hop_output[i] + (1.0 - dw) * input[i]) * gain;
        }

        if !all_finite(output) {
            log::warn!("non-finite mixed output detected");
            fallback_output(&self.last_hop_output, input, output);
            return;
        }

        // Update status
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        let hop_ms = HOP_LENGTH as f32 / SAMPLE_RATE as f32 * 1000.0;
        if elapsed > hop_ms {
            self.consecutive_overruns += 1;
        } else {
            self.consecutive_overruns = 0;
        }

        if self.consecutive_overruns > 3 {
            self.effective_q = (self.effective_q - 0.2).max(0.0);
            if self.hq_mode {
                self.crossfade_counter = CROSSFADE_FRAMES;
                self.crossfade_direction = false; // to live
                log::warn!("Overrun detected, degrading to live mode");
            }
        }
        if self.consecutive_overruns > 10 {
            self.effective_q = 0.0;
            self.dry_bypass_frames = self.dry_bypass_frames.max(100);
            log::warn!("Sustained overrun detected, enabling temporary dry bypass");
        }

        if let Some(ref status) = self.status {
            status.inference_ms.store(elapsed, Ordering::Relaxed);
            status
                .output_level_db
                .store(rms_db(output), Ordering::Relaxed);
            status.latency_quality_q.store(self.effective_q, Ordering::Relaxed);
            status.frame_count.fetch_add(1, Ordering::Relaxed);

            if elapsed > hop_ms {
                status.overrun_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.last_hop_output.copy_from_slice(output);
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
        self.acoustic_params_cached = [0.0; N_ACOUSTIC_PARAMS];
        self.frame_counter = 0;
        self.context_copy.fill(0.0);
        self.content_buffer.reset();
        self.hq_mode = false;
        self.crossfade_counter = 0;
        self.crossfade_direction = false;
        self.prev_features.fill(0.0);
        self.consecutive_overruns = 0;
        self.effective_q = 0.0;
        self.dry_bypass_frames = 0;
        self.last_hop_output.fill(0.0);
        self.smoothed_log_f0 = 0.0;
        self.spk_embed_effective.fill(0.0);
        self.lora_delta_effective.fill(0.0);
        dsp::init_mel_filterbank(self.tensor_pool.mel_filterbank_mut());
        self.mel_filterbank_copy
            .copy_from_slice(self.tensor_pool.mel_filterbank());
        self.content_out.fill(0.0);
        self.content_state_out.fill(0.0);
        self.ir_state_out.fill(0.0);
        self.features_out.fill(0.0);
        self.content_7_scratch.fill(0.0);
        self.conv_hq_state_out.fill(0.0);
        self.conv_state_out.fill(0.0);
        self.mag_out.fill(0.0);
        self.phase_out.fill(0.0);
        self.voc_state_out.fill(0.0);
        self.time_signal.fill(0.0);
    }

    /// Accumulate current mel frame into mel_chunk for IR estimator.
    fn accumulate_mel_chunk(&mut self) {
        let slot = self.frame_counter % IR_UPDATE_INTERVAL;
        let mut mel = [0.0f32; N_MELS];
        mel.copy_from_slice(self.tensor_pool.mel_frame());
        let chunk = self.tensor_pool.mel_chunk_mut();
        // mel_chunk layout: [N_MELS x IR_UPDATE_INTERVAL], column-major
        for m in 0..N_MELS {
            chunk[m * IR_UPDATE_INTERVAL + slot] = mel[m];
        }
    }
}

fn fallback_output(last_output: &[f32], input: &[f32], output: &mut [f32]) {
    if all_finite(last_output) {
        output.copy_from_slice(last_output);
    } else {
        output.copy_from_slice(input);
    }
}

fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn apply_articulation_tilt(features: &mut [f32], delta: f32) {
    // Keep the correction conservative since feature scale depends on model export.
    let gain = delta.clamp(-4.0, 4.0) * 0.08;
    let denom = (features.len().saturating_sub(1)).max(1) as f32;
    for (i, v) in features.iter_mut().enumerate() {
        let x = i as f32 / denom; // 0.0 (low) .. 1.0 (high)
        *v += gain * (x - 0.5);
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


