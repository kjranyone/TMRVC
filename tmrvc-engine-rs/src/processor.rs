//! Streaming engine for Codec-Latent pipeline.
//!
//! Token-based streaming voice conversion:
//!   audio → codec_encoder → tokens → token_model → tokens → codec_decoder → audio
//!
//! Frame size: 480 samples (20ms @ 24kHz)
//! Token rate: 50 Hz (200 tokens/sec with 4 codebooks)

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use atomic_float::AtomicF32;
use rand::Rng;

use crate::character::CharacterFile;
use crate::constants::*;
use crate::f0_tracker::F0Tracker;
use crate::ort_bundle::OrtBundle;
use crate::ping_pong::PingPongState;
use crate::speaker::SpeakerFile;
use crate::style::StyleFile;

// ============================================================================
// Public Types
// ============================================================================

/// Per-frame processing parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameParams {
    /// Dry/wet mix: 0.0 = input only, 1.0 = processed only
    pub dry_wet: f32,
    /// Output gain (linear)
    pub output_gain: f32,
    /// Target timbre strength (0.0 - 1.0)
    pub alpha_timbre: f32,
    /// Target prosody strength (0.0 - 1.0)
    pub beta_prosody: f32,
    /// Target articulation strength (0.0 - 1.0)
    pub gamma_articulation: f32,
    /// Voice source preset blend (0.0 - 1.0)
    pub voice_source_alpha: f32,
    /// Latency/quality trade-off: 0.0 = low latency, 1.0 = high quality
    pub latency_quality_q: f32,
    /// Pitch shift in semitones (-24 to +24)
    pub pitch_shift: f32,
    /// CFG scale for classifier-free guidance
    pub cfg_scale: f32,
    /// Temperature for acoustic token sampling
    pub temperature_a: f32,
    /// Temperature for control token sampling
    pub temperature_b: f32,
    /// Top-k for acoustic token sampling
    pub top_k_a: usize,
    /// Top-k for control token sampling
    pub top_k_b: usize,
    /// Voice state parameters [breathiness, tension, arousal, valence, roughness, voicing, energy, rate]
    pub voice_state: [f32; D_VOICE_STATE],
}

/// Shared status for real-time monitoring (atomic values for thread-safe access).
pub struct SharedStatus {
    // Audio levels
    pub input_level_db: AtomicF32,
    pub output_level_db: AtomicF32,

    // Timing
    pub inference_ms: AtomicF32,
    pub inference_p50_ms: AtomicF32,
    pub inference_p95_ms: AtomicF32,

    // Counters
    pub frame_count: AtomicU64,
    pub overrun_count: AtomicU64,
    pub underrun_count: AtomicU64,

    // Current parameters (mirrored for GUI)
    pub latency_quality_q: AtomicF32,
    pub alpha_timbre: AtomicF32,
    pub beta_prosody: AtomicF32,
    pub gamma_articulation: AtomicF32,

    // Estimated values
    pub estimated_log_f0: AtomicF32,

    // Style targets (if style loaded)
    pub style_target_log_f0: AtomicF32,
    pub style_target_articulation: AtomicF32,

    // Flags
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

    pub fn reset(&self) {
        self.input_level_db.store(-100.0, Ordering::SeqCst);
        self.output_level_db.store(-100.0, Ordering::SeqCst);
        self.inference_ms.store(0.0, Ordering::SeqCst);
        self.frame_count.store(0, Ordering::SeqCst);
        self.overrun_count.store(0, Ordering::SeqCst);
        self.underrun_count.store(0, Ordering::SeqCst);
    }
}

impl Default for SharedStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing statistics.
pub struct TimingStats {
    pub frame_count: AtomicU64,
    pub overrun_count: AtomicU64,
    pub avg_frame_us: AtomicF32,
    pub max_frame_us: AtomicF32,
}

impl TimingStats {
    pub fn new() -> Self {
        Self {
            frame_count: AtomicU64::new(0),
            overrun_count: AtomicU64::new(0),
            avg_frame_us: AtomicF32::new(0.0),
            max_frame_us: AtomicF32::new(0.0),
        }
    }

    pub fn reset(&self) {
        self.frame_count.store(0, Ordering::SeqCst);
        self.overrun_count.store(0, Ordering::SeqCst);
        self.avg_frame_us.store(0.0, Ordering::SeqCst);
        self.max_frame_us.store(0.0, Ordering::SeqCst);
    }
}

impl Default for TimingStats {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Internal Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct PointerState {
    pub text_index: usize,
    pub progress: f32,
    pub frames_on_current_unit: usize,
    pub frames_generated: u64,
    pub stall_frames: usize,
    pub max_frames_per_unit: usize,
    pub skip_protection_threshold: f32,
    pub forced_advance_count: u64,
    pub skip_protection_count: u64,
}

impl Default for PointerState {
    fn default() -> Self {
        Self {
            text_index: 0,
            progress: 0.0,
            frames_on_current_unit: 0,
            frames_generated: 0,
            stall_frames: 0,
            max_frames_per_unit: 400, // Matching Python default
            skip_protection_threshold: 0.5,
            forced_advance_count: 0,
            skip_protection_count: 0,
        }
    }
}

impl PointerState {
    /// SOTA: Update pointer with continuous integration (GEMINI.md Mandate).
    pub fn step(&mut self, advance_prob: f32, progress_delta: f32, boundary_confidence: f32, hold_bias: f32) -> bool {
        self.frames_on_current_unit += 1;
        self.frames_generated += 1;

        // Integration with drag
        let drag = (hold_bias * 0.02).max(0.0);
        self.progress += (progress_delta - drag).max(0.0);

        let mut advanced = false;
        let mut forced = false;

        if self.frames_on_current_unit >= self.max_frames_per_unit {
            advanced = true;
            forced = true;
            self.forced_advance_count += 1;
        } else if advance_prob > 0.5 && self.progress >= 1.0 {
            if boundary_confidence >= self.skip_protection_threshold {
                advanced = true;
            } else {
                self.skip_protection_count += 1;
            }
        } else if hold_bias > 2.0 {
            if advance_prob > 0.5 && self.progress >= 1.0 {
                advanced = true;
            }
        } else if advance_prob > 0.5 || self.progress >= 1.0 {
            advanced = true;
        }

        if advanced {
            self.text_index += 1;
            // SOTA: Carry over surplus progress
            self.progress = if forced { 0.0 } else { (self.progress - 1.0).max(0.0) };
            self.frames_on_current_unit = 0;
            self.stall_frames = 0;
        } else {
            self.stall_frames += 1;
        }

        advanced
    }
}

struct TokenBuffer {
    tokens: Vec<i64>,
    write_pos: usize,
    count: usize,
}

impl TokenBuffer {
    fn new() -> Self {
        Self {
            tokens: vec![0; CONTEXT_LENGTH * N_CODEBOOKS],
            write_pos: 0,
            count: 0,
        }
    }

    fn push(&mut self, tokens: &[i64; N_CODEBOOKS]) {
        let start = self.write_pos * N_CODEBOOKS;
        self.tokens[start..start + N_CODEBOOKS].copy_from_slice(tokens);
        self.write_pos = (self.write_pos + 1) % CONTEXT_LENGTH;
        if self.count < CONTEXT_LENGTH {
            self.count += 1;
        }
    }

    fn is_full(&self) -> bool {
        self.count >= CONTEXT_LENGTH
    }

    fn get_context(&self) -> Vec<i64> {
        let mut ctx = vec![0i64; CONTEXT_LENGTH * N_CODEBOOKS];
        for cb in 0..N_CODEBOOKS {
            for frame in 0..CONTEXT_LENGTH {
                let read_pos = (self.write_pos + frame) % CONTEXT_LENGTH;
                let src_idx = read_pos * N_CODEBOOKS + cb;
                let dst_idx = cb * CONTEXT_LENGTH + frame;
                ctx[dst_idx] = self.tokens[src_idx];
            }
        }
        ctx
    }

    fn reset(&mut self) {
        self.tokens.fill(0);
        self.write_pos = 0;
        self.count = 0;
    }
}

struct ModelStates {
    codec_encoder: PingPongState,
    token_model: PingPongState,
    codec_decoder: PingPongState,
}

impl ModelStates {
    fn new() -> Self {
        Self {
            codec_encoder: PingPongState::new(CODEC_ENCODER_STATE_SIZE),
            token_model: PingPongState::new(KV_CACHE_SIZE),
            codec_decoder: PingPongState::new(CODEC_DECODER_STATE_SIZE),
        }
    }

    fn reset(&mut self) {
        self.codec_encoder.reset();
        self.token_model.reset();
        self.codec_decoder.reset();
    }
}

// ============================================================================
// StreamingEngine
// ============================================================================

/// Main streaming engine for real-time voice conversion and TTS.
pub struct StreamingEngine {
    models: Option<OrtBundle>,
    states: Option<ModelStates>,
    token_buffer: TokenBuffer,
    f0_tracker: F0Tracker,
    speaker: Option<SpeakerFile>,
    style: Option<StyleFile>,
    character: Option<CharacterFile>,
    running: AtomicBool,
    timing: TimingStats,
    temperature: f32,
    top_k: usize,
    status: Option<Arc<SharedStatus>>,
    models_loaded: bool,
    ptr: PointerState,
}

impl StreamingEngine {
    /// Create a new streaming engine.
    pub fn new(status: Option<Arc<SharedStatus>>) -> Self {
        Self {
            models: None,
            states: None,
            token_buffer: TokenBuffer::new(),
            f0_tracker: F0Tracker::new(SAMPLE_RATE as u32, 220.0),
            speaker: None,
            style: None,
            character: None,
            running: AtomicBool::new(false),
            timing: TimingStats::new(),
            temperature: 1.0,
            top_k: 50,
            status,
            models_loaded: false,
            ptr: PointerState::default(),
        }
    }

    /// Load ONNX models from directory.
    pub fn load_models(&mut self, dir: &Path) -> Result<()> {
        let models = OrtBundle::new_codec_latent(dir)?;
        let states = ModelStates::new();
        self.models = Some(models);
        self.states = Some(states);
        self.models_loaded = true;
        log::info!("Models loaded from {:?}", dir);
        Ok(())
    }

    /// Load speaker file (.tmrvc_speaker).
    pub fn load_speaker(&mut self, path: &Path) -> Result<()> {
        let speaker = SpeakerFile::load(path)?;
        self.f0_tracker.set_f0_mean(speaker.f0_mean);
        self.speaker = Some(speaker);
        log::info!("Speaker loaded from {:?}", path);
        Ok(())
    }

    /// Load style file (.tmrvc_style).
    pub fn load_style(&mut self, path: &Path) -> Result<()> {
        self.style = Some(StyleFile::load(path)?);
        if let Some(ref status) = self.status {
            status.style_loaded.store(true, Ordering::SeqCst);
        }
        log::info!("Style loaded from {:?}", path);
        Ok(())
    }

    /// Load character file (.tmrvc_character).
    pub fn load_character(&mut self, path: &Path) -> Result<()> {
        self.character = Some(CharacterFile::load(path)?);
        log::info!("Character loaded from {:?}", path);
        Ok(())
    }

    /// Clear loaded style.
    pub fn clear_style(&mut self) {
        self.style = None;
        if let Some(ref status) = self.status {
            status.style_loaded.store(false, Ordering::SeqCst);
        }
    }

    /// Check if style is loaded.
    pub fn has_style(&self) -> bool {
        self.style.is_some()
    }

    /// Check if engine is ready for processing.
    pub fn is_ready(&self) -> bool {
        self.models_loaded
    }

    /// Set sampling parameters.
    pub fn set_sampling_params(&mut self, temperature: f32, top_k: usize) {
        self.temperature = temperature;
        self.top_k = top_k;
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        if let Some(ref mut states) = self.states {
            states.reset();
        }
        self.token_buffer.reset();
        self.f0_tracker.reset();
        self.timing.reset();
        if let Some(ref status) = self.status {
            status.reset();
        }
    }

    /// Start processing.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        if let Some(ref status) = self.status {
            status.is_running.store(true, Ordering::SeqCst);
        }
    }

    /// Stop processing.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(ref status) = self.status {
            status.is_running.store(false, Ordering::SeqCst);
        }
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get timing statistics.
    pub fn timing(&self) -> &TimingStats {
        &self.timing
    }

    /// Process one frame of audio.
    ///
    /// Args:
    /// - `input`: Input audio samples (FRAME_SIZE samples)
    /// - `output`: Output audio samples (FRAME_SIZE samples)
    /// - `params`: Per-frame parameters
    pub fn process_one_frame(&mut self, input: &[f32], output: &mut [f32], params: &FrameParams) {
        debug_assert_eq!(input.len(), FRAME_SIZE);
        debug_assert_eq!(output.len(), FRAME_SIZE);

        // Check if ready
        if !self.is_ready() {
            output.copy_from_slice(input);
            return;
        }

        let start_time = std::time::Instant::now();

        // Update F0 tracker with pitch shift parameter
        self.f0_tracker.set_pitch_shift(params.pitch_shift);

        // Update input level
        let input_rms: f32 = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32).sqrt();
        let input_db = if input_rms > 1e-10 {
            20.0 * input_rms.log10()
        } else {
            -100.0
        };

        // Run the Codec-Latent pipeline
        let result = self.process_frame_internal(input, &params);

        // Mix output
        match result {
            Ok(processed) => {
                // Dry/wet mix
                for (i, out) in output.iter_mut().enumerate() {
                    let dry = input[i];
                    let wet = processed[i];
                    *out =
                        (dry * (1.0 - params.dry_wet) + wet * params.dry_wet) * params.output_gain;
                }
            }
            Err(e) => {
                log::error!("Frame processing error: {}", e);
                output.copy_from_slice(input);
            }
        }

        // Update output level
        let output_rms: f32 =
            (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
        let output_db = if output_rms > 1e-10 {
            20.0 * output_rms.log10()
        } else {
            -100.0
        };

        // Update timing stats
        let elapsed_us = start_time.elapsed().as_micros() as f32;
        let elapsed_ms = elapsed_us / 1000.0;
        let frame_count = self.timing.frame_count.fetch_add(1, Ordering::SeqCst);

        if frame_count > 0 {
            let prev_avg = self.timing.avg_frame_us.load(Ordering::SeqCst);
            let new_avg = prev_avg + (elapsed_us - prev_avg) / (frame_count + 1) as f32;
            self.timing.avg_frame_us.store(new_avg, Ordering::SeqCst);
        }

        let prev_max = self.timing.max_frame_us.load(Ordering::SeqCst);
        if elapsed_us > prev_max {
            self.timing.max_frame_us.store(elapsed_us, Ordering::SeqCst);
        }

        if elapsed_us > 20000.0 {
            self.timing.overrun_count.fetch_add(1, Ordering::SeqCst);
        }

        // Update shared status
        if let Some(ref status) = self.status {
            status.input_level_db.store(input_db, Ordering::Relaxed);
            status.output_level_db.store(output_db, Ordering::Relaxed);
            status.inference_ms.store(elapsed_ms, Ordering::Relaxed);
            status
                .latency_quality_q
                .store(params.latency_quality_q, Ordering::Relaxed);
            status
                .alpha_timbre
                .store(params.alpha_timbre, Ordering::Relaxed);
            status
                .beta_prosody
                .store(params.beta_prosody, Ordering::Relaxed);
            status
                .gamma_articulation
                .store(params.gamma_articulation, Ordering::Relaxed);
        }
    }

    fn process_frame_internal(&mut self, audio_in: &[f32], params: &FrameParams) -> Result<Vec<f32>> {
        let temperature = self.temperature;
        let top_k = self.top_k;

        let models = self
            .models
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Models not loaded"))?;
        let states = self
            .states
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("States not initialized"))?;

        // 1. Determine Input Mode (VC vs TTS)
        // If audio_in is silent and we have characters/text, we could enter TTS mode.
        // For now, we follow params to decide.
        let is_tts = audio_in.iter().all(|&x| x.abs() < 1e-6);

        let tokens_out = if is_tts {
            // --- TTS Mode: Pointer-driven generation ---
            let (kv_in, kv_out) = states.token_model.get_both();
            
            // Get current phoneme feature from character memory
            // (Simplification for v0: assume character memory is pre-loaded)
            let text_ctx = vec![0.0f32; D_MODEL]; // Placeholder for TextEncoder output
            
            let mut logits_out = vec![0.0f32; N_CODEBOOKS * CODEBOOK_SIZE];
            let mut hidden_states = vec![0.0f32; D_MODEL];

            // Run core transformer
            models.run_token_model(
                &[], // source tokens empty in TTS
                &params.speaker_embed,
                &params.f0_condition,
                &text_ctx, // Pass text features as memory
                kv_in,
                &mut logits_out,
                kv_out,
                Some(&mut hidden_states),
            )?;

            // Run pointer head to get transition signals
            let mut p_signals = [0.0f32; 3]; // adv_logit, progress_delta, boundary_conf
            models.run_pointer_head(&hidden_states, &mut p_signals)?;

            // SOTA: Update pointer with continuous integration
            self.ptr.step(
                p_signals[0], // advance_logit (converted to prob in step if needed, or use sigmoid)
                p_signals[1], // progress_delta
                p_signals[2], // boundary_confidence
                params.hold_bias
            );

            sample_tokens(&logits_out, top_k, temperature)
        } else {
            // --- VC Mode: Source-driven generation ---
            let (state_in, state_out) = states.codec_encoder.get_both();
            let tokens_in = models.run_codec_encoder(audio_in, state_in, state_out)?;
            self.token_buffer.push(&tokens_in);

            if self.token_buffer.is_full() {
                let tokens_ctx = self.token_buffer.get_context();
                let mut logits_out = vec![0.0f32; N_CODEBOOKS * CODEBOOK_SIZE];
                let (kv_in, kv_out) = states.token_model.get_both();
                
                models.run_token_model(
                    &tokens_ctx,
                    &params.speaker_embed,
                    &params.f0_condition,
                    &vec![0.0f32; D_MODEL],
                    kv_in,
                    &mut logits_out,
                    kv_out,
                    None,
                )?;
                sample_tokens(&logits_out, top_k, temperature)
            } else {
                tokens_in
            }
        };

        // 3. Codec decoder: tokens → audio
        let (dec_in, dec_out) = states.codec_decoder.get_both();
        let audio = models.run_codec_decoder(
            &tokens_out,
            &params.control_tokens,
            &params.voice_state,
            &vec![0.0f32; D_EVENT_TRACE],
            dec_in,
            &mut vec![0.0f32; D_EVENT_TRACE],
            dec_out,
        )?;

        let mut audio_out = vec![0.0f32; FRAME_SIZE];
        audio_out.copy_from_slice(&audio[..FRAME_SIZE]);

        states.codec_encoder.swap();
        states.token_model.swap();
        states.codec_decoder.swap();

        Ok(audio_out)
    }
}

fn sample_tokens(logits: &[f32], top_k: usize, temperature: f32) -> [i64; N_CODEBOOKS] {
    let mut tokens = [0i64; N_CODEBOOKS];

    for cb in 0..N_CODEBOOKS {
        let cb_logits = &logits[cb * CODEBOOK_SIZE..(cb + 1) * CODEBOOK_SIZE];
        tokens[cb] = sample_top_k(cb_logits, top_k, temperature);
    }

    tokens
}

fn sample_top_k(logits: &[f32], k: usize, temperature: f32) -> i64 {
    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();

    let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    let max_logit = indexed[0].1;
    let exp_sum: f32 = indexed.iter().map(|(_, x)| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = indexed
        .iter()
        .map(|(_, x)| (x - max_logit).exp() / exp_sum)
        .collect();

    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, prob) in probs.iter().enumerate() {
        cumsum += prob;
        if r < cumsum {
            return indexed[i].0 as i64;
        }
    }

    indexed[0].0 as i64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_buffer() {
        let mut buf = TokenBuffer::new();

        assert!(!buf.is_full());

        for i in 0..CONTEXT_LENGTH {
            buf.push(&[i as i64; N_CODEBOOKS]);
        }

        assert!(buf.is_full());
    }

    #[test]
    fn test_shared_status() {
        let status = SharedStatus::new();
        assert_eq!(status.input_level_db.load(Ordering::Relaxed), -100.0);
        status.input_level_db.store(-20.0, Ordering::Relaxed);
        assert_eq!(status.input_level_db.load(Ordering::Relaxed), -20.0);
    }

    #[test]
    fn test_frame_params() {
        let params = FrameParams {
            dry_wet: 0.8,
            output_gain: 1.5,
            alpha_timbre: 1.0,
            beta_prosody: 0.0,
            gamma_articulation: 0.0,
            voice_source_alpha: 0.0,
            latency_quality_q: 0.0,
            pitch_shift: 0.0,
            cfg_scale: 1.0,
            temperature_a: 1.0,
            temperature_b: 1.0,
            top_k_a: 50,
            top_k_b: 20,
            voice_state: [0.5f32; D_VOICE_STATE],
        };
        assert_eq!(params.dry_wet, 0.8);
    }

    #[test]
    fn test_streaming_engine_new() {
        let engine = StreamingEngine::new(None);
        assert!(!engine.is_ready());
        assert!(!engine.is_running());
    }
}
