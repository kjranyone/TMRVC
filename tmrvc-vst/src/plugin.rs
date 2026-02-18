use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nih_plug::prelude::*;

use tmrvc_engine_rs::constants::{HOP_LENGTH, SAMPLE_RATE};
use tmrvc_engine_rs::nam::NamChain;
use tmrvc_engine_rs::processor::{FrameParams, StreamingEngine};
use tmrvc_engine_rs::resampler::PolyphaseResampler;

use crate::params::TmrvcParams;

/// Default data directory: %LOCALAPPDATA%/TMRVC/ (Windows)
/// or ~/.local/share/TMRVC/ (Linux) or ~/Library/Application Support/TMRVC/ (macOS).
fn default_data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("TMRVC"))
}

/// Search for the models directory.
/// Priority: 1. Persisted path 2. Default location.
fn resolve_models_dir(persisted: &str) -> Option<PathBuf> {
    if !persisted.is_empty() {
        let p = PathBuf::from(persisted);
        if p.is_dir() {
            return Some(p);
        }
        nih_warn!("Persisted models_dir not found: {:?}", p);
    }
    // Fallback: default data dir
    if let Some(dir) = default_data_dir() {
        let models = dir.join("models");
        if models.is_dir() {
            return Some(models);
        }
    }
    None
}

/// Search for the speaker file.
/// Priority: 1. Persisted path 2. Default location.
fn resolve_speaker_path(persisted: &str) -> Option<PathBuf> {
    if !persisted.is_empty() {
        let p = PathBuf::from(persisted);
        if p.is_file() {
            return Some(p);
        }
        nih_warn!("Persisted speaker_path not found: {:?}", p);
    }
    // Fallback: scan default data dir for first .tmrvc_speaker file
    if let Some(dir) = default_data_dir() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("tmrvc_speaker") {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// TMRVC Voice Conversion VST3 Plugin.
pub struct TmrvcPlugin {
    params: Arc<TmrvcParams>,
    engine: StreamingEngine,

    /// DAW sample rate (set in initialize)
    daw_rate: f32,

    /// Downsampler: DAW rate -> 24kHz
    resampler_in: Option<PolyphaseResampler>,
    /// Upsampler: 24kHz -> DAW rate
    resampler_out: Option<PolyphaseResampler>,

    /// Accumulation buffer for 24kHz samples (input side)
    accum_in: Vec<f32>,
    /// Number of valid samples in accum_in
    accum_in_len: usize,

    /// Buffer for upsampled engine output at DAW rate
    accum_out: Vec<f32>,
    /// Read position in accum_out
    accum_out_read: usize,
    /// Number of valid samples in accum_out
    accum_out_len: usize,

    /// Scratch buffer for resampler input processing
    resample_scratch: Vec<f32>,
    /// Scratch buffer for engine output (24kHz)
    engine_out_scratch: Vec<f32>,
    /// Scratch buffer for resampler output processing
    upsample_scratch: Vec<f32>,

    /// NAM (Neural Amp Modeler) processing chain
    nam_chain: NamChain,
}

impl Default for TmrvcPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(TmrvcParams::default()),
            engine: StreamingEngine::new(None),
            daw_rate: 48000.0,
            resampler_in: None,
            resampler_out: None,
            accum_in: Vec::new(),
            accum_in_len: 0,
            accum_out: Vec::new(),
            accum_out_read: 0,
            accum_out_len: 0,
            resample_scratch: Vec::new(),
            engine_out_scratch: Vec::new(),
            upsample_scratch: Vec::new(),
            nam_chain: NamChain::new(48000),
        }
    }
}

impl Plugin for TmrvcPlugin {
    const NAME: &'static str = "TMRVC";
    const VENDOR: &'static str = "kjranyone";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.daw_rate = buffer_config.sample_rate;
        let daw_rate_u32 = buffer_config.sample_rate.round() as u32;
        let engine_rate = SAMPLE_RATE as u32;

        // --- Resamplers ---
        if daw_rate_u32 != engine_rate {
            self.resampler_in = Some(PolyphaseResampler::new(daw_rate_u32, engine_rate));
            self.resampler_out = Some(PolyphaseResampler::new(engine_rate, daw_rate_u32));
        } else {
            self.resampler_in = None;
            self.resampler_out = None;
        }

        // --- Scratch buffer allocation ---
        let max_block = buffer_config.max_buffer_size as usize;
        let max_down = PolyphaseResampler::max_output_len(max_block, daw_rate_u32, engine_rate);
        self.accum_in = vec![0.0; max_down + HOP_LENGTH * 2];
        self.accum_in_len = 0;

        let max_hops = (max_down + HOP_LENGTH - 1) / HOP_LENGTH + 2;
        self.engine_out_scratch = vec![0.0; max_hops * HOP_LENGTH];

        let max_up =
            PolyphaseResampler::max_output_len(max_hops * HOP_LENGTH, engine_rate, daw_rate_u32);
        self.upsample_scratch = vec![0.0; max_up + max_block];
        self.accum_out = vec![0.0; max_up + max_block * 2];
        self.accum_out_read = 0;
        self.accum_out_len = 0;
        self.resample_scratch = vec![0.0; max_down + HOP_LENGTH * 2];

        // --- Latency reporting ---
        let latency_samples = (2.0 * HOP_LENGTH as f32 / SAMPLE_RATE as f32
            * buffer_config.sample_rate)
            .round() as u32;
        context.set_latency_samples(latency_samples);

        // --- NAM chain ---
        self.nam_chain = NamChain::new(daw_rate_u32);

        // --- Model loading ---
        // initialize() runs on a non-RT thread, so blocking I/O is safe here.
        // Paths come from: 1. #[persist] (DAW project restore) 2. default locations.
        self.load_models_and_speaker();

        // --- NAM profile loading ---
        let nam_path = self.params.nam_profile_path.lock().clone();
        if !nam_path.is_empty() {
            match self.nam_chain.load_profile(Path::new(&nam_path)) {
                Ok(()) => nih_log!("NAM profile loaded from {:?}", nam_path),
                Err(e) => nih_error!("Failed to load NAM profile: {}", e),
            }
        }

        true
    }

    fn reset(&mut self) {
        self.engine.reset();
        if let Some(ref mut r) = self.resampler_in {
            r.reset();
        }
        if let Some(ref mut r) = self.resampler_out {
            r.reset();
        }
        self.accum_in_len = 0;
        self.accum_out_read = 0;
        self.accum_out_len = 0;
        self.nam_chain.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        if num_samples == 0 {
            return ProcessStatus::Normal;
        }

        // Read parameters (smoothed values)
        let dry_wet = self.params.dry_wet.smoothed.next();
        let output_gain = self.params.output_gain.smoothed.next();
        let latency_quality_q = self.params.latency_quality_q.smoothed.next();
        let frame_params = FrameParams {
            dry_wet,
            output_gain,
            latency_quality_q,
        };

        // Get channel 0 (mono)
        let channel_data = buffer.as_slice();
        let samples = &mut channel_data[0][..num_samples];

        // Step 1: Downsample DAW input -> 24kHz
        let downsampled_len;
        if let Some(ref mut resampler) = self.resampler_in {
            downsampled_len = resampler.process(samples, &mut self.resample_scratch);
        } else {
            self.resample_scratch[..num_samples].copy_from_slice(samples);
            downsampled_len = num_samples;
        }

        // Step 2: Append downsampled samples to accumulation buffer
        let new_len = self.accum_in_len + downsampled_len;
        if new_len <= self.accum_in.len() {
            self.accum_in[self.accum_in_len..new_len]
                .copy_from_slice(&self.resample_scratch[..downsampled_len]);
            self.accum_in_len = new_len;
        }

        // Step 3: Process complete hops through the engine
        let mut engine_out_len = 0;
        while self.accum_in_len >= HOP_LENGTH {
            let input_hop: Vec<f32> = self.accum_in[..HOP_LENGTH].to_vec();
            let out_start = engine_out_len;
            let out_end = out_start + HOP_LENGTH;
            self.engine.process_one_frame(
                &input_hop,
                &mut self.engine_out_scratch[out_start..out_end],
                &frame_params,
            );
            engine_out_len += HOP_LENGTH;

            self.accum_in.copy_within(HOP_LENGTH..self.accum_in_len, 0);
            self.accum_in_len -= HOP_LENGTH;
        }

        // Step 4: Upsample engine output -> DAW rate
        if engine_out_len > 0 {
            let upsampled_len;
            if let Some(ref mut resampler) = self.resampler_out {
                upsampled_len = resampler.process(
                    &self.engine_out_scratch[..engine_out_len],
                    &mut self.upsample_scratch,
                );
            } else {
                self.upsample_scratch[..engine_out_len]
                    .copy_from_slice(&self.engine_out_scratch[..engine_out_len]);
                upsampled_len = engine_out_len;
            }

            // Compact output accumulation buffer
            let remaining = self.accum_out_len - self.accum_out_read;
            if remaining > 0 && self.accum_out_read > 0 {
                self.accum_out
                    .copy_within(self.accum_out_read..self.accum_out_len, 0);
            }
            self.accum_out_read = 0;
            self.accum_out_len = remaining;

            let new_out_end = self.accum_out_len + upsampled_len;
            if new_out_end <= self.accum_out.len() {
                self.accum_out[self.accum_out_len..new_out_end]
                    .copy_from_slice(&self.upsample_scratch[..upsampled_len]);
                self.accum_out_len = new_out_end;
            }
        }

        // Step 4.5: NAM processing (at DAW rate, on upsampled output)
        if self.params.nam_enabled.value() {
            self.nam_chain
                .set_mix(self.params.nam_mix.smoothed.next());
            self.nam_chain.set_enabled(true);
            let nam_start = self.accum_out_read;
            let nam_end = self.accum_out_len;
            if nam_end > nam_start {
                self.nam_chain.process(&mut self.accum_out[nam_start..nam_end]);
            }
        } else {
            self.nam_chain.set_enabled(false);
        }

        // Step 5: Write available output to DAW buffer
        let available = self.accum_out_len - self.accum_out_read;
        let to_write = num_samples.min(available);
        if to_write > 0 {
            samples[..to_write].copy_from_slice(
                &self.accum_out[self.accum_out_read..self.accum_out_read + to_write],
            );
            self.accum_out_read += to_write;
        }
        // Zero-fill if not enough output yet (startup transient)
        for s in samples[to_write..num_samples].iter_mut() {
            *s = 0.0;
        }

        ProcessStatus::Normal
    }
}

impl TmrvcPlugin {
    /// Attempt to load ONNX models and speaker from persisted paths or defaults.
    ///
    /// Called from `initialize()` (non-RT thread). Logs results.
    fn load_models_and_speaker(&mut self) {
        let persisted_models = self.params.models_dir.lock().clone();
        let persisted_speaker = self.params.speaker_path.lock().clone();

        // --- Models ---
        if let Some(dir) = resolve_models_dir(&persisted_models) {
            match self.engine.load_models(&dir) {
                Ok(()) => {
                    nih_log!("ONNX models loaded from {:?}", dir);
                    // Update persisted path with the resolved path
                    *self.params.models_dir.lock() = dir.display().to_string();
                }
                Err(e) => {
                    nih_error!("Failed to load ONNX models from {:?}: {}", dir, e);
                }
            }
        } else {
            nih_warn!(
                "No ONNX model directory found. Place models in {:?} or set path via GUI.",
                default_data_dir().map(|d| d.join("models"))
            );
        }

        // --- Speaker ---
        if let Some(path) = resolve_speaker_path(&persisted_speaker) {
            match self.engine.load_speaker(&path) {
                Ok(()) => {
                    nih_log!("Speaker loaded from {:?}", path);
                    *self.params.speaker_path.lock() = path.display().to_string();
                }
                Err(e) => {
                    nih_error!("Failed to load speaker from {:?}: {}", path, e);
                }
            }
        } else {
            nih_warn!(
                "No speaker file found. Place .tmrvc_speaker in {:?} or set path via GUI.",
                default_data_dir()
            );
        }

        if self.engine.is_ready() {
            nih_log!("TMRVC engine ready (models + speaker loaded)");
        } else {
            nih_log!("TMRVC engine in bypass mode (models/speaker not loaded)");
        }
    }
}

impl Vst3Plugin for TmrvcPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"TMRVCVoiceConvrt";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

nih_export_vst3!(TmrvcPlugin);
