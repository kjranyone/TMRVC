use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nih_plug::prelude::*;

use tmrvc_engine_rs::constants::{D_ACTING_LATENT, D_VOICE_STATE};
use tmrvc_engine_rs::constants::{FRAME_SIZE, SAMPLE_RATE};
use tmrvc_engine_rs::nam::NamChain;
use tmrvc_engine_rs::processor::{FrameParams, StreamingEngine};
use tmrvc_engine_rs::resampler::PolyphaseResampler;

use crate::params::TmrvcParams;

fn default_data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("TMRVC"))
}

fn resolve_models_dir(persisted: &str) -> Option<PathBuf> {
    if !persisted.is_empty() {
        let p = PathBuf::from(persisted);
        if p.is_dir() {
            return Some(p);
        }
        nih_warn!("Persisted models_dir not found: {:?}", p);
    }
    if let Some(dir) = default_data_dir() {
        let models = dir.join("models").join("fp32");
        if models.is_dir() {
            return Some(models);
        }
    }
    None
}

fn resolve_speaker_path(persisted: &str) -> Option<PathBuf> {
    if !persisted.is_empty() {
        let p = PathBuf::from(persisted);
        if p.is_file() {
            return Some(p);
        }
        nih_warn!("Persisted speaker_path not found: {:?}", p);
    }
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

pub struct TmrvcPlugin {
    params: Arc<TmrvcParams>,
    processor: Option<StreamingEngine>,

    daw_rate: f32,

    resampler_in: Option<PolyphaseResampler>,
    resampler_out: Option<PolyphaseResampler>,

    accum_in: Vec<f32>,
    accum_in_len: usize,

    accum_out: Vec<f32>,
    accum_out_read: usize,
    accum_out_len: usize,

    resample_scratch: Vec<f32>,
    engine_out_scratch: Vec<f32>,
    upsample_scratch: Vec<f32>,

    reported_latency_samples: u32,

    nam_chain: NamChain,

    dry_wet: f32,
    output_gain: f32,
}

impl Default for TmrvcPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(TmrvcParams::default()),
            processor: None,
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
            reported_latency_samples: 0,
            nam_chain: NamChain::new(48000),
            dry_wet: 1.0,
            output_gain: 1.0,
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

        if daw_rate_u32 != engine_rate {
            self.resampler_in = Some(PolyphaseResampler::new(daw_rate_u32, engine_rate));
            self.resampler_out = Some(PolyphaseResampler::new(engine_rate, daw_rate_u32));
        } else {
            self.resampler_in = None;
            self.resampler_out = None;
        }

        let max_block = buffer_config.max_buffer_size as usize;
        let max_down = PolyphaseResampler::max_output_len(max_block, daw_rate_u32, engine_rate);
        self.accum_in = vec![0.0; max_down + FRAME_SIZE * 2];
        self.accum_in_len = 0;

        let max_frames = (max_down + FRAME_SIZE - 1) / FRAME_SIZE + 2;
        self.engine_out_scratch = vec![0.0; max_frames * FRAME_SIZE];

        let max_up =
            PolyphaseResampler::max_output_len(max_frames * FRAME_SIZE, engine_rate, daw_rate_u32);
        self.upsample_scratch = vec![0.0; max_up + max_block];
        self.accum_out = vec![0.0; max_up + max_block * 2];
        self.accum_out_read = 0;
        self.accum_out_len = 0;
        self.resample_scratch = vec![0.0; max_down + FRAME_SIZE * 2];

        let latency_samples = (2.0 * FRAME_SIZE as f32 / SAMPLE_RATE as f32
            * buffer_config.sample_rate)
            .round() as u32;
        context.set_latency_samples(latency_samples);
        self.reported_latency_samples = latency_samples;

        self.nam_chain = NamChain::new(daw_rate_u32);

        self.load_models_and_speaker();

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
        if let Some(ref mut proc) = self.processor {
            proc.reset();
        }
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

        self.dry_wet = self.params.dry_wet.smoothed.next();
        self.output_gain = self.params.output_gain.smoothed.next();

        let channel_data = buffer.as_slice();
        let samples = &mut channel_data[0][..num_samples];

        let downsampled_len;
        if let Some(ref mut resampler) = self.resampler_in {
            downsampled_len = resampler.process(samples, &mut self.resample_scratch);
        } else {
            self.resample_scratch[..num_samples].copy_from_slice(samples);
            downsampled_len = num_samples;
        }

        let new_len = self.accum_in_len + downsampled_len;
        if new_len <= self.accum_in.len() {
            self.accum_in[self.accum_in_len..new_len]
                .copy_from_slice(&self.resample_scratch[..downsampled_len]);
            self.accum_in_len = new_len;
        }

        let mut engine_out_len = 0;
        while self.accum_in_len >= FRAME_SIZE {
            if let Some(ref mut proc) = self.processor {
                let input_frame: Vec<f32> = self.accum_in[..FRAME_SIZE].to_vec();
                let out_start = engine_out_len;
                let out_end = out_start + FRAME_SIZE;

                let frame_params = FrameParams {
                    dry_wet: 1.0,
                    output_gain: 1.0,
                    alpha_timbre: 1.0,
                    beta_prosody: 0.0,
                    gamma_articulation: 0.0,
                    voice_source_alpha: 0.0,
                    latency_quality_q: self.params.latency_quality_q.smoothed.next(),
                    pitch_shift: self.params.pitch_shift.smoothed.next(),
                    cfg_scale: 1.0,
                    temperature_a: 1.0,
                    temperature_b: 1.0,
                    top_k_a: 50,
                    top_k_b: 20,
                    voice_state: [0.5f32; D_VOICE_STATE],
                    acting_texture_latent: [0.0f32; D_ACTING_LATENT],
                    pace: 1.0,
                    hold_bias: 0.0,
                    boundary_bias: 0.0,
                    phrase_pressure: 0.0,
                    breath_tendency: 0.0,
                };

                proc.process_one_frame(
                    &input_frame,
                    &mut self.engine_out_scratch[out_start..out_end],
                    &frame_params,
                );
            } else {
                let out_start = engine_out_len;
                let out_end = out_start + FRAME_SIZE;
                self.engine_out_scratch[out_start..out_end]
                    .copy_from_slice(&self.accum_in[..FRAME_SIZE]);
            }
            engine_out_len += FRAME_SIZE;

            self.accum_in.copy_within(FRAME_SIZE..self.accum_in_len, 0);
            self.accum_in_len -= FRAME_SIZE;
        }

        if engine_out_len > 0 {
            for i in 0..engine_out_len {
                self.engine_out_scratch[i] *= self.output_gain;
            }

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

        if self.params.nam_enabled.value() {
            self.nam_chain.set_mix(self.params.nam_mix.smoothed.next());
            self.nam_chain.set_enabled(true);
            let nam_start = self.accum_out_read;
            let nam_end = self.accum_out_len;
            if nam_end > nam_start {
                self.nam_chain
                    .process(&mut self.accum_out[nam_start..nam_end]);
            }
        } else {
            self.nam_chain.set_enabled(false);
        }

        let available = self.accum_out_len - self.accum_out_read;
        let to_write = num_samples.min(available);
        if to_write > 0 {
            samples[..to_write].copy_from_slice(
                &self.accum_out[self.accum_out_read..self.accum_out_read + to_write],
            );
            self.accum_out_read += to_write;
        }
        for s in samples[to_write..num_samples].iter_mut() {
            *s = 0.0;
        }

        ProcessStatus::Normal
    }
}

impl TmrvcPlugin {
    fn load_models_and_speaker(&mut self) {
        let persisted_models = self.params.models_dir.lock().clone();
        let persisted_speaker = self.params.speaker_path.lock().clone();

        if let Some(dir) = resolve_models_dir(&persisted_models) {
            let mut proc = StreamingEngine::new(None);
            match proc.load_models(&dir) {
                Ok(()) => {
                    nih_log!("ONNX models loaded from {:?}", dir);
                    *self.params.models_dir.lock() = dir.display().to_string();
                    self.processor = Some(proc);
                }
                Err(e) => {
                    nih_error!("Failed to load ONNX models from {:?}: {}", dir, e);
                }
            }
        } else {
            nih_warn!(
                "No ONNX model directory found. Place models in {:?} or set path via GUI.",
                default_data_dir().map(|d| d.join("models").join("fp32"))
            );
        }

        if let Some(path) = resolve_speaker_path(&persisted_speaker) {
            if let Some(ref mut proc) = self.processor {
                match proc.load_speaker(&path) {
                    Ok(()) => {
                        nih_log!("Speaker loaded from {:?}", path);
                        *self.params.speaker_path.lock() = path.display().to_string();
                    }
                    Err(e) => {
                        nih_error!("Failed to load speaker from {:?}: {}", path, e);
                    }
                }
            }
        } else {
            nih_warn!(
                "No speaker file found. Place .tmrvc_speaker in {:?} or set path via GUI.",
                default_data_dir()
            );
        }

        if let Some(ref mut proc) = self.processor {
            proc.start();
            nih_log!("TMRVC engine ready");
        }
    }
}

impl Vst3Plugin for TmrvcPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"TMRVCVoiceConvrt";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift];
}

nih_export_vst3!(TmrvcPlugin);
