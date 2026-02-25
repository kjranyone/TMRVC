use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::constants::*;

const LORA_INPUT_NAME: &str = "lora_delta";

/// Bundle of ONNX Runtime sessions for streaming inference.
///
/// speaker_encoder is excluded - it runs offline and produces
/// the .tmrvc_speaker file consumed by `speaker.rs`.
///
/// `converter_hq` is optional - loaded only if `converter_hq.onnx` exists.
///
/// TTS front-end models are optional - loaded only when the corresponding
/// ONNX files are present. They run offline (batch) rather than streaming.
pub struct OrtBundle {
    // --- VC streaming models (required) ---
    content_encoder: Session,
    ir_estimator: Session,
    converter: Session,
    converter_hq: Option<Session>,
    vocoder: Session,
    converter_accepts_lora: bool,
    converter_hq_accepts_lora: bool,
    // --- TTS front-end models (optional) ---
    text_encoder: Option<Session>,
    duration_predictor: Option<Session>,
    f0_predictor: Option<Session>,
    content_synthesizer: Option<Session>,
}

/// Helper to build a session with standard options.
fn build_session(model_path: impl AsRef<Path>) -> Result<Session> {
    Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path.as_ref())
        .map_err(Into::into)
}

fn session_has_input(session: &Session, input_name: &str) -> bool {
    session.inputs().iter().any(|inp| inp.name() == input_name)
}

fn load_optional(dir: &Path, filename: &str) -> Option<Session> {
    let path = dir.join(filename);
    if !path.exists() {
        return None;
    }
    match build_session(&path) {
        Ok(sess) => {
            log::info!("Loaded optional model: {}", filename);
            Some(sess)
        }
        Err(e) => {
            log::warn!("Failed to load optional model {}: {}", filename, e);
            None
        }
    }
}

impl OrtBundle {
    /// Load all streaming ONNX models from a directory.
    ///
    /// `converter_hq.onnx` is optional - if present, HQ mode is available.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let content_encoder = build_session(model_dir.join("content_encoder.onnx"))
            .context("Failed to load content_encoder.onnx")?;
        let ir_estimator = build_session(model_dir.join("ir_estimator.onnx"))
            .context("Failed to load ir_estimator.onnx")?;
        let converter = build_session(model_dir.join("converter.onnx"))
            .context("Failed to load converter.onnx")?;
        let vocoder =
            build_session(model_dir.join("vocoder.onnx")).context("Failed to load vocoder.onnx")?;

        let converter_accepts_lora = session_has_input(&converter, LORA_INPUT_NAME);
        if !converter_accepts_lora {
            bail!(
                "converter.onnx must expose '{}' input (no backward-compat fallback)",
                LORA_INPUT_NAME
            );
        }
        log::info!("converter.onnx supports '{}' input", LORA_INPUT_NAME);

        // Optional HQ converter
        let hq_path = model_dir.join("converter_hq.onnx");
        let (converter_hq, converter_hq_accepts_lora) = if hq_path.exists() {
            let sess = build_session(&hq_path).context("Failed to load converter_hq.onnx")?;
            if !session_has_input(&sess, LORA_INPUT_NAME) {
                bail!(
                    "converter_hq.onnx must expose '{}' input (no backward-compat fallback)",
                    LORA_INPUT_NAME
                );
            }
            log::info!("HQ converter loaded from {:?}", hq_path);
            log::info!("converter_hq.onnx supports '{}' input", LORA_INPUT_NAME);
            (Some(sess), true)
        } else {
            (None, false)
        };

        // Optional TTS front-end models
        let text_encoder = load_optional(model_dir, "text_encoder.onnx");
        let duration_predictor = load_optional(model_dir, "duration_predictor.onnx");
        let f0_predictor = load_optional(model_dir, "f0_predictor.onnx");
        let content_synthesizer = load_optional(model_dir, "content_synthesizer.onnx");

        let tts_count = [&text_encoder, &duration_predictor, &f0_predictor, &content_synthesizer]
            .iter()
            .filter(|s| s.is_some())
            .count();
        if tts_count > 0 {
            log::info!("TTS front-end: {}/4 models loaded", tts_count);
        }

        Ok(Self {
            content_encoder,
            ir_estimator,
            converter,
            converter_hq,
            vocoder,
            converter_accepts_lora,
            converter_hq_accepts_lora,
            text_encoder,
            duration_predictor,
            f0_predictor,
            content_synthesizer,
        })
    }

    /// Returns true if the HQ converter model is available.
    pub fn has_hq_converter(&self) -> bool {
        self.converter_hq.is_some()
    }

    /// Returns true if converter.onnx accepts a `lora_delta` input.
    pub fn converter_accepts_lora(&self) -> bool {
        self.converter_accepts_lora
    }

    /// Returns true if converter_hq.onnx accepts a `lora_delta` input.
    pub fn converter_hq_accepts_lora(&self) -> bool {
        self.converter_hq_accepts_lora
    }

    /// Run content_encoder: mel_frame[80] + f0[1] + state_in -> content[256] + state_out
    pub fn run_content_encoder(
        &mut self,
        mel: &[f32],
        f0: &[f32],
        state_in: &[f32],
        content_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let outputs = self.content_encoder.run(ort::inputs![
            "mel_frame" => TensorRef::from_array_view(([1, N_MELS, 1], mel))?,
            "f0" => TensorRef::from_array_view(([1usize, 1, 1], f0))?,
            "state_in" => TensorRef::from_array_view(([1, D_CONTENT, CONTENT_ENC_STATE_FRAMES], state_in))?,
        ])?;

        let content = outputs["content"].try_extract_tensor::<f32>()?;
        content_out.copy_from_slice(content.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    /// Run ir_estimator: mel_chunk[80x10] + state_in -> acoustic_params[32] + state_out
    pub fn run_ir_estimator(
        &mut self,
        mel_chunk: &[f32],
        state_in: &[f32],
        acoustic_params_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let outputs = self.ir_estimator.run(ort::inputs![
            "mel_chunk" => TensorRef::from_array_view(([1, N_MELS, IR_UPDATE_INTERVAL], mel_chunk))?,
            "state_in" => TensorRef::from_array_view(([1, D_IR_ESTIMATOR_HIDDEN, IR_EST_STATE_FRAMES], state_in))?,
        ])?;

        let ir = outputs["acoustic_params"].try_extract_tensor::<f32>()?;
        acoustic_params_out.copy_from_slice(ir.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    /// Run converter: content[256] + spk[192] + lora[15872] + acoustic[32] + state
    /// -> features[513] + state
    pub fn run_converter(
        &mut self,
        content: &[f32],
        spk_embed: &[f32],
        lora_delta: &[f32],
        acoustic_params: &[f32],
        state_in: &[f32],
        features_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let outputs = self.converter.run(ort::inputs![
            "content" => TensorRef::from_array_view(([1, D_CONTENT, 1], content))?,
            "spk_embed" => TensorRef::from_array_view(([1usize, D_SPEAKER], spk_embed))?,
            LORA_INPUT_NAME => TensorRef::from_array_view(([1usize, LORA_DELTA_SIZE], lora_delta))?,
            "acoustic_params" => TensorRef::from_array_view(([1usize, N_ACOUSTIC_PARAMS], acoustic_params))?,
            "state_in" => TensorRef::from_array_view(([1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES], state_in))?,
        ])?;

        let feats = outputs["pred_features"].try_extract_tensor::<f32>()?;
        features_out.copy_from_slice(feats.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    /// Run converter_hq: content[256x7] + spk[192] + lora[15872] + acoustic[32] + state
    /// -> features[513] + state
    pub fn run_converter_hq(
        &mut self,
        content: &[f32],
        spk_embed: &[f32],
        lora_delta: &[f32],
        acoustic_params: &[f32],
        state_in: &[f32],
        features_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let session = self
            .converter_hq
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("HQ converter not loaded"))?;

        let t_in = 1 + MAX_LOOKAHEAD_HOPS;
        let outputs = session.run(ort::inputs![
            "content" => TensorRef::from_array_view(([1, D_CONTENT, t_in], content))?,
            "spk_embed" => TensorRef::from_array_view(([1usize, D_SPEAKER], spk_embed))?,
            LORA_INPUT_NAME => TensorRef::from_array_view(([1usize, LORA_DELTA_SIZE], lora_delta))?,
            "acoustic_params" => TensorRef::from_array_view(([1usize, N_ACOUSTIC_PARAMS], acoustic_params))?,
            "state_in" => TensorRef::from_array_view(([1, D_CONVERTER_HIDDEN, CONVERTER_HQ_STATE_FRAMES], state_in))?,
        ])?;

        let feats = outputs["pred_features"].try_extract_tensor::<f32>()?;
        features_out.copy_from_slice(feats.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    // --- TTS front-end methods ---

    /// Returns true if all 4 TTS front-end models are loaded.
    pub fn has_tts(&self) -> bool {
        self.text_encoder.is_some()
            && self.duration_predictor.is_some()
            && self.f0_predictor.is_some()
            && self.content_synthesizer.is_some()
    }

    /// Run text_encoder: phoneme_ids[1,L] + language_ids[1] → text_features[1,256,L]
    ///
    /// Returns a newly allocated Vec<f32> of shape [256 * L].
    pub fn run_text_encoder(
        &mut self,
        phoneme_ids: &[i64],
        language_id: i64,
    ) -> Result<Vec<f32>> {
        let session = self
            .text_encoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("text_encoder not loaded"))?;

        let l = phoneme_ids.len();
        let lang_ids = [language_id];
        let outputs = session.run(ort::inputs![
            "phoneme_ids" => TensorRef::from_array_view(([1usize, l], phoneme_ids))?,
            "language_ids" => TensorRef::from_array_view(([1usize], &lang_ids[..]))?,
        ])?;

        let feats = outputs["text_features"].try_extract_tensor::<f32>()?;
        Ok(feats.1.to_vec())
    }

    /// Run duration_predictor: text_features[1,256,L] + style[1,32] → durations[1,L]
    ///
    /// Returns durations as Vec<f32>.
    pub fn run_duration_predictor(
        &mut self,
        text_features: &[f32],
        l: usize,
        style: &[f32],
    ) -> Result<Vec<f32>> {
        let session = self
            .duration_predictor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("duration_predictor not loaded"))?;

        let outputs = session.run(ort::inputs![
            "text_features" => TensorRef::from_array_view(([1, D_TEXT_ENCODER, l], text_features))?,
            "style" => TensorRef::from_array_view(([1usize, D_STYLE], style))?,
        ])?;

        let durs = outputs["durations"].try_extract_tensor::<f32>()?;
        Ok(durs.1.to_vec())
    }

    /// Run f0_predictor: text_features[1,256,T] + style[1,32] → (f0[1,1,T], voiced[1,1,T])
    ///
    /// Returns (f0_vec, voiced_vec).
    pub fn run_f0_predictor(
        &mut self,
        text_features: &[f32],
        t: usize,
        style: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let session = self
            .f0_predictor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("f0_predictor not loaded"))?;

        let outputs = session.run(ort::inputs![
            "text_features" => TensorRef::from_array_view(([1, D_TEXT_ENCODER, t], text_features))?,
            "style" => TensorRef::from_array_view(([1usize, D_STYLE], style))?,
        ])?;

        let f0 = outputs["f0"].try_extract_tensor::<f32>()?;
        let voiced = outputs["voiced"].try_extract_tensor::<f32>()?;
        Ok((f0.1.to_vec(), voiced.1.to_vec()))
    }

    /// Run content_synthesizer: text_features[1,256,T] → content[1,256,T]
    ///
    /// Returns content as Vec<f32>.
    pub fn run_content_synthesizer(
        &mut self,
        text_features: &[f32],
        t: usize,
    ) -> Result<Vec<f32>> {
        let session = self
            .content_synthesizer
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("content_synthesizer not loaded"))?;

        let outputs = session.run(ort::inputs![
            "text_features" => TensorRef::from_array_view(([1, D_TEXT_ENCODER, t], text_features))?,
        ])?;

        let content = outputs["content"].try_extract_tensor::<f32>()?;
        Ok(content.1.to_vec())
    }

    /// Run vocoder: features[513] + state -> mag[513] + phase[513] + state
    pub fn run_vocoder(
        &mut self,
        features: &[f32],
        state_in: &[f32],
        mag_out: &mut [f32],
        phase_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let outputs = self.vocoder.run(ort::inputs![
            "features" => TensorRef::from_array_view(([1, N_FREQ_BINS, 1], features))?,
            "state_in" => TensorRef::from_array_view(([1, D_VOCODER_HIDDEN, VOCODER_STATE_FRAMES], state_in))?,
        ])?;

        let mag = outputs["stft_mag"].try_extract_tensor::<f32>()?;
        mag_out.copy_from_slice(mag.1);

        let phase = outputs["stft_phase"].try_extract_tensor::<f32>()?;
        phase_out.copy_from_slice(phase.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }
}
