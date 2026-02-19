use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::constants::*;

const LORA_INPUT_NAME: &str = "lora_delta";

/// Bundle of ONNX Runtime sessions for streaming inference.
///
/// speaker_encoder is excluded - it runs offline and produces
/// the .tmrvc_speaker file consumed by `speaker.rs`.
///
/// `converter_hq` is optional - loaded only if `converter_hq.onnx` exists.
pub struct OrtBundle {
    content_encoder: Session,
    ir_estimator: Session,
    converter: Session,
    converter_hq: Option<Session>,
    vocoder: Session,
    converter_accepts_lora: bool,
    converter_hq_accepts_lora: bool,
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

        Ok(Self {
            content_encoder,
            ir_estimator,
            converter,
            converter_hq,
            vocoder,
            converter_accepts_lora,
            converter_hq_accepts_lora,
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
            "mel_frame" => Tensor::from_array(([1, N_MELS, 1], mel.to_vec()))?,
            "f0" => Tensor::from_array(([1usize, 1, 1], f0.to_vec()))?,
            "state_in" => Tensor::from_array(([1, D_CONTENT, CONTENT_ENC_STATE_FRAMES], state_in.to_vec()))?,
        ])?;

        let content = outputs["content"].try_extract_tensor::<f32>()?;
        content_out.copy_from_slice(content.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    /// Run ir_estimator: mel_chunk[80x10] + state_in -> ir_params[24] + state_out
    pub fn run_ir_estimator(
        &mut self,
        mel_chunk: &[f32],
        state_in: &[f32],
        ir_params_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let outputs = self.ir_estimator.run(ort::inputs![
            "mel_chunk" => Tensor::from_array(([1, N_MELS, IR_UPDATE_INTERVAL], mel_chunk.to_vec()))?,
            "state_in" => Tensor::from_array(([1, 128usize, IR_EST_STATE_FRAMES], state_in.to_vec()))?,
        ])?;

        let ir = outputs["ir_params"].try_extract_tensor::<f32>()?;
        ir_params_out.copy_from_slice(ir.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    /// Run converter: content[256] + spk[192] + lora[24576] + ir[24] + state
    /// -> features[513] + state
    pub fn run_converter(
        &mut self,
        content: &[f32],
        spk_embed: &[f32],
        lora_delta: &[f32],
        ir_params: &[f32],
        state_in: &[f32],
        features_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<()> {
        let outputs = self.converter.run(ort::inputs![
            "content" => Tensor::from_array(([1, D_CONTENT, 1], content.to_vec()))?,
            "spk_embed" => Tensor::from_array(([1usize, D_SPEAKER], spk_embed.to_vec()))?,
            LORA_INPUT_NAME => Tensor::from_array(([1usize, LORA_DELTA_SIZE], lora_delta.to_vec()))?,
            "ir_params" => Tensor::from_array(([1usize, N_IR_PARAMS], ir_params.to_vec()))?,
            "state_in" => Tensor::from_array(([1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES], state_in.to_vec()))?,
        ])?;

        let feats = outputs["pred_features"].try_extract_tensor::<f32>()?;
        features_out.copy_from_slice(feats.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
    }

    /// Run converter_hq: content[256x7] + spk[192] + lora[24576] + ir[24] + state
    /// -> features[513] + state
    pub fn run_converter_hq(
        &mut self,
        content: &[f32],
        spk_embed: &[f32],
        lora_delta: &[f32],
        ir_params: &[f32],
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
            "content" => Tensor::from_array(([1, D_CONTENT, t_in], content.to_vec()))?,
            "spk_embed" => Tensor::from_array(([1usize, D_SPEAKER], spk_embed.to_vec()))?,
            LORA_INPUT_NAME => Tensor::from_array(([1usize, LORA_DELTA_SIZE], lora_delta.to_vec()))?,
            "ir_params" => Tensor::from_array(([1usize, N_IR_PARAMS], ir_params.to_vec()))?,
            "state_in" => Tensor::from_array(([1, D_CONVERTER_HIDDEN, CONVERTER_HQ_STATE_FRAMES], state_in.to_vec()))?,
        ])?;

        let feats = outputs["pred_features"].try_extract_tensor::<f32>()?;
        features_out.copy_from_slice(feats.1);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        state_out.copy_from_slice(state.1);

        Ok(())
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
            "features" => Tensor::from_array(([1, N_FREQ_BINS, 1], features.to_vec()))?,
            "state_in" => Tensor::from_array(([1, D_CONTENT, VOCODER_STATE_FRAMES], state_in.to_vec()))?,
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
