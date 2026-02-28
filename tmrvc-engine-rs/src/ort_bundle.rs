//! ONNX Runtime session bundle for Codec-Latent pipeline.
//!
//! Loads 3 streaming models:
//! - codec_encoder.onnx
//! - token_model.onnx
//! - codec_decoder.onnx
//!
//! Plus optional TTS models.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::constants::*;

fn build_session(model_path: impl AsRef<Path>) -> Result<Session> {
    Session::builder()?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path.as_ref())
        .map_err(Into::into)
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

/// Bundle of ONNX Runtime sessions for Codec-Latent streaming inference.
pub struct OrtBundle {
    pub codec_encoder: Session,
    pub token_model: Session,
    pub codec_decoder: Session,
    
    text_encoder: Option<Session>,
    duration_predictor: Option<Session>,
    f0_predictor: Option<Session>,
}

impl OrtBundle {
    pub fn new_codec_latent(model_dir: &Path) -> Result<Self> {
        let codec_encoder = build_session(model_dir.join("codec_encoder.onnx"))
            .context("Failed to load codec_encoder.onnx")?;
        let token_model = build_session(model_dir.join("token_model.onnx"))
            .context("Failed to load token_model.onnx")?;
        let codec_decoder = build_session(model_dir.join("codec_decoder.onnx"))
            .context("Failed to load codec_decoder.onnx")?;
        
        log::info!("Loaded Codec-Latent models from {:?}", model_dir);
        
        let text_encoder = load_optional(model_dir, "text_encoder.onnx");
        let duration_predictor = load_optional(model_dir, "duration_predictor.onnx");
        let f0_predictor = load_optional(model_dir, "f0_predictor.onnx");
        
        Ok(Self {
            codec_encoder,
            token_model,
            codec_decoder,
            text_encoder,
            duration_predictor,
            f0_predictor,
        })
    }
    
    pub fn run_codec_encoder(
        &mut self,
        audio: &[f32],
        state_in: &[f32],
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let outputs = self.codec_encoder.run(ort::inputs![
            "audio_frame" => TensorRef::from_array_view(([1usize, 1, FRAME_SIZE], audio))?,
            "state_in" => TensorRef::from_array_view(([1usize, LATENT_DIM, 32], state_in))?,
        ])?;
        
        let tokens = outputs["tokens"].try_extract_tensor::<i64>()?;
        let tokens_vec: Vec<i64> = tokens.1.to_vec();
        
        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        let state_vec: Vec<f32> = state.1.to_vec();
        
        Ok((tokens_vec, state_vec))
    }
    
    pub fn run_token_model(
        &mut self,
        tokens: &[i64],
        spk_embed: &[f32],
        f0_condition: &[f32],
        kv_cache_in: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let outputs = self.token_model.run(ort::inputs![
            "tokens_in" => TensorRef::from_array_view(([1usize, N_CODEBOOKS, CONTEXT_LENGTH], tokens))?,
            "spk_embed" => TensorRef::from_array_view(([1usize, D_SPEAKER], spk_embed))?,
            "f0_condition" => TensorRef::from_array_view(([1usize, CONTEXT_LENGTH, 2], f0_condition))?,
            "kv_cache_in" => TensorRef::from_array_view(([N_LAYERS * 2, 1, N_HEADS, CONTEXT_LENGTH, HEAD_DIM], kv_cache_in))?,
        ])?;
        
        let logits = outputs["logits"].try_extract_tensor::<f32>()?;
        let logits_vec: Vec<f32> = logits.1.to_vec();
        
        let kv_cache = outputs["kv_cache_out"].try_extract_tensor::<f32>()?;
        let kv_cache_vec: Vec<f32> = kv_cache.1.to_vec();
        
        Ok((logits_vec, kv_cache_vec))
    }
    
    pub fn run_codec_decoder(
        &mut self,
        tokens: &[i64],
        state_in: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let outputs = self.codec_decoder.run(ort::inputs![
            "tokens" => TensorRef::from_array_view(([1usize, N_CODEBOOKS], tokens))?,
            "state_in" => TensorRef::from_array_view(([1usize, 256, 32], state_in))?,
        ])?;
        
        let audio = outputs["audio_frame"].try_extract_tensor::<f32>()?;
        let audio_vec: Vec<f32> = audio.1.to_vec();
        
        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        let state_vec: Vec<f32> = state.1.to_vec();
        
        Ok((audio_vec, state_vec))
    }
    
    pub fn run_text_encoder(
        &mut self,
        phonemes: &[i64],
        lang_id: i64,
    ) -> Result<Vec<f32>> {
        let session = self.text_encoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("text_encoder not loaded"))?;
        
        let t = phonemes.len();
        let lang_id_arr = [lang_id];
        let outputs = session.run(ort::inputs![
            "phonemes" => TensorRef::from_array_view(([1usize, t], phonemes))?,
            "lang_id" => TensorRef::from_array_view(([1usize], lang_id_arr.as_slice()))?,
        ])?;
        
        let features = outputs["text_features"].try_extract_tensor::<f32>()?;
        Ok(features.1.to_vec())
    }
    
    pub fn has_tts(&self) -> bool {
        self.text_encoder.is_some() 
            && self.duration_predictor.is_some() 
            && self.f0_predictor.is_some()
    }
    
    pub fn run_duration_predictor(
        &mut self,
        text_features: &[f32],
        seq_len: usize,
        style: &[f32],
    ) -> Result<Vec<f32>> {
        let session = self.duration_predictor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("duration_predictor not loaded"))?;
        
        let outputs = session.run(ort::inputs![
            "text_features" => TensorRef::from_array_view(([1usize, D_TEXT_ENCODER, seq_len], text_features))?,
            "style" => TensorRef::from_array_view(([1usize, N_STYLE_PARAMS], style))?,
        ])?;
        
        let durations = outputs["durations"].try_extract_tensor::<f32>()?;
        Ok(durations.1.to_vec())
    }
    
    pub fn run_f0_predictor(
        &mut self,
        expanded_features: &[f32],
        n_frames: usize,
        style: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let session = self.f0_predictor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("f0_predictor not loaded"))?;
        
        let outputs = session.run(ort::inputs![
            "expanded_features" => TensorRef::from_array_view(([1usize, D_TEXT_ENCODER, n_frames], expanded_features))?,
            "style" => TensorRef::from_array_view(([1usize, N_STYLE_PARAMS], style))?,
        ])?;
        
        let f0 = outputs["f0"].try_extract_tensor::<f32>()?;
        let voiced = outputs["voiced"].try_extract_tensor::<f32>()?;
        
        Ok((f0.1.to_vec(), voiced.1.to_vec()))
    }
    
    pub fn run_content_synthesizer(
        &mut self,
        expanded_features: &[f32],
        n_frames: usize,
    ) -> Result<Vec<f32>> {
        let session = self.text_encoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("text_encoder not loaded (content_synthesizer)"))?;
        
        let outputs = session.run(ort::inputs![
            "expanded_features" => TensorRef::from_array_view(([1usize, D_TEXT_ENCODER, n_frames], expanded_features))?,
        ])?;
        
        let content = outputs["content"].try_extract_tensor::<f32>()?;
        Ok(content.1.to_vec())
    }
}
