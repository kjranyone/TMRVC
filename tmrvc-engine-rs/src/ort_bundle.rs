//! ONNX Runtime session bundle for Disentangled UCLM pipeline.
//!
//! ONNX models:
//! - codec_encoder.onnx — audio → acoustic tokens A_t
//! - vc_encoder.onnx — source A_t → VQ content features
//! - voice_state_enc.onnx — explicit + SSL → state_cond
//! - uclm_core.onnx — dual-stream token predictor (A_t, B_t)
//! - codec_decoder.onnx — tokens → audio
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

/// Bundle of ONNX Runtime sessions for Disentangled UCLM streaming inference.
pub struct OrtBundle {
    pub codec_encoder: Session,
    pub vc_encoder: Session,
    pub voice_state_enc: Session,
    pub uclm_core: Session,
    pub codec_decoder: Session,

    text_encoder: Option<Session>,
    duration_predictor: Option<Session>,
    f0_predictor: Option<Session>,
    pointer_head: Option<Session>,
}

impl OrtBundle {
    pub fn new_uclm(model_dir: &Path) -> Result<Self> {
        let codec_encoder = build_session(model_dir.join("codec_encoder.onnx"))
            .context("Failed to load codec_encoder.onnx")?;
        let vc_encoder = build_session(model_dir.join("vc_encoder.onnx"))
            .context("Failed to load vc_encoder.onnx")?;
        let voice_state_enc = build_session(model_dir.join("voice_state_enc.onnx"))
            .context("Failed to load voice_state_enc.onnx")?;
        let uclm_core = build_session(model_dir.join("uclm_core.onnx"))
            .context("Failed to load uclm_core.onnx")?;
        let codec_decoder = build_session(model_dir.join("codec_decoder.onnx"))
            .context("Failed to load codec_decoder.onnx")?;

        log::info!("Loaded UCLM models from {:?}", model_dir);

        let text_encoder = load_optional(model_dir, "text_encoder.onnx");
        let duration_predictor = load_optional(model_dir, "duration_predictor.onnx");
        let f0_predictor = load_optional(model_dir, "f0_predictor.onnx");
        let pointer_head = load_optional(model_dir, "pointer_head.onnx");

        Ok(Self {
            codec_encoder,
            vc_encoder,
            voice_state_enc,
            uclm_core,
            codec_decoder,
            text_encoder,
            duration_predictor,
            f0_predictor,
            pointer_head,
        })
    }

    /// Legacy alias for compatibility
    pub fn new_codec_latent(model_dir: &Path) -> Result<Self> {
        Self::new_uclm(model_dir)
    }

    /// Run codec encoder: audio_frame → acoustic_tokens (A_src_t)
    ///
    /// Input: `[1, 1, 240]` audio frame
    /// Output: `[1, 8]` acoustic tokens
    /// RT-safe: writes state to pre-allocated buffer instead of allocating Vec
    pub fn run_codec_encoder(
        &mut self,
        audio: &[f32],
        state_in: &[f32],
        state_out: &mut [f32],
    ) -> Result<[i64; N_CODEBOOKS]> {
        let outputs = self.codec_encoder.run(ort::inputs![
            "audio_frame" => TensorRef::from_array_view(([1usize, 1, FRAME_SIZE], audio))?,
            "state_in" => TensorRef::from_array_view(([1usize, ENC_STATE_DIM, ENC_STATE_FRAMES], state_in))?,
        ])?;

        let tokens = outputs["acoustic_tokens"].try_extract_tensor::<i64>()?;
        let mut tokens_arr = [0i64; N_CODEBOOKS];
        tokens_arr.copy_from_slice(&tokens.1[..N_CODEBOOKS]);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        let state_len = state.1.len().min(state_out.len());
        state_out[..state_len].copy_from_slice(&state.1[..state_len]);

        Ok(tokens_arr)
    }

    /// Legacy codec encoder for old processor.rs compatibility.
    /// Returns tokens vec and state vec (allocates).
    #[deprecated(note = "Use run_codec_encoder with pre-allocated buffers for RT-safe")]
    pub fn run_codec_encoder_legacy(
        &mut self,
        audio: &[f32],
        state_in: &[f32],
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut state_out = vec![0.0f32; ENC_STATE_DIM * ENC_STATE_FRAMES];
        let tokens = self.run_codec_encoder(audio, state_in, &mut state_out)?;
        Ok((tokens.to_vec(), state_out))
    }

    /// Run VC encoder (VQ bottleneck): source_A_t → vq_content_features
    ///
    /// Input: `[1, 8, L]` source acoustic tokens (context)
    /// Output: `[1, d_model, L]` VQ bottlenecked content features
    /// RT-safe: writes features to pre-allocated buffer
    /// Returns: number of elements written
    pub fn run_vc_encoder(
        &mut self,
        source_tokens: &[i64],
        context_len: usize,
        features_out: &mut [f32],
    ) -> Result<usize> {
        let outputs = self.vc_encoder.run(ort::inputs![
            "source_A_t" => TensorRef::from_array_view(([1usize, N_CODEBOOKS, context_len], source_tokens))?,
        ])?;

        let features = outputs["vq_content_features"].try_extract_tensor::<f32>()?;
        let feat_len = features.1.len().min(features_out.len());
        features_out[..feat_len].copy_from_slice(&features.1[..feat_len]);

        Ok(feat_len)
    }

    /// Run voice state encoder: explicit + SSL + delta → state_cond
    ///
    /// Input: `[1, 8]` explicit, `[1, 128]` ssl_state, `[1, 8]` delta_state
    /// Output: `[1, d_model]` fused state condition
    /// RT-safe: writes to pre-allocated buffer
    pub fn run_voice_state_enc(
        &mut self,
        explicit_state: &[f32; D_VOICE_STATE_EXPLICIT],
        ssl_state: &[f32],
        delta_state: &[f32; D_VOICE_STATE_EXPLICIT],
        state_cond_out: &mut [f32],
    ) -> Result<()> {
        if ssl_state.len() < D_VOICE_STATE_SSL {
            for (i, &v) in ssl_state.iter().enumerate() {
                state_cond_out[i] = v;
            }
            return Ok(());
        }

        let outputs = self.voice_state_enc.run(ort::inputs![
            "explicit_state" => TensorRef::from_array_view(([1usize, D_VOICE_STATE_EXPLICIT], explicit_state.as_slice()))?,
            "ssl_state" => TensorRef::from_array_view(([1usize, D_VOICE_STATE_SSL], &ssl_state[..D_VOICE_STATE_SSL]))?,
            "delta_state" => TensorRef::from_array_view(([1usize, D_VOICE_STATE_EXPLICIT], delta_state.as_slice()))?,
        ])?;

        let state_cond = outputs["state_cond"].try_extract_tensor::<f32>()?;
        let cond_len = state_cond.1.len().min(state_cond_out.len());
        state_cond_out[..cond_len].copy_from_slice(&state_cond.1[..cond_len]);

        Ok(())
    }

    /// Run UCLM core: dual-stream token predictor
    ///
    /// Inputs:
    /// - content_features: `[1, d_model, L]`
    /// - b_ctx: `[1, 4, L]` control context
    /// - spk_embed: `[1, 192]`
    /// - state_cond: `[1, d_model]`
    /// - cfg_scale: `[1]`
    /// - kv_cache_in: flattened KV cache
    ///
    /// Outputs (RT-safe, written to pre-allocated buffers):
    /// - logits_a: `[1, 8, 1024]` next A_t distribution
    /// - logits_b: `[1, 4, 64]` next B_t distribution
    /// - kv_cache_out: updated KV cache
    pub fn run_uclm_core(
        &mut self,
        content_features: &[f32],
        b_ctx: &[i64],
        spk_embed: &[f32],
        state_cond: &[f32],
        cfg_scale: f32,
        kv_cache_in: &[f32],
        context_len: usize,
        logits_a_out: &mut [f32],
        logits_b_out: &mut [f32],
        kv_cache_out: &mut [f32],
    ) -> Result<()> {
        let cfg_arr = [cfg_scale];

        let kv_cache_shape: [usize; 5] = [N_LAYERS * 2, 1, N_HEADS, context_len, HEAD_DIM];

        let outputs = self.uclm_core.run(ort::inputs![
            "content_features" => TensorRef::from_array_view(([1usize, D_MODEL, context_len], content_features))?,
            "b_ctx" => TensorRef::from_array_view(([1usize, CONTROL_SLOTS, context_len], b_ctx))?,
            "spk_embed" => TensorRef::from_array_view(([1usize, D_SPEAKER], spk_embed))?,
            "state_cond" => TensorRef::from_array_view(([1usize, D_MODEL], state_cond))?,
            "cfg_scale" => TensorRef::from_array_view(([1usize], cfg_arr.as_slice()))?,
            "kv_cache_in" => TensorRef::from_array_view((kv_cache_shape, kv_cache_in))?,
        ])?;

        let logits_a = outputs["logits_a"].try_extract_tensor::<f32>()?;
        let logits_b = outputs["logits_b"].try_extract_tensor::<f32>()?;
        let kv_cache = outputs["kv_cache_out"].try_extract_tensor::<f32>()?;

        let la_len = logits_a.1.len().min(logits_a_out.len());
        logits_a_out[..la_len].copy_from_slice(&logits_a.1[..la_len]);

        let lb_len = logits_b.1.len().min(logits_b_out.len());
        logits_b_out[..lb_len].copy_from_slice(&logits_b.1[..lb_len]);

        let kv_len = kv_cache.1.len().min(kv_cache_out.len());
        kv_cache_out[..kv_len].copy_from_slice(&kv_cache.1[..kv_len]);

        Ok(())
    }

    /// Legacy method name for compatibility
    pub fn run_token_model(
        &mut self,
        tokens_ctx: &[i64],
        spk_embed: &[f32],
        _f0_condition: &[f32],
        voice_state_ctx: &[f32],
        kv_cache_in: &[f32],
        logits_out: &mut [f32],
        kv_cache_out: &mut [f32],
    ) -> Result<()> {
        let context_len = tokens_ctx.len() / N_CODEBOOKS;

        let mut content_features = vec![0.0f32; D_MODEL * context_len];
        self.run_vc_encoder(tokens_ctx, context_len, &mut content_features)?;

        let b_ctx: Vec<i64> = vec![0i64; CONTROL_SLOTS * context_len];

        let mut state_cond = vec![0.0f32; D_MODEL];
        let vs_len = voice_state_ctx.len().min(D_MODEL);
        state_cond[..vs_len].copy_from_slice(&voice_state_ctx[..vs_len]);

        let mut logits_b_buf = vec![0.0f32; CONTROL_SLOTS * CONTROL_VOCAB_SIZE];

        self.run_uclm_core(
            &content_features,
            &b_ctx,
            spk_embed,
            &state_cond,
            1.0,
            kv_cache_in,
            context_len,
            logits_out,
            &mut logits_b_buf,
            kv_cache_out,
        )
    }

    /// Legacy token model for old processor.rs compatibility (Vec-based).
    #[deprecated(note = "Use run_token_model with pre-allocated buffers for RT-safe")]
    pub fn run_token_model_legacy(
        &mut self,
        tokens_ctx: &[i64],
        spk_embed: &[f32],
        kv_cache_in: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let context_len = tokens_ctx.len() / N_CODEBOOKS;
        let mut logits_out = vec![0.0f32; N_CODEBOOKS * CODEBOOK_SIZE * context_len];
        let mut kv_cache_out = vec![0.0f32; kv_cache_in.len()];

        let voice_state_ctx = vec![0.0f32; D_MODEL];
        let f0_condition = vec![0.0f32; context_len * 2];

        self.run_token_model(
            tokens_ctx,
            spk_embed,
            &f0_condition,
            &voice_state_ctx,
            kv_cache_in,
            &mut logits_out,
            &mut kv_cache_out,
        )?;

        Ok((logits_out, kv_cache_out))
    }

    /// Run codec decoder: tokens → audio
    ///
    /// Inputs:
    /// - acoustic_tokens: `[1, 8]` A_t
    /// - control_tokens: `[1, 4]` B_t
    /// - voice_state: `[1, 8]`
    /// - event_trace_in: `[1, D_EVENT_TRACE]`
    /// - state_in: decoder state
    ///
    /// Output: `[1, 1, 240]` audio frame
    /// RT-safe: writes event_trace and state to pre-allocated buffers
    pub fn run_codec_decoder(
        &mut self,
        acoustic_tokens: &[i64; N_CODEBOOKS],
        control_tokens: &[i64; CONTROL_SLOTS],
        voice_state: &[f32; D_VOICE_STATE],
        event_trace_in: &[f32],
        state_in: &[f32],
        event_trace_out: &mut [f32],
        state_out: &mut [f32],
    ) -> Result<[f32; FRAME_SIZE]> {
        let outputs = self.codec_decoder.run(ort::inputs![
            "acoustic_tokens" => TensorRef::from_array_view(([1usize, N_CODEBOOKS], acoustic_tokens.as_slice()))?,
            "control_tokens" => TensorRef::from_array_view(([1usize, CONTROL_SLOTS], control_tokens.as_slice()))?,
            "voice_state" => TensorRef::from_array_view(([1usize, D_VOICE_STATE], voice_state.as_slice()))?,
            "event_trace_in" => TensorRef::from_array_view(([1usize, D_EVENT_TRACE], event_trace_in))?,
            "state_in" => TensorRef::from_array_view(([1usize, DEC_STATE_DIM, DEC_STATE_FRAMES], state_in))?,
        ])?;

        let audio = outputs["audio_frame"].try_extract_tensor::<f32>()?;
        let mut audio_arr = [0.0f32; FRAME_SIZE];
        audio_arr.copy_from_slice(&audio.1[..FRAME_SIZE]);

        let event_trace = outputs["event_trace_out"].try_extract_tensor::<f32>()?;
        let et_len = event_trace.1.len().min(event_trace_out.len());
        event_trace_out[..et_len].copy_from_slice(&event_trace.1[..et_len]);

        let state = outputs["state_out"].try_extract_tensor::<f32>()?;
        let st_len = state.1.len().min(state_out.len());
        state_out[..st_len].copy_from_slice(&state.1[..st_len]);

        Ok(audio_arr)
    }

    /// Legacy codec decoder for old processor.rs compatibility.
    /// Returns audio vec and state vec (allocates).
    #[deprecated(note = "Use run_codec_decoder with pre-allocated buffers for RT-safe")]
    pub fn run_codec_decoder_legacy(
        &mut self,
        acoustic_tokens: &[i64; N_CODEBOOKS],
        state_in: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let control_tokens = [0i64; CONTROL_SLOTS];
        let voice_state = [0.5f32; D_VOICE_STATE];
        let event_trace_in = vec![0.0f32; D_EVENT_TRACE];
        let mut event_trace_out = vec![0.0f32; D_EVENT_TRACE];
        let mut state_out = vec![0.0f32; state_in.len()];

        let audio = self.run_codec_decoder(
            acoustic_tokens,
            &control_tokens,
            &voice_state,
            &event_trace_in,
            state_in,
            &mut event_trace_out,
            &mut state_out,
        )?;

        Ok((audio.to_vec(), state_out))
    }

    pub fn run_text_encoder(
        &mut self,
        phonemes: &[i64],
        lang_id: i64,
        features_out: &mut [f32],
    ) -> Result<usize> {
        let session = self
            .text_encoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("text_encoder not loaded"))?;

        let t = phonemes.len();
        let lang_id_arr = [lang_id];
        let outputs = session.run(ort::inputs![
            "phonemes" => TensorRef::from_array_view(([1usize, t], phonemes))?,
            "lang_id" => TensorRef::from_array_view(([1usize], lang_id_arr.as_slice()))?,
        ])?;

        let features = outputs["text_features"].try_extract_tensor::<f32>()?;
        let feat_len = features.1.len().min(features_out.len());
        features_out[..feat_len].copy_from_slice(&features.1[..feat_len]);

        Ok(feat_len)
    }

    pub fn has_tts(&self) -> bool {
        self.text_encoder.is_some()
            && self.duration_predictor.is_some()
            && self.f0_predictor.is_some()
            && self.pointer_head.is_some()
    }

    /// Run pointer head: hidden_states → advance_logit + progress_delta
    ///
    /// Inputs:
    /// - hidden_states: `[1, seq_len, d_model]` flattened hidden states from UCLM core
    /// - seq_len: sequence length
    ///
    /// Outputs (RT-safe, written to pre-allocated buffers):
    /// - advance_logit_out: `[1, seq_len, 1]` advance vs hold logit
    /// - progress_delta_out: `[1, seq_len, 1]` pointer progress update (sigmoid, 0-1)
    pub fn run_pointer_head(
        &mut self,
        hidden_states: &[f32],
        seq_len: usize,
        advance_logit_out: &mut [f32],
        progress_delta_out: &mut [f32],
    ) -> Result<()> {
        let session = self
            .pointer_head
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("pointer_head not loaded"))?;

        let outputs = session.run(ort::inputs![
            "hidden_states" => TensorRef::from_array_view(([1usize, seq_len, D_MODEL], hidden_states))?,
        ])?;

        let advance_logit = outputs["advance_logit"].try_extract_tensor::<f32>()?;
        let al_len = advance_logit.1.len().min(advance_logit_out.len());
        advance_logit_out[..al_len].copy_from_slice(&advance_logit.1[..al_len]);

        let progress_delta = outputs["progress_delta"].try_extract_tensor::<f32>()?;
        let pd_len = progress_delta.1.len().min(progress_delta_out.len());
        progress_delta_out[..pd_len].copy_from_slice(&progress_delta.1[..pd_len]);

        Ok(())
    }

    pub fn run_duration_predictor(
        &mut self,
        text_features: &[f32],
        seq_len: usize,
        style: &[f32],
        durations_out: &mut [f32],
    ) -> Result<usize> {
        let session = self
            .duration_predictor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("duration_predictor not loaded"))?;

        let outputs = session.run(ort::inputs![
            "text_features" => TensorRef::from_array_view(([1usize, D_TEXT_ENCODER, seq_len], text_features))?,
            "style" => TensorRef::from_array_view(([1usize, N_ACOUSTIC_PARAMS], style))?,
        ])?;

        let durations = outputs["durations"].try_extract_tensor::<f32>()?;
        let dur_len = durations.1.len().min(durations_out.len());
        durations_out[..dur_len].copy_from_slice(&durations.1[..dur_len]);

        Ok(dur_len)
    }

    pub fn run_f0_predictor(
        &mut self,
        expanded_features: &[f32],
        n_frames: usize,
        style: &[f32],
        f0_out: &mut [f32],
        voiced_out: &mut [f32],
    ) -> Result<()> {
        let session = self
            .f0_predictor
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("f0_predictor not loaded"))?;

        let outputs = session.run(ort::inputs![
            "expanded_features" => TensorRef::from_array_view(([1usize, D_TEXT_ENCODER, n_frames], expanded_features))?,
            "style" => TensorRef::from_array_view(([1usize, N_ACOUSTIC_PARAMS], style))?,
        ])?;

        let f0 = outputs["f0"].try_extract_tensor::<f32>()?;
        let voiced = outputs["voiced"].try_extract_tensor::<f32>()?;

        let f0_len = f0.1.len().min(f0_out.len());
        f0_out[..f0_len].copy_from_slice(&f0.1[..f0_len]);

        let voiced_len = voiced.1.len().min(voiced_out.len());
        voiced_out[..voiced_len].copy_from_slice(&voiced.1[..voiced_len]);

        Ok(())
    }

    pub fn run_content_synthesizer(
        &mut self,
        expanded_features: &[f32],
        n_frames: usize,
        content_out: &mut [f32],
    ) -> Result<usize> {
        let session = self
            .text_encoder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("text_encoder not loaded (content_synthesizer)"))?;

        let outputs = session.run(ort::inputs![
            "expanded_features" => TensorRef::from_array_view(([1usize, D_TEXT_ENCODER, n_frames], expanded_features))?,
        ])?;

        let content = outputs["content"].try_extract_tensor::<f32>()?;
        let cont_len = content.1.len().min(content_out.len());
        content_out[..cont_len].copy_from_slice(&content.1[..cont_len]);

        Ok(cont_len)
    }
}
