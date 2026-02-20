use std::path::Path;

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::constants::*;

const LORA_OUTPUT_NAME: &str = "lora_delta";
const SPK_EMBED_OUTPUT_NAME: &str = "spk_embed";

/// ONNX session wrapper for the speaker encoder model.
///
/// Runs offline: `mel_ref[1, N_MELS, T_ref]` → `spk_embed[1, D_SPEAKER]` + optional `lora_delta[1, LORA_DELTA_SIZE]`.
pub struct SpeakerEncoderSession {
    session: Session,
    has_lora_output: bool,
}

impl SpeakerEncoderSession {
    /// Load speaker_encoder.onnx from the given path.
    pub fn load(model_path: &Path) -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(2)?
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load speaker_encoder: {:?}", model_path))?;

        let has_lora_output = session
            .outputs()
            .iter()
            .any(|out| out.name() == LORA_OUTPUT_NAME);

        log::info!(
            "SpeakerEncoderSession loaded (lora_output={})",
            has_lora_output
        );

        Ok(Self {
            session,
            has_lora_output,
        })
    }

    /// Whether this model produces a `lora_delta` output.
    pub fn has_lora_output(&self) -> bool {
        self.has_lora_output
    }

    /// Run the speaker encoder on a mel spectrogram.
    ///
    /// - `mel_data`: `[N_MELS, num_frames]` row-major
    /// - Returns `(spk_embed[D_SPEAKER], lora_delta[LORA_DELTA_SIZE])`
    /// - If the model has no lora_delta output, lora_delta is all zeros.
    /// - The spk_embed is L2-normalized.
    pub fn run(
        &mut self,
        mel_data: &[f32],
        num_frames: usize,
    ) -> Result<([f32; D_SPEAKER], Vec<f32>)> {
        assert_eq!(mel_data.len(), N_MELS * num_frames);

        let outputs = self.session.run(ort::inputs![
            "mel_ref" => TensorRef::from_array_view(([1usize, N_MELS, num_frames], mel_data))?,
        ])?;

        // Extract spk_embed
        let embed_tensor = outputs[SPK_EMBED_OUTPUT_NAME].try_extract_tensor::<f32>()?;
        let embed_slice = embed_tensor.1;
        let mut spk_embed = [0.0f32; D_SPEAKER];
        spk_embed.copy_from_slice(&embed_slice[..D_SPEAKER]);

        // L2 normalize
        let norm = spk_embed.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut spk_embed {
                *v /= norm;
            }
        }

        // Extract lora_delta (or zero)
        let lora_delta = if self.has_lora_output {
            let lora_tensor = outputs[LORA_OUTPUT_NAME].try_extract_tensor::<f32>()?;
            lora_tensor.1.to_vec()
        } else {
            vec![0.0f32; LORA_DELTA_SIZE]
        };

        Ok((spk_embed, lora_delta))
    }
}
