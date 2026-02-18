#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 4] = b"TMST";
const VERSION: u32 = 1;
// 4 (magic) + 4 (version) + 4 (metadata_size) + 4 * 3 (style params)
const HEADER_SIZE: usize = 24;
const CHECKSUM_SIZE: usize = 32; // SHA-256

/// Metadata embedded in a .tmrvc_style file.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct StyleMetadata {
    pub display_name: String,
    pub created_at: String,
    pub description: String,
    pub source_audio_files: Vec<String>,
    pub source_sample_count: u64,
}

/// Parsed .tmrvc_style file.
#[derive(Debug, Clone)]
pub struct StyleFile {
    /// Reference prosody target: mean voiced log(F0 + 1).
    pub target_log_f0: f32,
    /// Articulation proxy from reference utterances.
    pub target_articulation: f32,
    /// Fraction of voiced frames in reference utterances.
    pub voiced_ratio: f32,
    pub metadata: StyleMetadata,
}

impl StyleFile {
    /// Load and validate a .tmrvc_style file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).context("Failed to read style file")?;

        let min_size = HEADER_SIZE + CHECKSUM_SIZE;
        if data.len() < min_size {
            bail!(
                "Style file too small: expected at least {} bytes, got {}",
                min_size,
                data.len()
            );
        }

        if &data[0..4] != MAGIC {
            bail!("Invalid style magic: expected TMST");
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            bail!(
                "Unsupported style version: {} (expected {})",
                version,
                VERSION
            );
        }

        let metadata_size = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let expected_size = HEADER_SIZE + metadata_size + CHECKSUM_SIZE;
        if data.len() != expected_size {
            bail!(
                "Style file size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            );
        }

        let payload = &data[..data.len() - CHECKSUM_SIZE];
        let stored_hash = &data[data.len() - CHECKSUM_SIZE..];
        let computed_hash = Sha256::digest(payload);
        if computed_hash.as_slice() != stored_hash {
            bail!("Style SHA-256 checksum mismatch");
        }

        let target_log_f0 = f32::from_le_bytes(data[12..16].try_into().unwrap());
        let target_articulation = f32::from_le_bytes(data[16..20].try_into().unwrap());
        let voiced_ratio = f32::from_le_bytes(data[20..24].try_into().unwrap());

        let metadata = if metadata_size > 0 {
            let meta_bytes = &data[HEADER_SIZE..HEADER_SIZE + metadata_size];
            serde_json::from_slice(meta_bytes).context("Failed to parse style metadata JSON")?
        } else {
            StyleMetadata::default()
        };

        Ok(Self {
            target_log_f0,
            target_articulation,
            voiced_ratio,
            metadata,
        })
    }

    /// Save a .tmrvc_style file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let metadata_json =
            serde_json::to_vec(&self.metadata).context("Failed to serialize style metadata")?;
        let metadata_size = metadata_json.len();

        let payload_size = HEADER_SIZE + metadata_size;
        let mut buf = Vec::with_capacity(payload_size + CHECKSUM_SIZE);

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(metadata_size as u32).to_le_bytes());
        buf.extend_from_slice(&self.target_log_f0.to_le_bytes());
        buf.extend_from_slice(&self.target_articulation.to_le_bytes());
        buf.extend_from_slice(&self.voiced_ratio.to_le_bytes());

        // metadata JSON
        buf.extend_from_slice(&metadata_json);

        // SHA-256
        let hash = Sha256::digest(&buf);
        buf.extend_from_slice(&hash);

        let mut file = fs::File::create(path).context("Failed to create style file")?;
        file.write_all(&buf).context("Failed to write style file")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn style_roundtrip() {
        let style = StyleFile {
            target_log_f0: 5.21,
            target_articulation: 1.05,
            voiced_ratio: 0.73,
            metadata: StyleMetadata {
                display_name: "Demo Style".to_string(),
                created_at: "2026-02-18T12:00:00Z".to_string(),
                description: "roundtrip test".to_string(),
                source_audio_files: vec!["a.wav".to_string(), "b.wav".to_string()],
                source_sample_count: 12345,
            },
        };

        let path = std::env::temp_dir().join("tmrvc_style_roundtrip.tmrvc_style");
        style.save(&path).expect("save failed");
        let loaded = StyleFile::load(&path).expect("load failed");

        assert!((loaded.target_log_f0 - style.target_log_f0).abs() < 1e-6);
        assert!((loaded.target_articulation - style.target_articulation).abs() < 1e-6);
        assert!((loaded.voiced_ratio - style.voiced_ratio).abs() < 1e-6);
        assert_eq!(loaded.metadata, style.metadata);

        let _ = fs::remove_file(path);
    }
}
