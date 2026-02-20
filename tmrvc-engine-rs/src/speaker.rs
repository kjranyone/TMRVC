#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::constants::*;

const MAGIC: &[u8; 4] = b"TMSP";
const VERSION: u32 = 2;
// 4 (magic) + 4 (version) + 4 (embed_size) + 4 (lora_size) + 4 (metadata_size) + 4 (thumbnail_size)
const HEADER_SIZE: usize = 24;
const CHECKSUM_SIZE: usize = 32; // SHA-256

/// Metadata embedded in a .tmrvc_speaker v2 file.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct SpeakerMetadata {
    pub profile_name: String,
    pub author_name: String,
    #[serde(default)]
    pub co_author_name: String,
    #[serde(default)]
    pub licence_url: String,
    #[serde(default)]
    pub thumbnail_b64: String,
    pub created_at: String,
    pub description: String,
    pub source_audio_files: Vec<String>,
    pub source_sample_count: u64,
    pub training_mode: String,
    pub checkpoint_name: String,
    #[serde(default)]
    pub voice_source_preset: Option<Vec<f32>>,
    #[serde(default)]
    pub voice_source_param_names: Vec<String>,
}

/// Parsed .tmrvc_speaker v2 file.
pub struct SpeakerFile {
    pub spk_embed: [f32; D_SPEAKER],
    pub lora_delta: Vec<f32>,
    pub metadata: SpeakerMetadata,
}

impl SpeakerFile {
    /// Extract voice source preset as a fixed-size array, if present in metadata.
    pub fn voice_source_preset(&self) -> Option<[f32; N_VOICE_SOURCE_PARAMS]> {
        let vec = self.metadata.voice_source_preset.as_ref()?;
        if vec.len() != N_VOICE_SOURCE_PARAMS {
            return None;
        }
        let mut arr = [0.0f32; N_VOICE_SOURCE_PARAMS];
        arr.copy_from_slice(vec);
        Some(arr)
    }

    /// Load and validate a .tmrvc_speaker v2 file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).context("Failed to read speaker file")?;

        // Minimum size: header + embed + lora + checksum (no metadata/thumbnail)
        let min_size = HEADER_SIZE + D_SPEAKER * 4 + LORA_DELTA_SIZE * 4 + CHECKSUM_SIZE;
        if data.len() < min_size {
            bail!(
                "Speaker file too small: expected at least {} bytes, got {}",
                min_size,
                data.len()
            );
        }

        // Magic
        if &data[0..4] != MAGIC {
            bail!("Invalid magic: expected TMSP");
        }

        // Version
        let version = u32::from_le_bytes(data[4..8].try_into().expect("version field is 4 bytes"));
        if version != VERSION {
            bail!("Unsupported version: {} (expected {})", version, VERSION);
        }

        // Embed size
        let embed_size =
            u32::from_le_bytes(data[8..12].try_into().expect("embed_size field is 4 bytes"))
                as usize;
        if embed_size != D_SPEAKER {
            bail!(
                "Speaker embed size mismatch: expected {}, got {}",
                D_SPEAKER,
                embed_size
            );
        }

        // Lora size
        let lora_size =
            u32::from_le_bytes(data[12..16].try_into().expect("lora_size field is 4 bytes"))
                as usize;
        if lora_size != LORA_DELTA_SIZE {
            bail!(
                "LoRA delta size mismatch: expected {}, got {}",
                LORA_DELTA_SIZE,
                lora_size
            );
        }

        // Metadata size
        let metadata_size = u32::from_le_bytes(
            data[16..20]
                .try_into()
                .expect("metadata_size field is 4 bytes"),
        ) as usize;

        // Thumbnail size
        let thumbnail_size = u32::from_le_bytes(
            data[20..24]
                .try_into()
                .expect("thumbnail_size field is 4 bytes"),
        ) as usize;

        // Verify total size
        let expected_size = HEADER_SIZE
            + D_SPEAKER * 4
            + LORA_DELTA_SIZE * 4
            + metadata_size
            + thumbnail_size
            + CHECKSUM_SIZE;
        if data.len() != expected_size {
            bail!(
                "Speaker file size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            );
        }

        // SHA-256 verification
        let payload = &data[..data.len() - CHECKSUM_SIZE];
        let stored_hash = &data[data.len() - CHECKSUM_SIZE..];
        let computed_hash = Sha256::digest(payload);
        if computed_hash.as_slice() != stored_hash {
            bail!("SHA-256 checksum mismatch");
        }

        // Parse spk_embed
        let mut spk_embed = [0.0f32; D_SPEAKER];
        let embed_bytes = &data[HEADER_SIZE..HEADER_SIZE + D_SPEAKER * 4];
        for (i, chunk) in embed_bytes.chunks_exact(4).enumerate() {
            spk_embed[i] = f32::from_le_bytes(chunk.try_into().expect("embed chunk is 4 bytes"));
        }

        // Parse lora_delta
        let lora_offset = HEADER_SIZE + D_SPEAKER * 4;
        let lora_bytes = &data[lora_offset..lora_offset + LORA_DELTA_SIZE * 4];
        let mut lora_delta = vec![0.0f32; LORA_DELTA_SIZE];
        for (i, chunk) in lora_bytes.chunks_exact(4).enumerate() {
            lora_delta[i] = f32::from_le_bytes(chunk.try_into().expect("lora chunk is 4 bytes"));
        }

        // Parse metadata JSON
        let meta_offset = lora_offset + LORA_DELTA_SIZE * 4;
        let metadata = if metadata_size > 0 {
            let meta_bytes = &data[meta_offset..meta_offset + metadata_size];
            serde_json::from_slice(meta_bytes).context("Failed to parse metadata JSON")?
        } else {
            SpeakerMetadata::default()
        };

        // Skip legacy raw thumbnail section (thumbnail is in metadata.thumbnail_b64)
        // thumbnail_size bytes are accounted for in total size validation above

        Ok(Self {
            spk_embed,
            lora_delta,
            metadata,
        })
    }

    /// Save a .tmrvc_speaker v2 file.
    ///
    /// Layout: MAGIC(4) + VERSION(4) + embed_size(4) + lora_size(4)
    ///       + metadata_size(4) + thumbnail_size(4)
    ///       + spk_embed(D_SPEAKER*4) + lora_delta(LORA_DELTA_SIZE*4)
    ///       + metadata_json(variable) + thumbnail_png(variable)
    ///       + SHA-256(32)
    pub fn save(&self, path: &Path) -> Result<()> {
        let metadata_json =
            serde_json::to_vec(&self.metadata).context("Failed to serialize metadata")?;
        let metadata_size = metadata_json.len();
        let thumbnail_size: usize = 0; // thumbnail stored as base64 in metadata JSON

        let payload_size =
            HEADER_SIZE + D_SPEAKER * 4 + LORA_DELTA_SIZE * 4 + metadata_size + thumbnail_size;
        let mut buf = Vec::with_capacity(payload_size + CHECKSUM_SIZE);

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(D_SPEAKER as u32).to_le_bytes());
        buf.extend_from_slice(&(LORA_DELTA_SIZE as u32).to_le_bytes());
        buf.extend_from_slice(&(metadata_size as u32).to_le_bytes());
        buf.extend_from_slice(&(thumbnail_size as u32).to_le_bytes());

        // spk_embed
        for &v in &self.spk_embed {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // lora_delta
        assert_eq!(self.lora_delta.len(), LORA_DELTA_SIZE);
        for &v in &self.lora_delta {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // metadata JSON
        buf.extend_from_slice(&metadata_json);

        // thumbnail_size = 0: no raw thumbnail section (stored as base64 in metadata)

        // SHA-256
        let hash = Sha256::digest(&buf);
        buf.extend_from_slice(&hash);

        let mut file = fs::File::create(path).context("Failed to create speaker file")?;
        file.write_all(&buf)
            .context("Failed to write speaker file")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64;
    use std::io::Write;

    fn default_metadata() -> SpeakerMetadata {
        SpeakerMetadata::default()
    }

    #[test]
    fn save_load_roundtrip() {
        let mut spk_embed = [0.0f32; D_SPEAKER];
        for (i, v) in spk_embed.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }
        let mut lora_delta = vec![0.0f32; LORA_DELTA_SIZE];
        lora_delta[0] = 1.23;
        lora_delta[LORA_DELTA_SIZE - 1] = -4.56;

        let original = SpeakerFile {
            spk_embed,
            lora_delta,
            metadata: default_metadata(),
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v2_roundtrip.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(original.spk_embed, loaded.spk_embed);
        assert_eq!(original.lora_delta, loaded.lora_delta);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn save_load_zero_lora() {
        let spk_embed = [0.5f32; D_SPEAKER];
        let lora_delta = vec![0.0f32; LORA_DELTA_SIZE];

        let original = SpeakerFile {
            spk_embed,
            lora_delta,
            metadata: default_metadata(),
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v2_zero_lora.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert!(loaded.lora_delta.iter().all(|&v| v == 0.0));
        assert_eq!(original.spk_embed, loaded.spk_embed);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_detects_corruption() {
        let spk = SpeakerFile {
            spk_embed: [1.0; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            metadata: default_metadata(),
        };
        let dir = std::env::temp_dir();
        let path = dir.join("test_v2_corrupt.tmrvc_speaker");
        spk.save(&path).expect("save failed");

        // Corrupt one byte in the embed region
        let mut data = fs::read(&path).unwrap();
        let mid = HEADER_SIZE + 100; // well inside spk_embed
        data[mid] ^= 0xFF;
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        drop(f);

        assert!(SpeakerFile::load(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn metadata_roundtrip() {
        let metadata = SpeakerMetadata {
            profile_name: "Test Speaker".to_string(),
            author_name: "Test Author".to_string(),
            co_author_name: "Co-Author".to_string(),
            licence_url: "https://example.com/licence".to_string(),
            thumbnail_b64: "dGVzdA==".to_string(), // base64("test")
            created_at: "2026-02-18T12:00:00Z".to_string(),
            description: "A test voice profile".to_string(),
            source_audio_files: vec!["ref1.wav".to_string(), "ref2.wav".to_string()],
            source_sample_count: 480000,
            training_mode: "embedding".to_string(),
            checkpoint_name: "".to_string(),
            voice_source_preset: None,
            voice_source_param_names: Vec::new(),
        };

        let original = SpeakerFile {
            spk_embed: [0.1; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            metadata: metadata.clone(),
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v2_metadata.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(loaded.metadata, metadata);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn thumbnail_b64_roundtrip() {
        use base64::{engine::general_purpose::STANDARD, Engine};

        // Fake PNG data
        let thumbnail_bytes = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4, 5,
        ];
        let b64 = STANDARD.encode(&thumbnail_bytes);

        let original = SpeakerFile {
            spk_embed: [0.2; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            metadata: SpeakerMetadata {
                profile_name: "Thumb Test".to_string(),
                thumbnail_b64: b64.clone(),
                ..Default::default()
            },
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v2_thumbnail_b64.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(loaded.metadata.thumbnail_b64, b64);
        assert_eq!(loaded.metadata.profile_name, "Thumb Test");

        // Verify base64 decodes back to original bytes
        let decoded = STANDARD.decode(&loaded.metadata.thumbnail_b64).unwrap();
        assert_eq!(decoded, thumbnail_bytes);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn empty_thumbnail_b64() {
        let original = SpeakerFile {
            spk_embed: [0.3; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            metadata: SpeakerMetadata {
                profile_name: "No Thumb".to_string(),
                ..Default::default()
            },
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v2_no_thumb_b64.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert!(loaded.metadata.thumbnail_b64.is_empty());
        assert_eq!(loaded.metadata.profile_name, "No Thumb");

        let _ = fs::remove_file(&path);
    }
}
