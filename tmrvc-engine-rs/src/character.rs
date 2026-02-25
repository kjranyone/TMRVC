#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::constants::*;

const MAGIC: &[u8; 4] = b"TMCH";
const VERSION: u32 = 1;
// 4 (magic) + 4 (version) + 4 (spk) + 4 (lora) + 4 (vs) + 4 (style) + 4 (profile)
const HEADER_SIZE: usize = 28;
const CHECKSUM_SIZE: usize = 32; // SHA-256

/// Character profile metadata embedded in a .tmrvc_character file.
#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq)]
pub struct CharacterProfile {
    pub name: String,
    pub personality: String,
    pub voice_description: String,
    pub language: String,
}

/// Parsed .tmrvc_character v1 file.
///
/// Extends .tmrvc_speaker with voice source preset, default emotion style,
/// and a character profile for context-aware TTS.
#[derive(Debug)]
pub struct CharacterFile {
    pub spk_embed: [f32; D_SPEAKER],
    pub lora_delta: Vec<f32>,
    pub voice_source_preset: [f32; N_VOICE_SOURCE_PARAMS],
    pub default_style: [f32; D_STYLE],
    pub profile: CharacterProfile,
}

impl CharacterFile {
    /// Load and validate a .tmrvc_character v1 file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).context("Failed to read character file")?;

        let min_size = HEADER_SIZE
            + D_SPEAKER * 4
            + LORA_DELTA_SIZE * 4
            + N_VOICE_SOURCE_PARAMS * 4
            + D_STYLE * 4
            + CHECKSUM_SIZE;
        if data.len() < min_size {
            bail!(
                "Character file too small: expected at least {} bytes, got {}",
                min_size,
                data.len()
            );
        }

        if &data[0..4] != MAGIC {
            bail!("Invalid magic: expected TMCH");
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            bail!(
                "Unsupported character version: {} (expected {})",
                version,
                VERSION
            );
        }

        let spk_size = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if spk_size != D_SPEAKER {
            bail!(
                "Speaker embed size mismatch: expected {}, got {}",
                D_SPEAKER,
                spk_size
            );
        }

        let lora_size = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        if lora_size != LORA_DELTA_SIZE {
            bail!(
                "LoRA delta size mismatch: expected {}, got {}",
                LORA_DELTA_SIZE,
                lora_size
            );
        }

        let vs_size = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let style_size = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let profile_size = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;

        let expected_size = HEADER_SIZE
            + spk_size * 4
            + lora_size * 4
            + vs_size * 4
            + style_size * 4
            + profile_size
            + CHECKSUM_SIZE;
        if data.len() != expected_size {
            bail!(
                "Character file size mismatch: expected {} bytes, got {}",
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

        // Parse fields
        let mut offset = HEADER_SIZE;

        let mut spk_embed = [0.0f32; D_SPEAKER];
        for (i, chunk) in data[offset..offset + D_SPEAKER * 4].chunks_exact(4).enumerate() {
            spk_embed[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        offset += D_SPEAKER * 4;

        let mut lora_delta = vec![0.0f32; LORA_DELTA_SIZE];
        for (i, chunk) in data[offset..offset + LORA_DELTA_SIZE * 4]
            .chunks_exact(4)
            .enumerate()
        {
            lora_delta[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        offset += LORA_DELTA_SIZE * 4;

        let mut voice_source_preset = [0.0f32; N_VOICE_SOURCE_PARAMS];
        for (i, chunk) in data[offset..offset + vs_size * 4].chunks_exact(4).enumerate() {
            if i < N_VOICE_SOURCE_PARAMS {
                voice_source_preset[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        offset += vs_size * 4;

        let mut default_style = [0.0f32; D_STYLE];
        for (i, chunk) in data[offset..offset + style_size * 4]
            .chunks_exact(4)
            .enumerate()
        {
            if i < D_STYLE {
                default_style[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        offset += style_size * 4;

        let profile = if profile_size > 0 {
            let profile_bytes = &data[offset..offset + profile_size];
            serde_json::from_slice(profile_bytes).context("Failed to parse character profile JSON")?
        } else {
            CharacterProfile::default()
        };

        Ok(Self {
            spk_embed,
            lora_delta,
            voice_source_preset,
            default_style,
            profile,
        })
    }

    /// Save a .tmrvc_character v1 file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let profile_json =
            serde_json::to_vec(&self.profile).context("Failed to serialize profile")?;

        let payload_size = HEADER_SIZE
            + D_SPEAKER * 4
            + LORA_DELTA_SIZE * 4
            + N_VOICE_SOURCE_PARAMS * 4
            + D_STYLE * 4
            + profile_json.len();
        let mut buf = Vec::with_capacity(payload_size + CHECKSUM_SIZE);

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(D_SPEAKER as u32).to_le_bytes());
        buf.extend_from_slice(&(LORA_DELTA_SIZE as u32).to_le_bytes());
        buf.extend_from_slice(&(N_VOICE_SOURCE_PARAMS as u32).to_le_bytes());
        buf.extend_from_slice(&(D_STYLE as u32).to_le_bytes());
        buf.extend_from_slice(&(profile_json.len() as u32).to_le_bytes());

        // spk_embed
        for &v in &self.spk_embed {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // lora_delta
        assert_eq!(self.lora_delta.len(), LORA_DELTA_SIZE);
        for &v in &self.lora_delta {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // voice_source_preset
        for &v in &self.voice_source_preset {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // default_style
        for &v in &self.default_style {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // profile JSON
        buf.extend_from_slice(&profile_json);

        // SHA-256
        let hash = Sha256::digest(&buf);
        buf.extend_from_slice(&hash);

        let mut file = fs::File::create(path).context("Failed to create character file")?;
        file.write_all(&buf)
            .context("Failed to write character file")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_profile() -> CharacterProfile {
        CharacterProfile {
            name: "Sakura".to_string(),
            personality: "Bright and cheerful".to_string(),
            voice_description: "High-pitched, slightly breathy".to_string(),
            language: "ja".to_string(),
        }
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

        let mut voice_source = [0.0f32; N_VOICE_SOURCE_PARAMS];
        voice_source[0] = 0.5;
        voice_source[N_VOICE_SOURCE_PARAMS - 1] = -0.3;

        let mut default_style = [0.0f32; D_STYLE];
        default_style[0] = 0.8;
        default_style[D_STYLE - 1] = -0.2;

        let original = CharacterFile {
            spk_embed,
            lora_delta,
            voice_source_preset: voice_source,
            default_style,
            profile: default_profile(),
        };

        let path = std::env::temp_dir().join("test_character_roundtrip.tmrvc_character");
        original.save(&path).expect("save failed");

        let loaded = CharacterFile::load(&path).expect("load failed");
        assert_eq!(original.spk_embed, loaded.spk_embed);
        assert_eq!(original.lora_delta, loaded.lora_delta);
        assert_eq!(original.voice_source_preset, loaded.voice_source_preset);
        assert_eq!(original.default_style, loaded.default_style);
        assert_eq!(original.profile, loaded.profile);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn save_load_zero_fields() {
        let original = CharacterFile {
            spk_embed: [0.5; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            voice_source_preset: [0.0; N_VOICE_SOURCE_PARAMS],
            default_style: [0.0; D_STYLE],
            profile: CharacterProfile::default(),
        };

        let path = std::env::temp_dir().join("test_character_zeros.tmrvc_character");
        original.save(&path).expect("save failed");

        let loaded = CharacterFile::load(&path).expect("load failed");
        assert!(loaded.lora_delta.iter().all(|&v| v == 0.0));
        assert!(loaded.voice_source_preset.iter().all(|&v| v == 0.0));
        assert!(loaded.default_style.iter().all(|&v| v == 0.0));
        assert_eq!(loaded.profile.name, "");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_detects_corruption() {
        let original = CharacterFile {
            spk_embed: [1.0; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            voice_source_preset: [0.0; N_VOICE_SOURCE_PARAMS],
            default_style: [0.0; D_STYLE],
            profile: default_profile(),
        };

        let path = std::env::temp_dir().join("test_character_corrupt.tmrvc_character");
        original.save(&path).expect("save failed");

        let mut data = fs::read(&path).unwrap();
        let mid = HEADER_SIZE + 100;
        data[mid] ^= 0xFF;
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        drop(f);

        assert!(CharacterFile::load(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let original = CharacterFile {
            spk_embed: [0.0; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            voice_source_preset: [0.0; N_VOICE_SOURCE_PARAMS],
            default_style: [0.0; D_STYLE],
            profile: CharacterProfile::default(),
        };

        let path = std::env::temp_dir().join("test_character_bad_magic.tmrvc_character");
        original.save(&path).expect("save failed");

        let mut data = fs::read(&path).unwrap();
        data[0] = b'X';
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&data).unwrap();
        drop(f);

        let err = CharacterFile::load(&path).unwrap_err();
        assert!(err.to_string().contains("magic"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn profile_json_roundtrip() {
        let profile = CharacterProfile {
            name: "テストキャラ".to_string(),
            personality: "明るく元気、たまにツンデレ".to_string(),
            voice_description: "高めの声、やや息混じり".to_string(),
            language: "ja".to_string(),
        };

        let original = CharacterFile {
            spk_embed: [0.1; D_SPEAKER],
            lora_delta: vec![0.0; LORA_DELTA_SIZE],
            voice_source_preset: [0.0; N_VOICE_SOURCE_PARAMS],
            default_style: [0.0; D_STYLE],
            profile: profile.clone(),
        };

        let path = std::env::temp_dir().join("test_character_profile.tmrvc_character");
        original.save(&path).expect("save failed");

        let loaded = CharacterFile::load(&path).expect("load failed");
        assert_eq!(loaded.profile, profile);

        let _ = fs::remove_file(&path);
    }
}
