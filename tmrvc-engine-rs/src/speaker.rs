#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::constants::*;

const MAGIC: &[u8; 4] = b"TMSP";
const VERSION_V3: u32 = 3;
const VERSION_V4: u32 = 4;
const HEADER_SIZE: usize = 32;
const HEADER_SIZE_V4: usize = 40;
const CHECKSUM_SIZE: usize = 32;

// Flags
const FLAG_HAS_STYLE: u32 = 1 << 0;
const FLAG_HAS_REF_TOKENS: u32 = 1 << 1;
const FLAG_HAS_LORA: u32 = 1 << 2;
const FLAG_HAS_ACTING_LATENT: u32 = 1 << 3;

/// Metadata embedded in a .tmrvc_speaker file.
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
    #[serde(default)]
    pub adaptation_level: String,
    pub checkpoint_name: String,
    #[serde(default)]
    pub voice_source_preset: Option<Vec<f32>>,
    #[serde(default)]
    pub voice_source_param_names: Vec<String>,
    #[serde(default)]
    pub style_embed: Option<Vec<f32>>,
    #[serde(default)]
    pub reference_tokens: Option<Vec<i32>>,
    #[serde(default)]
    pub ssl_state: Option<Vec<f32>>,
    #[serde(default)]
    pub acting_latent: Option<Vec<f32>>,
    #[serde(default = "default_f0_mean")]
    pub f0_mean: f32,
}

fn default_f0_mean() -> f32 {
    220.0
}

/// Parsed .tmrvc_speaker v3/v4 file.
pub struct SpeakerFile {
    pub spk_embed: [f32; D_SPEAKER],
    pub f0_mean: f32,
    pub style_embed: Option<Vec<f32>>,
    pub reference_tokens: Option<Vec<i32>>,
    pub lora_delta: Option<Vec<f32>>,
    pub ssl_state: Option<Vec<f32>>,
    pub acting_latent: Option<Vec<f32>>,
    pub metadata: SpeakerMetadata,
    pub version: u32,
}

impl SpeakerFile {
    /// Get lora_delta, defaulting to zeros if not present.
    pub fn lora_delta_or_zeros(&self) -> Vec<f32> {
        self.lora_delta
            .clone()
            .unwrap_or_else(|| vec![0.0f32; LORA_DELTA_SIZE])
    }

    /// Get ssl_state, defaulting to zeros if not present.
    pub fn ssl_state_or_zeros(&self) -> Vec<f32> {
        self.ssl_state
            .clone()
            .unwrap_or_else(|| vec![0.0f32; D_VOICE_STATE_SSL])
    }

    /// Get acting_latent, defaulting to zeros if not present.
    pub fn acting_latent_or_zeros(&self) -> Vec<f32> {
        self.acting_latent
            .clone()
            .unwrap_or_else(|| vec![0.0f32; D_ACTING_LATENT])
    }

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

    /// Load and validate a .tmrvc_speaker v3 file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).context("Failed to read speaker file")?;

        // Minimum size: header + spk_embed + f0_mean + checksum
        let min_size = HEADER_SIZE + D_SPEAKER * 4 + 4 + CHECKSUM_SIZE;
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
        if version != VERSION_V3 && version != VERSION_V4 {
            bail!("Unsupported version: {} (expected {} or {})", version, VERSION_V3, VERSION_V4);
        }

        // Parse v3 header
        let flags = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let spk_embed_size = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let style_embed_size = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let ref_tokens_frames = u32::from_le_bytes(data[20..24].try_into().unwrap()) as usize;
        let lora_size = u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize;
        let metadata_size = u32::from_le_bytes(data[28..32].try_into().unwrap()) as usize;

        // V4 extended header: acting_latent_size at bytes 32..36
        let acting_latent_size = if version == VERSION_V4 {
            u32::from_le_bytes(data[32..36].try_into().unwrap()) as usize
        } else {
            0
        };

        if spk_embed_size != D_SPEAKER {
            bail!(
                "Speaker embed size mismatch: expected {}, got {}",
                D_SPEAKER,
                spk_embed_size
            );
        }
        if (flags & FLAG_HAS_STYLE != 0) && (style_embed_size != D_STYLE) {
            bail!(
                "Style embed size mismatch: expected {}, got {}",
                D_STYLE,
                style_embed_size
            );
        }

        // Calculate expected size
        let header_sz = if version == VERSION_V4 { HEADER_SIZE_V4 } else { HEADER_SIZE };
        let mut expected_size = header_sz;
        expected_size += D_SPEAKER * 4; // spk_embed
        expected_size += 4; // f0_mean
        if flags & FLAG_HAS_STYLE != 0 {
            expected_size += D_STYLE * 4;
        }
        if flags & FLAG_HAS_REF_TOKENS != 0 {
            expected_size += ref_tokens_frames * 4 * 4; // [T, 4] int32
        }
        if flags & FLAG_HAS_LORA != 0 {
            expected_size += lora_size * 4;
        }
        if flags & FLAG_HAS_ACTING_LATENT != 0 {
            expected_size += acting_latent_size * 4;
        }
        expected_size += metadata_size;
        expected_size += CHECKSUM_SIZE;

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

        let mut offset = header_sz;

        // Parse spk_embed
        let mut spk_embed = [0.0f32; D_SPEAKER];
        for (i, chunk) in data[offset..offset + D_SPEAKER * 4]
            .chunks_exact(4)
            .enumerate()
        {
            spk_embed[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        offset += D_SPEAKER * 4;

        // Parse f0_mean (4 bytes after spk_embed)
        let f0_mean = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let f0_mean = if f0_mean > 0.0 {
            f0_mean
        } else {
            default_f0_mean()
        };
        offset += 4;

        // Parse style_embed (optional)
        let style_embed = if flags & FLAG_HAS_STYLE != 0 {
            let mut se = vec![0.0f32; D_STYLE];
            for (i, chunk) in data[offset..offset + D_STYLE * 4]
                .chunks_exact(4)
                .enumerate()
            {
                se[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            offset += D_STYLE * 4;
            Some(se)
        } else {
            None
        };

        // Parse reference_tokens (optional)
        let reference_tokens = if flags & FLAG_HAS_REF_TOKENS != 0 && ref_tokens_frames > 0 {
            let bytes_len = ref_tokens_frames * 4 * 4;
            let mut rt = vec![0i32; ref_tokens_frames * 4];
            for (i, chunk) in data[offset..offset + bytes_len].chunks_exact(4).enumerate() {
                rt[i] = i32::from_le_bytes(chunk.try_into().unwrap());
            }
            offset += bytes_len;
            Some(rt)
        } else {
            None
        };

        // Parse lora_delta (optional)
        let lora_delta = if flags & FLAG_HAS_LORA != 0 && lora_size > 0 {
            let mut ld = vec![0.0f32; lora_size];
            for (i, chunk) in data[offset..offset + lora_size * 4]
                .chunks_exact(4)
                .enumerate()
            {
                ld[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            offset += lora_size * 4;
            Some(ld)
        } else {
            None
        };

        // Parse acting_latent (optional, V4 only)
        let acting_latent = if flags & FLAG_HAS_ACTING_LATENT != 0 && acting_latent_size > 0 {
            let mut al = vec![0.0f32; acting_latent_size];
            for (i, chunk) in data[offset..offset + acting_latent_size * 4]
                .chunks_exact(4)
                .enumerate()
            {
                al[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            offset += acting_latent_size * 4;
            Some(al)
        } else {
            None
        };

        // Parse metadata JSON
        let metadata = if metadata_size > 0 {
            serde_json::from_slice(&data[offset..offset + metadata_size])
                .context("Failed to parse metadata JSON")?
        } else {
            SpeakerMetadata::default()
        };

        // Binary section is the single source of truth for f0_mean.

        // Extract ssl_state from metadata if present
        let ssl_state = metadata.ssl_state.clone();

        // For V3 files, also check metadata for acting_latent
        let acting_latent = acting_latent.or_else(|| metadata.acting_latent.clone());

        Ok(Self {
            spk_embed,
            f0_mean,
            style_embed,
            reference_tokens,
            lora_delta,
            ssl_state,
            acting_latent,
            metadata,
            version,
        })
    }

    /// Save as .tmrvc_speaker file (V3 or V4 depending on content).
    pub fn save(&self, path: &Path) -> Result<()> {
        // Include ssl_state and acting_latent in metadata for serialization
        let mut metadata = self.metadata.clone();
        metadata.ssl_state = self.ssl_state.clone();
        metadata.acting_latent = self.acting_latent.clone();
        let metadata_json =
            serde_json::to_vec(&metadata).context("Failed to serialize metadata")?;
        let metadata_size = metadata_json.len();

        // Determine version: use V4 if acting_latent is present
        let use_v4 = self.acting_latent.is_some() || self.version == VERSION_V4;
        let version = if use_v4 { VERSION_V4 } else { VERSION_V3 };

        // Calculate flags and sizes
        let mut flags: u32 = 0;
        if self.style_embed.is_some() {
            flags |= FLAG_HAS_STYLE;
        }
        if self.reference_tokens.is_some() {
            flags |= FLAG_HAS_REF_TOKENS;
        }
        if self.lora_delta.is_some() {
            flags |= FLAG_HAS_LORA;
        }
        if self.acting_latent.is_some() {
            flags |= FLAG_HAS_ACTING_LATENT;
        }

        let style_embed_size = if self.style_embed.is_some() {
            D_STYLE as u32
        } else {
            0
        };
        let ref_tokens_frames = self
            .reference_tokens
            .as_ref()
            .map(|t| (t.len() / 4) as u32)
            .unwrap_or(0);
        let lora_size = self
            .lora_delta
            .as_ref()
            .map(|l| l.len() as u32)
            .unwrap_or(0);

        let acting_latent_size = self
            .acting_latent
            .as_ref()
            .map(|a| a.len() as u32)
            .unwrap_or(0);

        // Build payload
        let mut buf = Vec::new();

        // Header (32 bytes for V3, 40 bytes for V4)
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&(D_SPEAKER as u32).to_le_bytes());
        buf.extend_from_slice(&style_embed_size.to_le_bytes());
        buf.extend_from_slice(&ref_tokens_frames.to_le_bytes());
        buf.extend_from_slice(&lora_size.to_le_bytes());
        buf.extend_from_slice(&(metadata_size as u32).to_le_bytes());

        // V4 extended header field
        if use_v4 {
            buf.extend_from_slice(&acting_latent_size.to_le_bytes());
            // Pad to 40 bytes
            buf.extend_from_slice(&[0u8; 4]);
        }

        // spk_embed
        for &v in &self.spk_embed {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // f0_mean
        buf.extend_from_slice(&self.f0_mean.to_le_bytes());

        // style_embed
        if let Some(ref se) = self.style_embed {
            for &v in se {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // reference_tokens
        if let Some(ref rt) = self.reference_tokens {
            for &v in rt {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // lora_delta
        if let Some(ref ld) = self.lora_delta {
            for &v in ld {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // acting_latent (V4)
        if let Some(ref al) = self.acting_latent {
            for &v in al {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // metadata JSON
        buf.extend_from_slice(&metadata_json);

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

    fn default_metadata() -> SpeakerMetadata {
        SpeakerMetadata::default()
    }

    #[test]
    fn save_load_roundtrip_light() {
        let mut spk_embed = [0.0f32; D_SPEAKER];
        for (i, v) in spk_embed.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }

        let original = SpeakerFile {
            spk_embed,
            f0_mean: 220.0,
            style_embed: None,
            reference_tokens: None,
            lora_delta: None,
            ssl_state: None,
            acting_latent: None,
            metadata: default_metadata(),
            version: VERSION_V3,
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v3_light.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(original.spk_embed, loaded.spk_embed);
        assert!((loaded.f0_mean - 220.0).abs() < 0.1);
        assert!(loaded.style_embed.is_none());
        assert!(loaded.reference_tokens.is_none());
        assert!(loaded.lora_delta.is_none());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn save_load_roundtrip_standard() {
        let spk_embed = [0.5f32; D_SPEAKER];
        let style_embed = Some(vec![0.3f32; D_STYLE]);
        let reference_tokens = Some(vec![100i32, 200, 300, 400, 101, 201, 301, 401]); // 2 frames

        let original = SpeakerFile {
            spk_embed,
            f0_mean: 180.0,
            style_embed,
            reference_tokens,
            lora_delta: None,
            ssl_state: None,
            acting_latent: None,
            metadata: default_metadata(),
            version: VERSION_V3,
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v3_standard.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(original.spk_embed, loaded.spk_embed);
        assert!((loaded.f0_mean - 180.0).abs() < 0.1);
        assert_eq!(original.style_embed, loaded.style_embed);
        assert_eq!(original.reference_tokens, loaded.reference_tokens);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn save_load_roundtrip_full() {
        let spk_embed = [0.1f32; D_SPEAKER];
        let style_embed = Some(vec![0.2f32; D_STYLE]);
        let reference_tokens = Some(vec![1i32, 2, 3, 4]);
        let lora_delta = Some(vec![0.3f32; LORA_DELTA_SIZE]);

        let original = SpeakerFile {
            spk_embed,
            f0_mean: 250.0,
            style_embed,
            reference_tokens,
            lora_delta,
            ssl_state: None,
            acting_latent: None,
            metadata: default_metadata(),
            version: VERSION_V3,
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v3_full.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(loaded.style_embed.as_ref().map(|s| s.len()), Some(D_STYLE));
        assert_eq!(loaded.reference_tokens.as_ref().map(|r| r.len()), Some(4));
        assert_eq!(
            loaded.lora_delta.as_ref().map(|l| l.len()),
            Some(LORA_DELTA_SIZE)
        );
        assert!((loaded.f0_mean - 250.0).abs() < 0.1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_detects_corruption() {
        let spk = SpeakerFile {
            spk_embed: [1.0; D_SPEAKER],
            f0_mean: 220.0,
            style_embed: None,
            reference_tokens: None,
            lora_delta: None,
            ssl_state: None,
            acting_latent: None,
            metadata: default_metadata(),
            version: VERSION_V3,
        };
        let dir = std::env::temp_dir();
        let path = dir.join("test_v3_corrupt.tmrvc_speaker");
        spk.save(&path).expect("save failed");

        // Corrupt one byte
        let mut data = fs::read(&path).unwrap();
        data[HEADER_SIZE + 100] ^= 0xFF;
        fs::write(&path, &data).unwrap();

        assert!(SpeakerFile::load(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn lora_delta_or_zeros() {
        let sf = SpeakerFile {
            spk_embed: [0.0; D_SPEAKER],
            f0_mean: 220.0,
            style_embed: None,
            reference_tokens: None,
            lora_delta: None,
            ssl_state: None,
            acting_latent: None,
            metadata: default_metadata(),
            version: VERSION_V3,
        };
        let zeros = sf.lora_delta_or_zeros();
        assert_eq!(zeros.len(), LORA_DELTA_SIZE);
        assert!(zeros.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn save_load_roundtrip_with_ssl_state() {
        let spk_embed = [0.5f32; D_SPEAKER];
        let ssl_state = Some(vec![0.25f32; D_VOICE_STATE_SSL]);

        let original = SpeakerFile {
            spk_embed,
            f0_mean: 200.0,
            style_embed: None,
            reference_tokens: None,
            lora_delta: None,
            ssl_state,
            acting_latent: None,
            metadata: default_metadata(),
            version: VERSION_V3,
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v3_ssl_state.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(original.spk_embed, loaded.spk_embed);
        assert_eq!(
            loaded.ssl_state.as_ref().map(|s| s.len()),
            Some(D_VOICE_STATE_SSL)
        );
        assert!(loaded
            .ssl_state
            .as_ref()
            .map(|s| s.iter().all(|&v| (v - 0.25).abs() < 1e-5))
            .unwrap_or(false));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_python_created_speaker_file() {
        // This test expects a speaker file created by Python with ssl_state
        // Run: PYTHONPATH=tmrvc-export/src:tmrvc-core/src .venv/bin/python -c "
        //   from tmrvc_export.speaker_file import write_speaker_file
        //   import numpy as np
        //   write_speaker_file('/tmp/test_python_speaker.tmrvc_speaker',
        //       spk_embed=np.random.randn(192).astype(np.float32),
        //       ssl_state=np.random.randn(128).astype(np.float32),
        //       metadata={'speaker_name': 'python_test'})
        // "
        let path = std::path::Path::new("/tmp/test_python_speaker.tmrvc_speaker");
        if !path.exists() {
            eprintln!(
                "Skipping test: {} not found. Create it with Python first.",
                path.display()
            );
            return;
        }
        let loaded = SpeakerFile::load(path).expect("load failed");
        assert_eq!(loaded.spk_embed.len(), D_SPEAKER);
        assert!(loaded.ssl_state.is_some());
        assert_eq!(
            loaded.ssl_state.as_ref().map(|s| s.len()),
            Some(D_VOICE_STATE_SSL)
        );
        println!(
            "Loaded Python speaker file with ssl_state: {:?}",
            loaded.ssl_state.as_ref().map(|s| &s[..5])
        );
    }

    #[test]
    fn load_f0_mean_from_python_file() {
        // Create a speaker file with custom f0_mean using Python
        // PYTHONPATH=tmrvc-export/src:tmrvc-core/src .venv/bin/python -c "
        //   from tmrvc_export.speaker_file import write_speaker_file
        //   import numpy as np
        //   write_speaker_file('/tmp/test_f0.tmrvc_speaker',
        //       spk_embed=np.zeros(192, dtype=np.float32),
        //       f0_mean=360.5,
        //       metadata={'speaker_name': 'f0_test'})
        // "
        let path = std::path::Path::new("/tmp/test_f0.tmrvc_speaker");
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            return;
        }
        let loaded = SpeakerFile::load(path).expect("load failed");
        assert!(
            (loaded.f0_mean - 360.5).abs() < 0.1,
            "f0_mean mismatch: {}",
            loaded.f0_mean
        );
        println!("Loaded f0_mean: {}", loaded.f0_mean);
    }

    #[test]
    fn save_load_roundtrip_v4_with_acting_latent() {
        let spk_embed = [0.5f32; D_SPEAKER];
        let acting_latent = Some(vec![0.42f32; D_ACTING_LATENT]);

        let original = SpeakerFile {
            spk_embed,
            f0_mean: 200.0,
            style_embed: None,
            reference_tokens: None,
            lora_delta: None,
            ssl_state: Some(vec![0.1f32; D_VOICE_STATE_SSL]),
            acting_latent,
            metadata: default_metadata(),
            version: VERSION_V4,
        };

        let dir = std::env::temp_dir();
        let path = dir.join("test_v4_acting.tmrvc_speaker");
        original.save(&path).expect("save failed");

        let loaded = SpeakerFile::load(&path).expect("load failed");
        assert_eq!(loaded.version, VERSION_V4);
        assert_eq!(original.spk_embed, loaded.spk_embed);
        assert!((loaded.f0_mean - 200.0).abs() < 0.1);
        assert_eq!(
            loaded.acting_latent.as_ref().map(|a| a.len()),
            Some(D_ACTING_LATENT)
        );
        assert!(loaded
            .acting_latent
            .as_ref()
            .map(|a| a.iter().all(|&v| (v - 0.42).abs() < 1e-5))
            .unwrap_or(false));
        assert_eq!(
            loaded.ssl_state.as_ref().map(|s| s.len()),
            Some(D_VOICE_STATE_SSL)
        );

        let _ = fs::remove_file(&path);
    }
}
