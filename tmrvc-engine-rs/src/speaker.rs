#![allow(dead_code)]

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};

use crate::constants::*;

const MAGIC: &[u8; 4] = b"TMSP";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 16; // 4 (magic) + 4 (version) + 4 (embed_size) + 4 (lora_size)
const CHECKSUM_SIZE: usize = 32; // SHA-256

/// Parsed .tmrvc_speaker file.
pub struct SpeakerFile {
    pub spk_embed: [f32; D_SPEAKER],
    pub lora_delta: Vec<f32>,
}

impl SpeakerFile {
    /// Load and validate a .tmrvc_speaker file.
    pub fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).context("Failed to read speaker file")?;

        let expected_size = HEADER_SIZE + D_SPEAKER * 4 + LORA_DELTA_SIZE * 4 + CHECKSUM_SIZE;
        if data.len() != expected_size {
            bail!(
                "Speaker file size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            );
        }

        // Magic
        if &data[0..4] != MAGIC {
            bail!("Invalid magic: expected TMSP");
        }

        // Version
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            bail!("Unsupported version: {}", version);
        }

        // Embed size
        let embed_size = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        if embed_size != D_SPEAKER {
            bail!(
                "Speaker embed size mismatch: expected {}, got {}",
                D_SPEAKER,
                embed_size
            );
        }

        // Lora size
        let lora_size = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        if lora_size != LORA_DELTA_SIZE {
            bail!(
                "LoRA delta size mismatch: expected {}, got {}",
                LORA_DELTA_SIZE,
                lora_size
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
            spk_embed[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        // Parse lora_delta
        let lora_offset = HEADER_SIZE + D_SPEAKER * 4;
        let lora_bytes = &data[lora_offset..lora_offset + LORA_DELTA_SIZE * 4];
        let mut lora_delta = vec![0.0f32; LORA_DELTA_SIZE];
        for (i, chunk) in lora_bytes.chunks_exact(4).enumerate() {
            lora_delta[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        Ok(Self {
            spk_embed,
            lora_delta,
        })
    }
}
