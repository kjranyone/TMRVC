use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

/// Top-level structure of a .nam JSON file.
pub(crate) struct NamFile {
    pub architecture: String,
    pub config: Value,
    pub weights: Vec<f64>,
    pub sample_rate: u32,
}

/// Parsed WaveNet configuration.
pub(crate) struct WaveNetConfig {
    pub condition_size: usize,
    pub channels: usize,
    pub head_size: usize,
    pub kernel_size: usize,
    pub dilations: Vec<usize>,
    pub gated: bool,
    pub head_bias: bool,
    pub num_blocks: usize,
}

/// Parsed LSTM configuration.
pub(crate) struct LstmConfig {
    pub num_layers: usize,
    pub input_size: usize,
    pub hidden_size: usize,
}

/// Intermediate serde struct for the top-level JSON.
#[derive(Deserialize)]
struct RawNamFile {
    version: Option<String>,
    architecture: String,
    config: Value,
    weights: Vec<f64>,
    sample_rate: Option<f64>,
    metadata: Option<RawMetadata>,
}

#[derive(Deserialize)]
struct RawMetadata {
    sample_rate: Option<f64>,
}

/// Parse a .nam JSON string into a `NamFile`.
pub(crate) fn parse_nam_json(json_str: &str) -> Result<NamFile> {
    let raw: RawNamFile = serde_json::from_str(json_str).context("Failed to parse .nam JSON")?;

    // Version check: accept 0.5.x or missing version
    if let Some(ref ver) = raw.version {
        if !ver.starts_with("0.5") && !ver.starts_with("0.6") && !ver.starts_with("0.7") {
            log::warn!(
                "NAM version '{}' may not be fully supported (expected 0.5.x-0.7.x)",
                ver
            );
        }
    }

    // Sample rate: top-level > metadata > default 48000
    let sample_rate = raw
        .sample_rate
        .or_else(|| raw.metadata.as_ref().and_then(|m| m.sample_rate))
        .unwrap_or(48000.0) as u32;

    if raw.weights.is_empty() {
        bail!("NAM file contains no weights");
    }

    Ok(NamFile {
        architecture: raw.architecture,
        config: raw.config,
        weights: raw.weights,
        sample_rate,
    })
}

/// Parse WaveNet-specific config from the `config` JSON value.
pub(crate) fn parse_wavenet_config(config: &Value) -> Result<WaveNetConfig> {
    let condition_size = config
        .get("condition_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    let channels = config
        .get("channels")
        .and_then(|v| v.as_u64())
        .context("WaveNet config missing 'channels'")? as usize;

    let head_size = config
        .get("head_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(8) as usize;

    let kernel_size = config
        .get("kernel_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as usize;

    let dilations: Vec<usize> = config
        .get("dilations")
        .and_then(|v| v.as_array())
        .context("WaveNet config missing 'dilations'")?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();

    if dilations.is_empty() {
        bail!("WaveNet config has empty 'dilations' array");
    }

    let gated = config
        .get("gated")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let head_bias = config
        .get("head_bias")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let num_blocks = config
        .get("num_blocks")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    if num_blocks == 0 {
        bail!("WaveNet config has num_blocks=0");
    }

    Ok(WaveNetConfig {
        condition_size,
        channels,
        head_size,
        kernel_size,
        dilations,
        gated,
        head_bias,
        num_blocks,
    })
}

/// Parse LSTM-specific config from the `config` JSON value.
pub(crate) fn parse_lstm_config(config: &Value) -> Result<LstmConfig> {
    let num_layers = config
        .get("num_layers")
        .and_then(|v| v.as_u64())
        .context("LSTM config missing 'num_layers'")? as usize;

    let input_size = config
        .get("input_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;

    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .context("LSTM config missing 'hidden_size'")? as usize;

    if num_layers == 0 || hidden_size == 0 {
        bail!("LSTM config has zero num_layers or hidden_size");
    }

    Ok(LstmConfig {
        num_layers,
        input_size,
        hidden_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wavenet_json() {
        let json = r#"{
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {
                "channels": 8,
                "head_size": 4,
                "kernel_size": 3,
                "dilations": [1, 2, 4],
                "gated": true,
                "head_bias": true
            },
            "weights": [1.0, 2.0, 3.0],
            "sample_rate": 48000.0
        }"#;
        let nam = parse_nam_json(json).unwrap();
        assert_eq!(nam.architecture, "WaveNet");
        assert_eq!(nam.sample_rate, 48000);
        assert_eq!(nam.weights.len(), 3);

        let config = parse_wavenet_config(&nam.config).unwrap();
        assert_eq!(config.channels, 8);
        assert_eq!(config.dilations, vec![1, 2, 4]);
        assert!(config.gated);
        assert_eq!(config.num_blocks, 1);
    }

    #[test]
    fn test_parse_lstm_json() {
        let json = r#"{
            "version": "0.5.1",
            "architecture": "LSTM",
            "config": {
                "num_layers": 3,
                "input_size": 1,
                "hidden_size": 32
            },
            "weights": [0.5, -0.5],
            "sample_rate": 48000.0
        }"#;
        let nam = parse_nam_json(json).unwrap();
        assert_eq!(nam.architecture, "LSTM");
        let config = parse_lstm_config(&nam.config).unwrap();
        assert_eq!(config.num_layers, 3);
        assert_eq!(config.hidden_size, 32);
    }

    #[test]
    fn test_default_sample_rate() {
        let json = r#"{
            "architecture": "LSTM",
            "config": {"num_layers": 1, "hidden_size": 8},
            "weights": [1.0]
        }"#;
        let nam = parse_nam_json(json).unwrap();
        assert_eq!(nam.sample_rate, 48000);
    }

    #[test]
    fn test_metadata_sample_rate() {
        let json = r#"{
            "architecture": "LSTM",
            "config": {"num_layers": 1, "hidden_size": 8},
            "weights": [1.0],
            "metadata": {"sample_rate": 44100.0}
        }"#;
        let nam = parse_nam_json(json).unwrap();
        assert_eq!(nam.sample_rate, 44100);
    }

    #[test]
    fn test_empty_weights_error() {
        let json = r#"{
            "architecture": "WaveNet",
            "config": {"channels": 8, "dilations": [1]},
            "weights": []
        }"#;
        assert!(parse_nam_json(json).is_err());
    }

    #[test]
    fn test_unsupported_version_warns() {
        let json = r#"{
            "version": "0.3.0",
            "architecture": "LSTM",
            "config": {"num_layers": 1, "hidden_size": 8},
            "weights": [1.0]
        }"#;
        // Should parse without error (just warns)
        assert!(parse_nam_json(json).is_ok());
    }
}
