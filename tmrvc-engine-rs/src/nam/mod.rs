mod lstm;
mod parse;
mod processor;
mod wavenet;

pub use processor::NamChain;

use std::path::Path;

use anyhow::Result;

/// Trait for NAM (Neural Amp Modeler) model inference.
///
/// Implementations process audio sample-by-sample with internal state.
/// All methods are RT-safe after construction (no allocation/IO).
pub trait NamModel: Send {
    /// Process a block of audio samples.
    /// `input` and `output` must have the same length.
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// Reset all internal state (ring buffers, hidden states, etc.).
    fn reset(&mut self);

    /// The sample rate this model expects (typically 48000).
    fn expected_sample_rate(&self) -> u32;
}

/// Load a `.nam` profile from disk and return a boxed model ready for inference.
///
/// Supports WaveNet, LSTM, and CatLSTM architectures.
/// Returns an error for unsupported architectures (ConvNet, Linear) or
/// malformed files.
pub fn load_nam_file(path: &Path) -> Result<Box<dyn NamModel>> {
    let raw = std::fs::read_to_string(path)?;
    let nam_file = parse::parse_nam_json(&raw)?;
    let sample_rate = nam_file.sample_rate;
    let weights: Vec<f32> = nam_file.weights.iter().map(|&w| w as f32).collect();

    match nam_file.architecture.as_str() {
        "WaveNet" => {
            let config = parse::parse_wavenet_config(&nam_file.config)?;
            let model = wavenet::WaveNet::from_weights(&config, &weights, sample_rate)?;
            Ok(Box::new(model))
        }
        "LSTM" => {
            let config = parse::parse_lstm_config(&nam_file.config)?;
            let model = lstm::Lstm::from_weights(&config, &weights, sample_rate, false)?;
            Ok(Box::new(model))
        }
        "CatLSTM" => {
            let config = parse::parse_lstm_config(&nam_file.config)?;
            let model = lstm::Lstm::from_weights(&config, &weights, sample_rate, true)?;
            Ok(Box::new(model))
        }
        arch => anyhow::bail!(
            "Unsupported NAM architecture: '{}'. Only WaveNet, LSTM, and CatLSTM are supported.",
            arch
        ),
    }
}
