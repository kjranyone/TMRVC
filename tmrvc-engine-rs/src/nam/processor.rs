use std::path::Path;

use anyhow::Result;

use crate::constants::MAX_DAW_BLOCK_SIZE;
use crate::resampler::PolyphaseResampler;

use super::NamModel;

/// NAM processing chain: model + optional resampling + wet/dry mix.
///
/// Designed to be inserted as a post-effect after voice conversion,
/// processing audio at the DAW sample rate.
///
/// `process()` is RT-safe (no allocation or I/O).
/// `load_profile()` is NOT RT-safe (reads file, allocates model).
pub struct NamChain {
    model: Option<Box<dyn NamModel>>,
    resampler_to_nam: Option<PolyphaseResampler>,
    resampler_from_nam: Option<PolyphaseResampler>,
    daw_rate: u32,
    // Pre-allocated scratch buffers for resampling path
    nam_in_buf: Vec<f32>,
    nam_out_buf: Vec<f32>,
    resample_out_buf: Vec<f32>,
    dry_buf: Vec<f32>,
    enabled: bool,
    mix: f32,
}

impl NamChain {
    /// Create a new NamChain for the given DAW sample rate.
    pub fn new(daw_rate: u32) -> Self {
        Self {
            model: None,
            resampler_to_nam: None,
            resampler_from_nam: None,
            daw_rate,
            nam_in_buf: vec![0.0; MAX_DAW_BLOCK_SIZE * 2],
            nam_out_buf: vec![0.0; MAX_DAW_BLOCK_SIZE * 2],
            resample_out_buf: vec![0.0; MAX_DAW_BLOCK_SIZE * 2],
            dry_buf: vec![0.0; MAX_DAW_BLOCK_SIZE],
            enabled: false,
            mix: 1.0,
        }
    }

    /// Load a NAM profile from a `.nam` file.
    ///
    /// **Not RT-safe** — call from a non-audio thread.
    /// Replaces any previously loaded model.
    pub fn load_profile(&mut self, path: &Path) -> Result<()> {
        let model = super::load_nam_file(path)?;
        let nam_rate = model.expected_sample_rate();

        // Set up resamplers if DAW rate differs from NAM model rate
        if self.daw_rate != nam_rate {
            self.resampler_to_nam = Some(PolyphaseResampler::new(self.daw_rate, nam_rate));
            self.resampler_from_nam = Some(PolyphaseResampler::new(nam_rate, self.daw_rate));
            // Pre-allocate scratch buffers for a reasonable max block size
            let max_block = 4096;
            let max_resampled =
                PolyphaseResampler::max_output_len(max_block, self.daw_rate, nam_rate);
            self.nam_in_buf = vec![0.0; max_resampled + 64];
            self.nam_out_buf = vec![0.0; max_resampled + 64];
            let max_back =
                PolyphaseResampler::max_output_len(max_resampled, nam_rate, self.daw_rate);
            self.resample_out_buf = vec![0.0; max_back + 64];
        } else {
            self.resampler_to_nam = None;
            self.resampler_from_nam = None;
        }

        self.model = Some(model);
        self.enabled = true;
        log::info!("NAM profile loaded from {:?} (rate={})", path, nam_rate);
        Ok(())
    }

    /// Unload the current NAM profile.
    pub fn unload(&mut self) {
        self.model = None;
        self.resampler_to_nam = None;
        self.resampler_from_nam = None;
        self.enabled = false;
    }

    /// Process audio samples in-place at the DAW sample rate.
    ///
    /// **RT-safe** — no allocation or I/O.
    /// If no model is loaded or NAM is disabled, this is a no-op.
    /// If samples.len() exceeds MAX_DAW_BLOCK_SIZE, processing is skipped.
    pub fn process(&mut self, samples: &mut [f32]) {
        if !self.enabled {
            return;
        }
        let model = match self.model.as_mut() {
            Some(m) => m,
            None => return,
        };

        if let (Some(rs_to), Some(rs_from)) =
            (&mut self.resampler_to_nam, &mut self.resampler_from_nam)
        {
            // Resampling path
            let n_resampled = rs_to.process(samples, &mut self.nam_in_buf);

            if n_resampled > self.nam_out_buf.len() {
                debug_assert!(
                    false,
                    "n_resampled = {} exceeds nam_out_buf capacity",
                    n_resampled
                );
                return;
            }

            model.process(
                &self.nam_in_buf[..n_resampled],
                &mut self.nam_out_buf[..n_resampled],
            );

            let n_back =
                rs_from.process(&self.nam_out_buf[..n_resampled], &mut self.resample_out_buf);

            let n_copy = samples.len().min(n_back).min(self.resample_out_buf.len());
            if (self.mix - 1.0).abs() > 1e-6 {
                let m = self.mix;
                for i in 0..n_copy {
                    samples[i] = samples[i] * (1.0 - m) + self.resample_out_buf[i] * m;
                }
            } else {
                samples[..n_copy].copy_from_slice(&self.resample_out_buf[..n_copy]);
            }
        } else {
            // No resampling: process in-place at DAW rate
            if samples.len() > self.dry_buf.len() || samples.len() > self.nam_out_buf.len() {
                debug_assert!(
                    false,
                    "samples.len() = {} exceeds pre-allocated buffer",
                    samples.len()
                );
                return;
            }

            if (self.mix - 1.0).abs() > 1e-6 {
                self.dry_buf[..samples.len()].copy_from_slice(samples);
            }

            model.process(samples, &mut self.nam_out_buf[..samples.len()]);

            if (self.mix - 1.0).abs() > 1e-6 {
                let m = self.mix;
                for i in 0..samples.len() {
                    samples[i] = self.dry_buf[i] * (1.0 - m) + self.nam_out_buf[i] * m;
                }
            } else {
                samples.copy_from_slice(&self.nam_out_buf[..samples.len()]);
            }
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_mix(&mut self, mix: f32) {
        self.mix = mix.clamp(0.0, 1.0);
    }

    /// Returns true if a NAM model is loaded and enabled.
    pub fn is_active(&self) -> bool {
        self.enabled && self.model.is_some()
    }

    /// Reset all internal state (model + resamplers).
    pub fn reset(&mut self) {
        if let Some(ref mut model) = self.model {
            model.reset();
        }
        if let Some(ref mut r) = self.resampler_to_nam {
            r.reset();
        }
        if let Some(ref mut r) = self.resampler_from_nam {
            r.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nam_chain_no_model_passthrough() {
        let mut chain = NamChain::new(48000);
        let mut samples = vec![1.0, 2.0, 3.0, 4.0];
        chain.process(&mut samples);
        // No model loaded → samples unchanged
        assert_eq!(samples, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_nam_chain_disabled_passthrough() {
        let mut chain = NamChain::new(48000);
        chain.set_enabled(false);
        let mut samples = vec![1.0, 2.0, 3.0];
        chain.process(&mut samples);
        assert_eq!(samples, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_nam_chain_mix_clamp() {
        let mut chain = NamChain::new(48000);
        chain.set_mix(1.5);
        assert!((chain.mix - 1.0).abs() < 1e-6);
        chain.set_mix(-0.5);
        assert!((chain.mix - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_nam_chain_reset_no_panic() {
        let mut chain = NamChain::new(48000);
        chain.reset(); // Should not panic even without model
    }
}
