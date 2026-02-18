#![allow(dead_code)]

use crate::constants::*;

/// TensorPool: single contiguous allocation for all intermediate tensors.
///
/// Mirrors the C++ TensorPool design. All sub-regions are accessed via
/// offset-based typed accessors. No dynamic allocation during processing.
pub struct TensorPool {
    data: Vec<f32>,
}

// Layout offsets (in f32 elements)
const OFF_MEL_FRAME: usize = 0; // [80]
const OFF_F0_FRAME: usize = OFF_MEL_FRAME + N_MELS; // [1]
const OFF_CONTENT: usize = OFF_F0_FRAME + 1; // [256]
const OFF_IR_PARAMS: usize = OFF_CONTENT + D_CONTENT; // [24]
const OFF_SPK_EMBED: usize = OFF_IR_PARAMS + N_IR_PARAMS; // [192]
const OFF_PRED_FEATURES: usize = OFF_SPK_EMBED + D_SPEAKER; // [513]
const OFF_STFT_MAG: usize = OFF_PRED_FEATURES + N_FREQ_BINS; // [513]
const OFF_STFT_PHASE: usize = OFF_STFT_MAG + N_FREQ_BINS; // [513]
const OFF_MEL_CHUNK: usize = OFF_STFT_PHASE + N_FREQ_BINS; // [80 * 10 = 800]
const OFF_FFT_REAL: usize = OFF_MEL_CHUNK + N_MELS * IR_UPDATE_INTERVAL; // [1024]
const OFF_FFT_IMAG: usize = OFF_FFT_REAL + N_FFT; // [1024]
const OFF_OLA_BUFFER: usize = OFF_FFT_IMAG + N_FFT; // [960]
const OFF_HANN_WINDOW: usize = OFF_OLA_BUFFER + WINDOW_LENGTH; // [960]
const OFF_CONTEXT_BUFFER: usize = OFF_HANN_WINDOW + WINDOW_LENGTH; // [960]
const OFF_WINDOWED: usize = OFF_CONTEXT_BUFFER + WINDOW_LENGTH; // [960]
const OFF_PADDED: usize = OFF_WINDOWED + WINDOW_LENGTH; // [1024]
const OFF_MEL_FILTERBANK: usize = OFF_PADDED + N_FFT; // [80 * 513 = 41040]
const TOTAL_SIZE: usize = OFF_MEL_FILTERBANK + N_MELS * N_FREQ_BINS;

impl TensorPool {
    /// Allocate and initialize the tensor pool.
    pub fn new() -> Self {
        let mut pool = Self {
            data: vec![0.0; TOTAL_SIZE],
        };
        pool.init_hann_window();
        pool
    }

    /// Total number of floats allocated.
    pub fn total_floats(&self) -> usize {
        TOTAL_SIZE
    }

    // --- Typed sub-slice accessors ---

    pub fn mel_frame(&self) -> &[f32] {
        &self.data[OFF_MEL_FRAME..OFF_MEL_FRAME + N_MELS]
    }
    pub fn mel_frame_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_MEL_FRAME..OFF_MEL_FRAME + N_MELS]
    }

    pub fn f0_frame(&self) -> &[f32] {
        &self.data[OFF_F0_FRAME..OFF_F0_FRAME + 1]
    }
    pub fn f0_frame_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_F0_FRAME..OFF_F0_FRAME + 1]
    }

    pub fn content(&self) -> &[f32] {
        &self.data[OFF_CONTENT..OFF_CONTENT + D_CONTENT]
    }
    pub fn content_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_CONTENT..OFF_CONTENT + D_CONTENT]
    }

    pub fn ir_params(&self) -> &[f32] {
        &self.data[OFF_IR_PARAMS..OFF_IR_PARAMS + N_IR_PARAMS]
    }
    pub fn ir_params_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_IR_PARAMS..OFF_IR_PARAMS + N_IR_PARAMS]
    }

    pub fn spk_embed(&self) -> &[f32] {
        &self.data[OFF_SPK_EMBED..OFF_SPK_EMBED + D_SPEAKER]
    }
    pub fn spk_embed_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_SPK_EMBED..OFF_SPK_EMBED + D_SPEAKER]
    }

    pub fn pred_features(&self) -> &[f32] {
        &self.data[OFF_PRED_FEATURES..OFF_PRED_FEATURES + N_FREQ_BINS]
    }
    pub fn pred_features_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_PRED_FEATURES..OFF_PRED_FEATURES + N_FREQ_BINS]
    }

    pub fn stft_mag(&self) -> &[f32] {
        &self.data[OFF_STFT_MAG..OFF_STFT_MAG + N_FREQ_BINS]
    }
    pub fn stft_mag_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_STFT_MAG..OFF_STFT_MAG + N_FREQ_BINS]
    }

    pub fn stft_phase(&self) -> &[f32] {
        &self.data[OFF_STFT_PHASE..OFF_STFT_PHASE + N_FREQ_BINS]
    }
    pub fn stft_phase_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_STFT_PHASE..OFF_STFT_PHASE + N_FREQ_BINS]
    }

    pub fn mel_chunk(&self) -> &[f32] {
        &self.data[OFF_MEL_CHUNK..OFF_MEL_CHUNK + N_MELS * IR_UPDATE_INTERVAL]
    }
    pub fn mel_chunk_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_MEL_CHUNK..OFF_MEL_CHUNK + N_MELS * IR_UPDATE_INTERVAL]
    }

    pub fn fft_real(&self) -> &[f32] {
        &self.data[OFF_FFT_REAL..OFF_FFT_REAL + N_FFT]
    }
    pub fn fft_real_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_FFT_REAL..OFF_FFT_REAL + N_FFT]
    }

    pub fn fft_imag(&self) -> &[f32] {
        &self.data[OFF_FFT_IMAG..OFF_FFT_IMAG + N_FFT]
    }
    pub fn fft_imag_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_FFT_IMAG..OFF_FFT_IMAG + N_FFT]
    }

    pub fn ola_buffer(&self) -> &[f32] {
        &self.data[OFF_OLA_BUFFER..OFF_OLA_BUFFER + WINDOW_LENGTH]
    }
    pub fn ola_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_OLA_BUFFER..OFF_OLA_BUFFER + WINDOW_LENGTH]
    }

    pub fn hann_window(&self) -> &[f32] {
        &self.data[OFF_HANN_WINDOW..OFF_HANN_WINDOW + WINDOW_LENGTH]
    }

    pub fn context_buffer(&self) -> &[f32] {
        &self.data[OFF_CONTEXT_BUFFER..OFF_CONTEXT_BUFFER + WINDOW_LENGTH]
    }
    pub fn context_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_CONTEXT_BUFFER..OFF_CONTEXT_BUFFER + WINDOW_LENGTH]
    }

    pub fn windowed(&self) -> &[f32] {
        &self.data[OFF_WINDOWED..OFF_WINDOWED + WINDOW_LENGTH]
    }
    pub fn windowed_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_WINDOWED..OFF_WINDOWED + WINDOW_LENGTH]
    }

    pub fn padded(&self) -> &[f32] {
        &self.data[OFF_PADDED..OFF_PADDED + N_FFT]
    }
    pub fn padded_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_PADDED..OFF_PADDED + N_FFT]
    }

    pub fn mel_filterbank(&self) -> &[f32] {
        &self.data[OFF_MEL_FILTERBANK..OFF_MEL_FILTERBANK + N_MELS * N_FREQ_BINS]
    }
    pub fn mel_filterbank_mut(&mut self) -> &mut [f32] {
        &mut self.data[OFF_MEL_FILTERBANK..OFF_MEL_FILTERBANK + N_MELS * N_FREQ_BINS]
    }

    /// Reset all buffers to zero (except hann_window and mel_filterbank).
    pub fn reset(&mut self) {
        self.data[OFF_MEL_FRAME..OFF_HANN_WINDOW].fill(0.0);
        self.data[OFF_CONTEXT_BUFFER..OFF_MEL_FILTERBANK].fill(0.0);
    }

    /// Pre-compute the Hann window (periodic, length = WINDOW_LENGTH).
    fn init_hann_window(&mut self) {
        let window = &mut self.data[OFF_HANN_WINDOW..OFF_HANN_WINDOW + WINDOW_LENGTH];
        for i in 0..WINDOW_LENGTH {
            let t = 2.0 * std::f32::consts::PI * i as f32 / WINDOW_LENGTH as f32;
            window[i] = 0.5 * (1.0 - t.cos());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_size() {
        let pool = TensorPool::new();
        // Verify total is reasonable (~50k floats = ~200 KB)
        assert!(pool.total_floats() > 40000);
        assert!(pool.total_floats() < 100000);
    }

    #[test]
    fn test_hann_window() {
        let pool = TensorPool::new();
        let w = pool.hann_window();
        assert_eq!(w.len(), WINDOW_LENGTH);
        // Hann window endpoints should be near 0
        assert!(w[0] < 0.001);
        // Midpoint should be near 1
        assert!((w[WINDOW_LENGTH / 2] - 1.0).abs() < 0.01);
    }
}
