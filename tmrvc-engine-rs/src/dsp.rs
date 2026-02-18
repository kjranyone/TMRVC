use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::constants::*;

/// Initialize the mel filterbank matrix (N_MELS × N_FREQ_BINS).
///
/// HTK-style mel scale. Output is stored row-major:
/// filterbank[mel_bin * N_FREQ_BINS + freq_bin].
pub fn init_mel_filterbank(filterbank: &mut [f32]) {
    assert_eq!(filterbank.len(), N_MELS * N_FREQ_BINS);

    let fft_freqs: Vec<f32> = (0..N_FREQ_BINS)
        .map(|i| i as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
        .collect();

    // Mel-scale edges
    let mel_min = hz_to_mel(MEL_FMIN);
    let mel_max = hz_to_mel(MEL_FMAX);
    let mel_points: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32))
        .collect();

    filterbank.fill(0.0);

    for m in 0..N_MELS {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        for k in 0..N_FREQ_BINS {
            let freq = fft_freqs[k];
            let weight = if freq >= f_left && freq <= f_center {
                if (f_center - f_left).abs() < 1e-10 {
                    0.0
                } else {
                    (freq - f_left) / (f_center - f_left)
                }
            } else if freq > f_center && freq <= f_right {
                if (f_right - f_center).abs() < 1e-10 {
                    0.0
                } else {
                    (f_right - freq) / (f_right - f_center)
                }
            } else {
                0.0
            };
            filterbank[m * N_FREQ_BINS + k] = weight;
        }
    }
}

/// Compute causal STFT for a single frame.
///
/// - `context_buffer`: past + current samples [WINDOW_LENGTH]
/// - `hann_window`: pre-computed Hann window [WINDOW_LENGTH]
/// - `windowed`: scratch for windowed signal [WINDOW_LENGTH]
/// - `padded`: scratch for zero-padded signal [N_FFT]
/// - `fft_real_out`, `fft_imag_out`: output real/imag parts [N_FREQ_BINS each]
/// - `planner`: rustfft planner (reuses internal scratch)
pub fn causal_stft(
    context_buffer: &[f32],
    hann_window: &[f32],
    windowed: &mut [f32],
    padded: &mut [f32],
    fft_real_out: &mut [f32],
    fft_imag_out: &mut [f32],
    planner: &mut FftPlanner<f32>,
) {
    // 1. Apply window
    for i in 0..WINDOW_LENGTH {
        windowed[i] = context_buffer[i] * hann_window[i];
    }

    // 2. Zero-pad to N_FFT
    padded[..WINDOW_LENGTH].copy_from_slice(&windowed[..WINDOW_LENGTH]);
    padded[WINDOW_LENGTH..N_FFT].fill(0.0);

    // 3. FFT (in-place, complex)
    let fft = planner.plan_fft_forward(N_FFT);
    let mut complex_buf: Vec<Complex<f32>> = padded.iter().map(|&r| Complex::new(r, 0.0)).collect();
    fft.process(&mut complex_buf);

    // 4. Extract first N_FREQ_BINS (positive frequencies)
    for i in 0..N_FREQ_BINS {
        fft_real_out[i] = complex_buf[i].re;
        fft_imag_out[i] = complex_buf[i].im;
    }
}

/// Compute log-mel spectrogram from STFT real/imag parts.
///
/// power = real² + imag²
/// mel = filterbank @ power
/// log_mel = log(max(mel, LOG_FLOOR))
pub fn compute_log_mel(
    fft_real: &[f32],
    fft_imag: &[f32],
    filterbank: &[f32],
    mel_out: &mut [f32],
) {
    for m in 0..N_MELS {
        let mut sum = 0.0f32;
        let row = &filterbank[m * N_FREQ_BINS..(m + 1) * N_FREQ_BINS];
        for k in 0..N_FREQ_BINS {
            let power = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
            sum += row[k] * power;
        }
        mel_out[m] = sum.max(LOG_FLOOR).ln();
    }
}

/// Inverse STFT: mag[N_FREQ_BINS] + phase[N_FREQ_BINS] → time-domain signal [N_FFT].
///
/// Returns the windowed time-domain signal in `time_out[..WINDOW_LENGTH]`.
pub fn istft(
    mag: &[f32],
    phase: &[f32],
    hann_window: &[f32],
    time_out: &mut [f32],
    planner: &mut FftPlanner<f32>,
) {
    // 1. Construct full complex spectrum (Hermitian symmetry)
    let mut complex_buf = vec![Complex::new(0.0f32, 0.0); N_FFT];
    for i in 0..N_FREQ_BINS {
        let (sin, cos) = phase[i].sin_cos();
        complex_buf[i] = Complex::new(mag[i] * cos, mag[i] * sin);
    }
    // Mirror for negative frequencies (conjugate symmetry)
    for i in 1..N_FFT - N_FREQ_BINS + 1 {
        complex_buf[N_FFT - i] = complex_buf[i].conj();
    }

    // 2. IFFT
    let ifft = planner.plan_fft_inverse(N_FFT);
    ifft.process(&mut complex_buf);

    // 3. Normalize and window
    let scale = 1.0 / N_FFT as f32;
    for i in 0..WINDOW_LENGTH {
        time_out[i] = complex_buf[i].re * scale * hann_window[i];
    }
}

/// Overlap-Add: accumulate windowed output into OLA buffer,
/// then extract HOP_LENGTH samples as the current frame output.
///
/// - `windowed_output`: windowed iSTFT output [WINDOW_LENGTH]
/// - `ola_buffer`: running OLA accumulator [WINDOW_LENGTH]
/// - `hop_output`: output samples for this frame [HOP_LENGTH]
pub fn overlap_add(windowed_output: &[f32], ola_buffer: &mut [f32], hop_output: &mut [f32]) {
    // 1. Extract the first HOP_LENGTH samples as output
    hop_output.copy_from_slice(&ola_buffer[..HOP_LENGTH]);

    // 2. Shift OLA buffer left by HOP_LENGTH
    ola_buffer.copy_within(HOP_LENGTH..WINDOW_LENGTH, 0);
    ola_buffer[WINDOW_LENGTH - HOP_LENGTH..WINDOW_LENGTH].fill(0.0);

    // 3. Add windowed output
    for i in 0..WINDOW_LENGTH {
        ola_buffer[i] += windowed_output[i];
    }
}

/// Update the context buffer with new hop samples.
///
/// Shifts left by HOP_LENGTH, appends new hop data at the end.
pub fn update_context_buffer(context: &mut [f32], hop_input: &[f32]) {
    context.copy_within(HOP_LENGTH..WINDOW_LENGTH, 0);
    context[PAST_CONTEXT..WINDOW_LENGTH].copy_from_slice(hop_input);
}

// --- Mel scale conversion (HTK) ---

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank_shape() {
        let mut fb = vec![0.0f32; N_MELS * N_FREQ_BINS];
        init_mel_filterbank(&mut fb);
        // Each mel bin should have some non-zero weights
        for m in 0..N_MELS {
            let row = &fb[m * N_FREQ_BINS..(m + 1) * N_FREQ_BINS];
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "mel bin {} has no non-zero weights", m);
        }
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz2 = mel_to_hz(mel);
        assert!((hz - hz2).abs() < 0.01);
    }

    #[test]
    fn test_overlap_add_silence() {
        let windowed = [0.0f32; WINDOW_LENGTH];
        let mut ola = [0.0f32; WINDOW_LENGTH];
        let mut hop_out = [0.0f32; HOP_LENGTH];
        overlap_add(&windowed, &mut ola, &mut hop_out);
        assert!(hop_out.iter().all(|&x| x == 0.0));
    }
}
