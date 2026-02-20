use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::constants::*;

/// Initialize the mel filterbank matrix (N_MELS x N_FREQ_BINS).
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
/// - `complex_scratch`: pre-allocated complex buffer [N_FFT]
/// - `planner`: rustfft planner (reuses internal scratch)
pub fn causal_stft(
    context_buffer: &[f32],
    hann_window: &[f32],
    windowed: &mut [f32],
    padded: &mut [f32],
    fft_real_out: &mut [f32],
    fft_imag_out: &mut [f32],
    complex_scratch: &mut [Complex<f32>],
    planner: &mut FftPlanner<f32>,
) {
    // 1. Apply window
    for i in 0..WINDOW_LENGTH {
        windowed[i] = context_buffer[i] * hann_window[i];
    }

    // 2. Zero-pad to N_FFT
    padded[..WINDOW_LENGTH].copy_from_slice(&windowed[..WINDOW_LENGTH]);
    padded[WINDOW_LENGTH..N_FFT].fill(0.0);

    // 3. FFT (in-place, complex) — uses pre-allocated buffer
    let fft = planner.plan_fft_forward(N_FFT);
    for (c, &r) in complex_scratch.iter_mut().zip(padded.iter()) {
        *c = Complex::new(r, 0.0);
    }
    fft.process(complex_scratch);

    // 4. Extract first N_FREQ_BINS (positive frequencies)
    for i in 0..N_FREQ_BINS {
        fft_real_out[i] = complex_scratch[i].re;
        fft_imag_out[i] = complex_scratch[i].im;
    }
}

/// Compute log-mel spectrogram from STFT real/imag parts.
///
/// power = real^2 + imag^2
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

/// Compute log-mel spectrogram for an entire waveform (offline).
///
/// Uses the same STFT parameters as the streaming pipeline:
/// N_FFT=1024, HOP=240, WIN=960, causal padding = PAST_CONTEXT (720 zeros).
///
/// Returns `(mel_data, num_frames)` where `mel_data` is `[N_MELS, num_frames]` row-major.
pub fn compute_mel_offline(waveform: &[f32], filterbank: &[f32]) -> (Vec<f32>, usize) {
    assert_eq!(filterbank.len(), N_MELS * N_FREQ_BINS);

    // Causal padding: prepend PAST_CONTEXT zeros
    let padded_len = waveform.len() + PAST_CONTEXT;
    let num_frames = if padded_len >= N_FFT {
        (padded_len - N_FFT) / HOP_LENGTH + 1
    } else {
        0
    };

    if num_frames == 0 {
        return (vec![], 0);
    }

    // Build padded waveform
    let mut padded = vec![0.0f32; padded_len];
    padded[PAST_CONTEXT..].copy_from_slice(waveform);

    // Pre-compute Hann window
    let hann: Vec<f32> = (0..WINDOW_LENGTH)
        .map(|i| {
            let x = std::f32::consts::PI * 2.0 * i as f32 / WINDOW_LENGTH as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect();

    let mut planner = FftPlanner::new();
    let mut windowed = vec![0.0f32; WINDOW_LENGTH];
    let mut fft_padded = vec![0.0f32; N_FFT];
    let mut fft_real = vec![0.0f32; N_FREQ_BINS];
    let mut fft_imag = vec![0.0f32; N_FREQ_BINS];
    let mut complex_scratch = vec![Complex::new(0.0f32, 0.0); N_FFT];
    let mut mel_frame = vec![0.0f32; N_MELS];

    // Output: [N_MELS, num_frames] row-major
    let mut mel_data = vec![0.0f32; N_MELS * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        let context = &padded[start..start + WINDOW_LENGTH];

        causal_stft(
            context,
            &hann,
            &mut windowed,
            &mut fft_padded,
            &mut fft_real,
            &mut fft_imag,
            &mut complex_scratch,
            &mut planner,
        );

        compute_log_mel(&fft_real, &fft_imag, filterbank, &mut mel_frame);

        // Store in [N_MELS, num_frames] row-major order
        for m in 0..N_MELS {
            mel_data[m * num_frames + frame_idx] = mel_frame[m];
        }
    }

    (mel_data, num_frames)
}

/// Inverse STFT: mag[N_FREQ_BINS] + phase[N_FREQ_BINS] -> time-domain signal [N_FFT].
///
/// Returns the windowed time-domain signal in `time_out[..WINDOW_LENGTH]`.
/// `complex_scratch`: pre-allocated complex buffer [N_FFT].
pub fn istft(
    mag: &[f32],
    phase: &[f32],
    hann_window: &[f32],
    time_out: &mut [f32],
    complex_scratch: &mut [Complex<f32>],
    planner: &mut FftPlanner<f32>,
) {
    // 1. Construct full complex spectrum (Hermitian symmetry)
    for c in complex_scratch.iter_mut() {
        *c = Complex::new(0.0, 0.0);
    }
    for i in 0..N_FREQ_BINS {
        let (sin, cos) = phase[i].sin_cos();
        complex_scratch[i] = Complex::new(mag[i] * cos, mag[i] * sin);
    }
    // Mirror for negative frequencies (conjugate symmetry)
    for i in 1..N_FFT - N_FREQ_BINS + 1 {
        complex_scratch[N_FFT - i] = complex_scratch[i].conj();
    }

    // 2. IFFT
    let ifft = planner.plan_fft_inverse(N_FFT);
    ifft.process(complex_scratch);

    // 3. Normalize and window
    let scale = 1.0 / N_FFT as f32;
    for i in 0..WINDOW_LENGTH {
        time_out[i] = complex_scratch[i].re * scale * hann_window[i];
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

/// Estimate per-frame log-F0 using normalized autocorrelation.
///
/// Returns ``log(f0_hz + 1.0)`` for voiced frames, ``0.0`` for unvoiced.
/// The search range is 60-500 Hz.
pub fn estimate_log_f0_autocorr(context: &[f32]) -> f32 {
    assert_eq!(context.len(), WINDOW_LENGTH);

    // Remove DC bias.
    let mean = context.iter().sum::<f32>() / WINDOW_LENGTH as f32;

    // Energy gate for silence.
    let mut energy = 0.0f32;
    for &x in context {
        let v = x - mean;
        energy += v * v;
    }
    if energy < 1e-4 {
        return 0.0;
    }

    let min_hz = 60.0f32;
    let max_hz = 500.0f32;
    let min_lag = (SAMPLE_RATE as f32 / max_hz).floor() as usize;
    let max_lag = (SAMPLE_RATE as f32 / min_hz).ceil() as usize;

    if max_lag >= WINDOW_LENGTH || min_lag == 0 || min_lag > max_lag {
        return 0.0;
    }

    let mut best_corr = 0.0f32;
    let mut best_lag = 0usize;

    for lag in min_lag..=max_lag {
        let mut num = 0.0f32;
        let mut den_a = 0.0f32;
        let mut den_b = 0.0f32;

        for i in lag..WINDOW_LENGTH {
            let a = context[i] - mean;
            let b = context[i - lag] - mean;
            num += a * b;
            den_a += a * a;
            den_b += b * b;
        }

        let denom = (den_a * den_b).sqrt();
        if denom <= 1e-8 {
            continue;
        }

        let corr = num / denom;
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    // Voicing threshold.
    if best_corr < 0.30 || best_lag == 0 {
        return 0.0;
    }

    let f0_hz = SAMPLE_RATE as f32 / best_lag as f32;
    (f0_hz + 1.0).ln()
}

/// Lightweight articulation proxy from one mel frame.
///
/// Larger values indicate stronger high-frequency emphasis.
pub fn mel_articulation_proxy(mel_frame: &[f32]) -> f32 {
    assert_eq!(mel_frame.len(), N_MELS);

    let lo_end = N_MELS / 4;
    let hi_start = N_MELS - lo_end;

    let low_mean = if lo_end > 0 {
        mel_frame[..lo_end].iter().sum::<f32>() / lo_end as f32
    } else {
        0.0
    };
    let high_mean = if hi_start < N_MELS {
        mel_frame[hi_start..].iter().sum::<f32>() / (N_MELS - hi_start) as f32
    } else {
        0.0
    };

    high_mean - low_mean
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
    fn test_mel_offline_basic() {
        let mut fb = vec![0.0f32; N_MELS * N_FREQ_BINS];
        init_mel_filterbank(&mut fb);

        // 1 second of 1kHz sine at 24kHz
        let n = SAMPLE_RATE;
        let waveform: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        let (mel_data, num_frames) = compute_mel_offline(&waveform, &fb);

        // Expected: (24000 + 720 - 1024) / 240 + 1 = 99 frames
        let padded_len = n + PAST_CONTEXT;
        let expected_frames = (padded_len - N_FFT) / HOP_LENGTH + 1;
        assert_eq!(num_frames, expected_frames);
        assert_eq!(mel_data.len(), N_MELS * num_frames);

        // All mel values should be finite
        assert!(mel_data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mel_offline_empty() {
        let mut fb = vec![0.0f32; N_MELS * N_FREQ_BINS];
        init_mel_filterbank(&mut fb);

        // Too short for even one frame
        let waveform = vec![0.0f32; 100];
        let (mel_data, num_frames) = compute_mel_offline(&waveform, &fb);
        assert_eq!(num_frames, 0);
        assert!(mel_data.is_empty());
    }

    #[test]
    fn test_mel_offline_matches_streaming() {
        // Verify that offline mel matches frame-by-frame streaming computation
        let mut fb = vec![0.0f32; N_MELS * N_FREQ_BINS];
        init_mel_filterbank(&mut fb);

        let n = HOP_LENGTH * 10; // 10 frames of audio
        let waveform: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        // Offline
        let (mel_offline, num_frames) = compute_mel_offline(&waveform, &fb);

        // Streaming: process frame by frame with causal padding
        let hann: Vec<f32> = (0..WINDOW_LENGTH)
            .map(|i| {
                let x = std::f32::consts::PI * 2.0 * i as f32 / WINDOW_LENGTH as f32;
                0.5 * (1.0 - x.cos())
            })
            .collect();

        let mut context = vec![0.0f32; WINDOW_LENGTH];
        let mut planner = rustfft::FftPlanner::new();
        let mut windowed = vec![0.0f32; WINDOW_LENGTH];
        let mut padded = vec![0.0f32; N_FFT];
        let mut fft_real = vec![0.0f32; N_FREQ_BINS];
        let mut fft_imag = vec![0.0f32; N_FREQ_BINS];
        let mut complex_scratch = vec![Complex::new(0.0f32, 0.0); N_FFT];
        let mut mel_frame = vec![0.0f32; N_MELS];

        for frame_idx in 0..num_frames {
            let hop_start = frame_idx * HOP_LENGTH;
            let hop_end = (hop_start + HOP_LENGTH).min(waveform.len());
            let mut hop = [0.0f32; HOP_LENGTH];
            let copy_len = hop_end.saturating_sub(hop_start);
            if copy_len > 0 {
                hop[..copy_len].copy_from_slice(&waveform[hop_start..hop_start + copy_len]);
            }

            update_context_buffer(&mut context, &hop);
            causal_stft(
                &context,
                &hann,
                &mut windowed,
                &mut padded,
                &mut fft_real,
                &mut fft_imag,
                &mut complex_scratch,
                &mut planner,
            );
            compute_log_mel(&fft_real, &fft_imag, &fb, &mut mel_frame);

            for m in 0..N_MELS {
                let offline_val = mel_offline[m * num_frames + frame_idx];
                let stream_val = mel_frame[m];
                assert!(
                    (offline_val - stream_val).abs() < 1e-4,
                    "mismatch at mel={}, frame={}: offline={}, stream={}",
                    m,
                    frame_idx,
                    offline_val,
                    stream_val
                );
            }
        }
    }

    #[test]
    fn test_overlap_add_silence() {
        let windowed = [0.0f32; WINDOW_LENGTH];
        let mut ola = [0.0f32; WINDOW_LENGTH];
        let mut hop_out = [0.0f32; HOP_LENGTH];
        overlap_add(&windowed, &mut ola, &mut hop_out);
        assert!(hop_out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_estimate_log_f0_autocorr_sine() {
        let freq = 220.0f32;
        let context: Vec<f32> = (0..WINDOW_LENGTH)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();

        let log_f0 = estimate_log_f0_autocorr(&context);
        assert!(log_f0 > 0.0);

        let f0 = log_f0.exp() - 1.0;
        assert!(
            (f0 - freq).abs() < 20.0,
            "estimated f0={}, expected {}",
            f0,
            freq
        );
    }

    #[test]
    fn test_estimate_log_f0_autocorr_silence() {
        let context = vec![0.0f32; WINDOW_LENGTH];
        let log_f0 = estimate_log_f0_autocorr(&context);
        assert_eq!(log_f0, 0.0);
    }

    #[test]
    fn test_mel_articulation_proxy() {
        let mut mel = vec![0.0f32; N_MELS];
        // Low bins weak, high bins strong -> positive proxy.
        for v in &mut mel[..N_MELS / 4] {
            *v = -4.0;
        }
        for v in &mut mel[N_MELS - N_MELS / 4..] {
            *v = 2.0;
        }
        assert!(mel_articulation_proxy(&mel) > 0.0);
    }
}
