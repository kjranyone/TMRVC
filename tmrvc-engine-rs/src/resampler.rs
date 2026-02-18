use std::f32::consts::PI;

/// Polyphase resampler for rational sample rate conversion.
///
/// Supports arbitrary rational ratios (e.g. 48kHz↔24kHz, 44.1kHz↔24kHz).
/// Uses Kaiser-windowed sinc FIR filter decomposed into polyphase branches.
/// `process()` performs no heap allocation.
pub struct PolyphaseResampler {
    /// Polyphase filter coefficients: [num_phases][taps_per_phase]
    phases: Vec<Vec<f32>>,
    /// FIR delay line (circular buffer)
    history: Vec<f32>,
    /// Write position into the delay line
    hist_pos: usize,
    /// Interpolation factor (L)
    up_factor: usize,
    /// Decimation factor (M)
    down_factor: usize,
    /// Fractional phase accumulator (tracks which polyphase branch to use)
    phase_acc: usize,
}

impl PolyphaseResampler {
    /// Create a new resampler for converting from `src_rate` to `dst_rate`.
    ///
    /// The filter is designed with a Kaiser window (beta=5) and
    /// cutoff at `min(π/L, π/M)` to prevent aliasing.
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        let g = gcd(src_rate, dst_rate);
        let up = (dst_rate / g) as usize; // L
        let down = (src_rate / g) as usize; // M

        // Filter design parameters
        let taps_per_phase = 16;
        let total_taps = taps_per_phase * up;
        let cutoff = PI / (up.max(down) as f32);
        let beta = 5.0;

        // Generate windowed sinc prototype filter
        let mut prototype = vec![0.0f32; total_taps];
        let center = (total_taps - 1) as f32 / 2.0;
        for i in 0..total_taps {
            let x = i as f32 - center;
            let sinc = if x.abs() < 1e-6 {
                1.0
            } else {
                (cutoff * x).sin() / (PI * x) * up as f32
            };
            let window = kaiser_window(i, total_taps, beta);
            prototype[i] = sinc * window;
        }

        // Decompose into polyphase branches
        // Phase p uses coefficients at indices p, p+L, p+2L, ...
        let mut phases = vec![vec![0.0f32; taps_per_phase]; up];
        for p in 0..up {
            for t in 0..taps_per_phase {
                let idx = p + t * up;
                if idx < total_taps {
                    phases[p][t] = prototype[idx];
                }
            }
        }

        Self {
            phases,
            history: vec![0.0; taps_per_phase],
            hist_pos: 0,
            up_factor: up,
            down_factor: down,
            phase_acc: 0,
        }
    }

    /// Process input samples and write resampled output.
    ///
    /// Returns the number of output samples written.
    /// `output` must have at least `max_output_len(input.len(), src, dst)` capacity.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let taps = self.phases[0].len();
        let mut out_idx = 0;
        let mut in_idx = 0;

        while in_idx < input.len() {
            // Push input sample into delay line
            self.history[self.hist_pos] = input[in_idx];
            self.hist_pos = (self.hist_pos + 1) % taps;
            in_idx += 1;

            // Generate output samples for this input
            // We produce an output whenever phase_acc < up_factor
            // then advance by down_factor
            loop {
                if self.phase_acc >= self.up_factor {
                    self.phase_acc -= self.up_factor;
                    break;
                }

                // Compute polyphase filter output for current phase
                let coeffs = &self.phases[self.phase_acc];
                let mut sum = 0.0f32;
                for t in 0..taps {
                    // Read from delay line in reverse order
                    let idx = (self.hist_pos + taps - 1 - t) % taps;
                    sum += self.history[idx] * coeffs[t];
                }

                if out_idx < output.len() {
                    output[out_idx] = sum;
                }
                out_idx += 1;
                self.phase_acc += self.down_factor;
            }
        }

        out_idx
    }

    /// Calculate the maximum number of output samples for a given input length.
    pub fn max_output_len(input_len: usize, src_rate: u32, dst_rate: u32) -> usize {
        // Ceiling division: (input_len * dst_rate + src_rate - 1) / src_rate
        let num = input_len as u64 * dst_rate as u64 + src_rate as u64 - 1;
        (num / src_rate as u64) as usize + 1
    }

    /// Reset the internal state (delay line and phase accumulator).
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.hist_pos = 0;
        self.phase_acc = 0;
    }
}

/// Greatest common divisor (Euclidean algorithm).
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Modified zeroth-order Bessel function of the first kind (I0).
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let x2 = x * x * 0.25;
    for k in 1..20 {
        term *= x2 / (k * k) as f32;
        sum += term;
        if term < 1e-10 * sum {
            break;
        }
    }
    sum
}

/// Kaiser window function.
fn kaiser_window(n: usize, length: usize, beta: f32) -> f32 {
    let center = (length - 1) as f32 / 2.0;
    let x = (n as f32 - center) / center;
    bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48000, 24000), 24000);
        assert_eq!(gcd(44100, 24000), 300);
        assert_eq!(gcd(24000, 48000), 24000);
    }

    #[test]
    fn test_48k_to_24k_downsample() {
        // 48kHz → 24kHz is 2:1 decimation (L=1, M=2)
        let mut resampler = PolyphaseResampler::new(48000, 24000);

        // Generate a 1kHz sine at 48kHz, 480 samples (10ms)
        let n_in = 480;
        let input: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();

        let max_out = PolyphaseResampler::max_output_len(n_in, 48000, 24000);
        let mut output = vec![0.0f32; max_out];
        let n_out = resampler.process(&input, &mut output);

        // Should produce ~240 samples (half)
        assert!(
            (n_out as i32 - 240).abs() <= 1,
            "Expected ~240 output samples, got {}",
            n_out
        );
    }

    #[test]
    fn test_24k_to_48k_upsample() {
        // 24kHz → 48kHz is 2:1 interpolation (L=2, M=1)
        let mut resampler = PolyphaseResampler::new(24000, 48000);

        let n_in = 240;
        let input: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 24000.0).sin())
            .collect();

        let max_out = PolyphaseResampler::max_output_len(n_in, 24000, 48000);
        let mut output = vec![0.0f32; max_out];
        let n_out = resampler.process(&input, &mut output);

        // Should produce ~480 samples (double)
        assert!(
            (n_out as i32 - 480).abs() <= 1,
            "Expected ~480 output samples, got {}",
            n_out
        );
    }

    #[test]
    fn test_44100_to_24000() {
        // 44.1kHz → 24kHz (L=80, M=147)
        let mut resampler = PolyphaseResampler::new(44100, 24000);

        let n_in = 441; // 10ms at 44.1kHz
        let input: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 44100.0).sin())
            .collect();

        let max_out = PolyphaseResampler::max_output_len(n_in, 44100, 24000);
        let mut output = vec![0.0f32; max_out];
        let n_out = resampler.process(&input, &mut output);

        // Expected: 441 * 24000/44100 = 240
        assert!(
            (n_out as i32 - 240).abs() <= 2,
            "Expected ~240 output samples, got {}",
            n_out
        );
    }

    #[test]
    fn test_roundtrip_energy_preservation() {
        // Down-sample then up-sample and check energy is approximately preserved
        let mut down = PolyphaseResampler::new(48000, 24000);
        let mut up = PolyphaseResampler::new(24000, 48000);

        // Input: 1kHz sine at 48kHz, long enough to get past filter transient
        let n_in = 4800; // 100ms
        let input: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();

        let input_energy: f32 = input.iter().map(|&x| x * x).sum::<f32>() / input.len() as f32;

        // Down-sample
        let max_mid = PolyphaseResampler::max_output_len(n_in, 48000, 24000);
        let mut mid = vec![0.0f32; max_mid];
        let n_mid = down.process(&input, &mut mid);

        // Up-sample back
        let max_out = PolyphaseResampler::max_output_len(n_mid, 24000, 48000);
        let mut output = vec![0.0f32; max_out];
        let n_out = up.process(&mid[..n_mid], &mut output);

        // Skip transient (first 200 samples) and compare energy
        let skip = 200;
        let len = n_out.min(n_in) - skip;
        let output_energy: f32 =
            output[skip..skip + len].iter().map(|&x| x * x).sum::<f32>() / len as f32;

        let energy_ratio = output_energy / input_energy;
        assert!(
            energy_ratio > 0.7 && energy_ratio < 1.3,
            "Energy ratio {:.3} is outside acceptable range [0.7, 1.3]",
            energy_ratio
        );
    }

    #[test]
    fn test_identity_same_rate() {
        // Same rate should pass through (approximately)
        let mut resampler = PolyphaseResampler::new(24000, 24000);

        let n_in = 480;
        let input: Vec<f32> = (0..n_in)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 24000.0).sin())
            .collect();

        let max_out = PolyphaseResampler::max_output_len(n_in, 24000, 24000);
        let mut output = vec![0.0f32; max_out];
        let n_out = resampler.process(&input, &mut output);

        assert_eq!(n_out, n_in);
    }
}
