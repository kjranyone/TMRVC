//! F0 (fundamental frequency) tracker for pitch conditioning.
//!
//! Provides lightweight streaming F0 estimation using autocorrelation
//! with octave error correction.
//! For more accurate tracking, CREPE-lite could be substituted.

const MIN_F0_HZ: f32 = 50.0;
const MAX_F0_HZ: f32 = 600.0;
const DEFAULT_F0_MEAN: f32 = 220.0;

pub struct F0Tracker {
    sample_rate: u32,
    f0_mean: f32,
    buffer: Vec<f32>,
    buffer_pos: usize,
    window_size: usize,
    last_f0: f32,
    pitch_shift: f32,
}

impl F0Tracker {
    pub fn new(sample_rate: u32, f0_mean: f32) -> Self {
        let window_size = (sample_rate as f32 / MIN_F0_HZ) as usize;
        Self {
            sample_rate,
            f0_mean: if f0_mean > 0.0 { f0_mean } else { DEFAULT_F0_MEAN },
            buffer: vec![0.0; window_size * 2],
            buffer_pos: 0,
            window_size,
            last_f0: f0_mean,
            pitch_shift: 0.0,
        }
    }

    pub fn set_f0_mean(&mut self, f0_mean: f32) {
        self.f0_mean = if f0_mean > 0.0 { f0_mean } else { DEFAULT_F0_MEAN };
    }

    pub fn set_pitch_shift(&mut self, semitones: f32) {
        self.pitch_shift = semitones;
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buffer_pos = 0;
        self.last_f0 = self.f0_mean;
    }

    pub fn process_frame(&mut self, frame: &[f32]) -> [f32; 2] {
        for &sample in frame {
            self.buffer[self.buffer_pos] = sample;
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer.len();
        }

        let f0 = self.detect_f0();
        let f0_norm = (f0 / self.f0_mean).log2();

        [f0_norm, self.pitch_shift]
    }

    fn detect_f0(&mut self) -> f32 {
        let min_period = (self.sample_rate as f32 / MAX_F0_HZ) as usize;
        let max_period = (self.sample_rate as f32 / MIN_F0_HZ) as usize;
        let max_period = max_period.min(self.window_size);

        let n = self.buffer.len() / 2;
        let start = self.buffer_pos;

        let mut best_period = min_period;
        let mut best_corr = 0.0f32;

        let last_period = (self.sample_rate as f32 / self.last_f0) as usize;

        for period in min_period..=max_period {
            let (corr, energy) = self.compute_correlation(start, n, period);

            if energy > 0.0 {
                let norm_corr = (corr / energy).min(1.0);
                
                let continuity_bonus = if last_period > 0 {
                    let period_diff = (period as i32 - last_period as i32).abs() as f32;
                    let max_diff = last_period as f32 * 0.2;
                    if period_diff <= max_diff {
                        0.02
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                
                let score = norm_corr + continuity_bonus;
                if score > best_corr {
                    best_corr = score;
                    best_period = period;
                }
            }
        }

        if best_corr < 0.3 {
            return self.last_f0;
        }

        best_period = self.octave_correct(best_period, best_corr, start, n);

        let f0 = self.sample_rate as f32 / best_period as f32;

        self.last_f0 = self.last_f0 * 0.7 + f0 * 0.3;
        self.last_f0
    }

    fn compute_correlation(&self, start: usize, n: usize, period: usize) -> (f32, f32) {
        let mut corr = 0.0f32;
        let mut energy = 0.0f32;

        for i in 0..n {
            let idx1 = (start + i) % self.buffer.len();
            let idx2 = (start + i + period) % self.buffer.len();
            corr += self.buffer[idx1] * self.buffer[idx2];
            energy += self.buffer[idx1] * self.buffer[idx1];
        }

        (corr, energy)
    }

    fn octave_correct(&self, period: usize, corr: f32, start: usize, n: usize) -> usize {
        let mut corrected_period = period;
        let mut best_score = corr;

        let min_period = (self.sample_rate as f32 / MAX_F0_HZ) as usize;

        let mut check_period = period / 2;
        while check_period >= min_period {
            let (check_corr, check_energy) = self.compute_correlation(start, n, check_period);
            if check_energy > 0.0 {
                let check_norm = check_corr / check_energy;
                if check_norm > best_score * 0.85 {
                    best_score = check_norm;
                    corrected_period = check_period;
                }
            }
            check_period /= 2;
        }

        corrected_period
    }

    pub fn f0_mean(&self) -> f32 {
        self.f0_mean
    }

    pub fn last_f0(&self) -> f32 {
        self.last_f0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sine(freq: f32, sample_rate: u32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_f0_detection() {
        let sample_rate = 24000;
        let mut tracker = F0Tracker::new(sample_rate, 220.0);

        let sine = generate_sine(220.0, sample_rate, 4800);
        for chunk in sine.chunks(480) {
            tracker.process_frame(chunk);
        }

        let detected = tracker.last_f0();
        assert!(detected > 200.0 && detected < 240.0, "Detected F0: {}", detected);
    }

    #[test]
    fn test_f0_normalization() {
        let mut tracker = F0Tracker::new(24000, 220.0);

        let sine = generate_sine(440.0, 24000, 9600);
        for chunk in sine.chunks(480) {
            tracker.process_frame(chunk);
        }

        let detected = tracker.last_f0();
        
        assert!(
            detected > 400.0 && detected < 480.0,
            "Detected F0 should be ~440Hz, got: {}",
            detected
        );

        let f0_norm = (detected / 220.0).log2();
        assert!(f0_norm > 0.8, "F0 normalized should be ~1.0 for octave up, got: {}", f0_norm);
    }

    #[test]
    fn test_pitch_shift() {
        let mut tracker = F0Tracker::new(24000, 220.0);
        tracker.set_pitch_shift(12.0);

        let sine = generate_sine(220.0, 24000, 480);
        let [f0_norm, pitch_shift] = tracker.process_frame(&sine);

        assert!((pitch_shift - 12.0).abs() < 0.01);
    }
}
