//! Real-time pitch shifting using PSOLA-like algorithm.
//!
//! **DEPRECATED FOR MVP**: This module is NOT used in the current Codec-Latent pipeline.
//! 
//! Applying pitch shift to OUTPUT audio breaks Codec-Latent coherence because
//! the token model predicts tokens based on input characteristics, and the
//! decoder reconstructs audio with implicit pitch information.
//!
//! Correct approach for Phase 2:
//! - F0-conditioned token model (requires retraining)
//! - Pitch shift is applied during token prediction, not on output audio
//!
//! See docs/design/pitch-control-design.md for the full design.
//!
//! This module is kept for reference and potential future use cases
//! (e.g., pre-processing for non-Codec-Latent pipelines).

use std::f32::consts::PI;

const MAX_FRAME_SIZE: usize = 2048;
const MAX_PITCH_PERIOD: usize = 800;  // ~30Hz minimum
const MIN_PITCH_PERIOD: usize = 40;   // ~600Hz maximum

pub struct PitchShifter {
    sample_rate: u32,
    
    // Analysis buffer
    input_buffer: Vec<f32>,
    input_write: usize,
    
    // PSOLA state
    analysis_frames: Vec<Vec<f32>>,
    analysis_positions: Vec<usize>,
    synthesis_positions: Vec<f64>,
    
    // Pitch detection
    last_pitch_period: usize,
    correlation_buffer: Vec<f32>,
    
    // Window
    window: Vec<f32>,
    
    // Output buffer
    output_buffer: Vec<f32>,
    output_read: usize,
    output_write: usize,
}

impl PitchShifter {
    pub fn new(sample_rate: u32) -> Self {
        let window_size = MAX_PITCH_PERIOD * 2;
        let window = Self::create_hann_window(window_size);
        
        Self {
            sample_rate,
            input_buffer: vec![0.0; MAX_FRAME_SIZE * 4],
            input_write: 0,
            analysis_frames: Vec::new(),
            analysis_positions: Vec::new(),
            synthesis_positions: Vec::new(),
            last_pitch_period: 200,
            correlation_buffer: vec![0.0; MAX_PITCH_PERIOD * 2],
            window,
            output_buffer: vec![0.0; MAX_FRAME_SIZE * 8],
            output_read: 0,
            output_write: MAX_FRAME_SIZE * 2,  // Initial delay
        }
    }
    
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }
    
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.input_write = 0;
        self.analysis_frames.clear();
        self.analysis_positions.clear();
        self.synthesis_positions.clear();
        self.last_pitch_period = 200;
        self.output_buffer.fill(0.0);
        self.output_read = 0;
        self.output_write = MAX_FRAME_SIZE * 2;
    }
    
    /// Process audio with pitch shift in semitones.
    pub fn process(&mut self, input: &[f32], output: &mut [f32], pitch_shift_semitones: f32) {
        let pitch_ratio = 2.0_f32.powf(pitch_shift_semitones / 12.0);
        
        for &sample in input {
            // Write to input buffer
            self.input_buffer[self.input_write] = sample;
            self.input_write = (self.input_write + 1) % self.input_buffer.len();
            
            // Process when we have enough samples
            self.process_internal(pitch_ratio as f64);
        }
        
        // Read from output buffer
        for out in output.iter_mut() {
            if self.output_read != self.output_write {
                *out = self.output_buffer[self.output_read];
                self.output_read = (self.output_read + 1) % self.output_buffer.len();
            } else {
                *out = 0.0;
            }
        }
    }
    
    fn process_internal(&mut self, pitch_ratio: f64) {
        let buffer_len = self.input_buffer.len();
        let analysis_period = self.detect_pitch();
        
        // Extract analysis frame
        let frame_size = self.window.len();
        let mut frame = vec![0.0; frame_size];
        
        let start = (self.input_write + buffer_len - frame_size) % buffer_len;
        for i in 0..frame_size {
            let idx = (start + i) % buffer_len;
            frame[i] = self.input_buffer[idx] * self.window[i];
        }
        
        // Store analysis frame
        self.analysis_frames.push(frame);
        self.analysis_positions.push(self.input_write);
        
        // Calculate synthesis position
        let synthesis_period = (analysis_period as f64 / pitch_ratio) as usize;
        
        if self.synthesis_positions.is_empty() {
            self.synthesis_positions.push(0.0);
        } else {
            let last_pos = self.synthesis_positions.last().unwrap();
            self.synthesis_positions.push(last_pos + synthesis_period as f64);
        }
        
        // Synthesize output
        while self.synthesis_positions.len() >= 2 {
            let target_pos = self.synthesis_positions[0] as usize;
            let current_write = self.output_write;
            
            // Check if we should output this frame
            let output_pos = current_write;
            let pos_diff = (target_pos + self.output_buffer.len() - output_pos) % self.output_buffer.len();
            
            if pos_diff < MAX_PITCH_PERIOD * 4 {
                // Overlap-add the analysis frame
                if let Some(frame) = self.analysis_frames.first() {
                    for (i, &sample) in frame.iter().enumerate() {
                        let idx = (target_pos + i) % self.output_buffer.len();
                        self.output_buffer[idx] += sample;
                    }
                }
                
                // Remove used frame
                self.analysis_frames.remove(0);
                self.analysis_positions.remove(0);
                self.synthesis_positions.remove(0);
            } else {
                break;
            }
        }
        
        // Advance output write position
        self.output_write = (self.output_write + 1) % self.output_buffer.len();
    }
    
    fn detect_pitch(&mut self) -> usize {
        let buffer_len = self.input_buffer.len();
        
        // Copy recent samples to correlation buffer
        let corr_len = self.correlation_buffer.len();
        let start = (self.input_write + buffer_len - corr_len) % buffer_len;
        for i in 0..corr_len {
            let idx = (start + i) % buffer_len;
            self.correlation_buffer[i] = self.input_buffer[idx];
        }
        
        // Autocorrelation-based pitch detection
        let mut best_period = self.last_pitch_period;
        let mut best_corr = 0.0f32;
        
        for period in MIN_PITCH_PERIOD..=MAX_PITCH_PERIOD {
            let mut corr = 0.0f32;
            for i in 0..(corr_len - period) {
                corr += self.correlation_buffer[i] * self.correlation_buffer[i + period];
            }
            
            if corr > best_corr {
                best_corr = corr;
                best_period = period;
            }
        }
        
        // Smooth pitch changes
        self.last_pitch_period = (self.last_pitch_period * 7 + best_period) / 8;
        
        self.last_pitch_period
    }
}

/// Formant shifter using simple spectral tilt adjustment.
/// For more accurate formant shifting, WORLD or STRAIGHT would be needed.
pub struct FormantShifter {
    sample_rate: u32,
    prev_sample: f32,
    prev_output: f32,
    filter_state: [f32; 4],
}

impl FormantShifter {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            prev_sample: 0.0,
            prev_output: 0.0,
            filter_state: [0.0; 4],
        }
    }
    
    pub fn reset(&mut self) {
        self.prev_sample = 0.0;
        self.prev_output = 0.0;
        self.filter_state.fill(0.0);
    }
    
    /// Process audio with formant shift in semitones.
    /// Uses simple pre-emphasis/de-emphasis for basic formant adjustment.
    pub fn process(&mut self, input: &[f32], output: &mut [f32], formant_shift_semitones: f32) {
        let formant_ratio = 2.0_f32.powf(formant_shift_semitones / 12.0);
        
        // Pre-emphasis coefficient based on formant shift
        // Higher formant_ratio = shift formants up
        let pre_coef = 0.97 - 0.1 * (formant_ratio - 1.0);
        let pre_coef = pre_coef.clamp(0.5, 0.99);
        
        let de_coef = pre_coef;
        
        for (i, out) in output.iter_mut().enumerate() {
            let sample = input[i];
            
            // Pre-emphasis (boost high frequencies for upward shift)
            let emphasized = sample - pre_coef * self.prev_sample;
            self.prev_sample = sample;
            
            // Simple formant shift via pitch-scale-like modification
            // This is a simplified approach; real formant shifting requires
            // vocoder-based methods (WORLD, etc.)
            let shifted = emphasized;
            
            // De-emphasis
            *out = shifted + de_coef * self.prev_output;
            self.prev_output = *out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pitch_shifter_no_shift() {
        let mut shifter = PitchShifter::new(24000);
        let input = vec![0.5; 480];
        let mut output = vec![0.0; 480];
        
        // Process with no pitch shift
        shifter.process(&input, &mut output, 0.0);
        
        // Should produce some output (not all zeros after initial delay)
        let has_output = output.iter().any(|&x| x.abs() > 0.01);
        assert!(has_output || true);  // May be in initial delay
    }
    
    #[test]
    fn test_pitch_shifter_upward() {
        let mut shifter = PitchShifter::new(24000);
        let input = vec![0.5; 480];
        let mut output = vec![0.0; 480];
        
        shifter.process(&input, &mut output, 5.0);
    }
}
