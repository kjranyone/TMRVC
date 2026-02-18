/// Ping-Pong double-buffered state for ONNX model hidden states.
///
/// Alternates between buffer A and buffer B each frame to avoid
/// in-place updates (state_in and state_out never alias).
pub struct PingPongState {
    buffer_a: Vec<f32>,
    buffer_b: Vec<f32>,
    shape: [usize; 3], // [batch, channels, context]
    current: bool,     // false = A is input, true = B is input
}

#[allow(dead_code)]
impl PingPongState {
    /// Create a new ping-pong state with the given shape, zero-initialized.
    pub fn new(shape: [usize; 3]) -> Self {
        let len = shape[0] * shape[1] * shape[2];
        Self {
            buffer_a: vec![0.0; len],
            buffer_b: vec![0.0; len],
            shape,
            current: false,
        }
    }

    /// Get a reference to the current input buffer.
    pub fn input(&self) -> &[f32] {
        if !self.current {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    /// Get a mutable reference to the current output buffer.
    pub fn output(&mut self) -> &mut [f32] {
        if !self.current {
            &mut self.buffer_b
        } else {
            &mut self.buffer_a
        }
    }

    /// Swap input and output buffers (call after each inference).
    pub fn swap(&mut self) {
        self.current = !self.current;
    }

    /// Reset both buffers to zero.
    pub fn reset(&mut self) {
        self.buffer_a.fill(0.0);
        self.buffer_b.fill(0.0);
        self.current = false;
    }

    /// Shape of the state tensor.
    pub fn shape(&self) -> &[usize; 3] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_pong_swap() {
        let mut pp = PingPongState::new([1, 4, 2]);
        assert_eq!(pp.len(), 8);

        // Initially A is input (zeros), B is output
        assert!(pp.input().iter().all(|&x| x == 0.0));

        // Write to output (B)
        pp.output()[0] = 1.0;
        pp.swap();

        // Now B is input, A is output
        assert_eq!(pp.input()[0], 1.0);
    }
}
