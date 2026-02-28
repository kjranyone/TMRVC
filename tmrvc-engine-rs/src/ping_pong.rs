/// Ping-Pong double-buffered state for ONNX model hidden states.
///
/// Alternates between buffer A and buffer B each frame to avoid
/// in-place updates (state_in and state_out never alias).
pub struct PingPongState {
    buffer_a: Vec<f32>,
    buffer_b: Vec<f32>,
    current: bool,
}

impl PingPongState {
    pub fn new(size: usize) -> Self {
        Self {
            buffer_a: vec![0.0; size],
            buffer_b: vec![0.0; size],
            current: false,
        }
    }

    pub fn current(&self) -> &[f32] {
        if !self.current {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    pub fn next(&mut self) -> &mut [f32] {
        if !self.current {
            &mut self.buffer_b
        } else {
            &mut self.buffer_a
        }
    }

    pub fn swap(&mut self) {
        self.current = !self.current;
    }

    pub fn reset(&mut self) {
        self.buffer_a.fill(0.0);
        self.buffer_b.fill(0.0);
        self.current = false;
    }

    pub fn size(&self) -> usize {
        self.buffer_a.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_pong_swap() {
        let mut pp = PingPongState::new(8);
        assert_eq!(pp.size(), 8);

        assert!(pp.current().iter().all(|&x| x == 0.0));

        pp.next()[0] = 1.0;
        pp.swap();

        assert_eq!(pp.current()[0], 1.0);
    }
}
