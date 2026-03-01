//! RT-safe token context buffers for Disentangled UCLM.
//!
//! Maintains rolling context for:
//! - ctx_A: Acoustic stream `[L, 8]` — past A_t tokens
//! - ctx_B: Control stream `[L, 4]` — past B_t tokens
//!
//! All buffers are pre-allocated at construction time.
//! NO heap allocations during `push` or `get_context` operations.

use crate::constants::*;

/// RT-safe ring buffer for acoustic tokens A_t.
///
/// Stores `CONTEXT_FRAMES` frames of `[N_CODEBOOKS]` tokens each.
/// Layout: `[CONTEXT_FRAMES][N_CODEBOOKS]` in row-major order.
pub struct AcousticTokenBuffer {
    buffer: [[i64; N_CODEBOOKS]; CONTEXT_FRAMES],
    write_pos: usize,
    count: usize,
}

impl AcousticTokenBuffer {
    pub fn new() -> Self {
        Self {
            buffer: [[0i64; N_CODEBOOKS]; CONTEXT_FRAMES],
            write_pos: 0,
            count: 0,
        }
    }

    /// Push one frame of acoustic tokens.
    /// RT-safe: no allocation.
    #[inline]
    pub fn push(&mut self, tokens: &[i64; N_CODEBOOKS]) {
        self.buffer[self.write_pos].copy_from_slice(tokens);
        self.write_pos = (self.write_pos + 1) % CONTEXT_FRAMES;
        if self.count < CONTEXT_FRAMES {
            self.count += 1;
        }
    }

    /// Check if buffer has enough context.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= CONTEXT_FRAMES
    }

    /// Get current count of stored frames.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Extract context in `[N_CODEBOOKS, CONTEXT_FRAMES]` layout for ONNX.
    /// This matches the ONNX input shape `[1, N_CODEBOOKS, L]`.
    /// RT-safe: writes to pre-allocated output buffer.
    #[inline]
    pub fn get_context(&self, output: &mut [i64]) {
        debug_assert_eq!(output.len(), N_CODEBOOKS * CONTEXT_FRAMES);

        for cb in 0..N_CODEBOOKS {
            for frame in 0..CONTEXT_FRAMES {
                let read_pos = (self.write_pos + frame) % CONTEXT_FRAMES;
                let src_val = self.buffer[read_pos][cb];
                let dst_idx = cb * CONTEXT_FRAMES + frame;
                output[dst_idx] = src_val;
            }
        }
    }

    /// Reset buffer to initial state.
    pub fn reset(&mut self) {
        for row in &mut self.buffer {
            *row = [0i64; N_CODEBOOKS];
        }
        self.write_pos = 0;
        self.count = 0;
    }
}

impl Default for AcousticTokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// RT-safe ring buffer for control tokens B_t.
///
/// Stores `CONTEXT_FRAMES` frames of `[CONTROL_SLOTS]` tokens each.
/// Layout: `[CONTEXT_FRAMES][CONTROL_SLOTS]` in row-major order.
pub struct ControlTokenBuffer {
    buffer: [[i64; CONTROL_SLOTS]; CONTEXT_FRAMES],
    write_pos: usize,
    count: usize,
}

impl ControlTokenBuffer {
    pub fn new() -> Self {
        Self {
            buffer: [[0i64; CONTROL_SLOTS]; CONTEXT_FRAMES],
            write_pos: 0,
            count: 0,
        }
    }

    /// Push one frame of control tokens.
    /// RT-safe: no allocation.
    #[inline]
    pub fn push(&mut self, tokens: &[i64; CONTROL_SLOTS]) {
        self.buffer[self.write_pos].copy_from_slice(tokens);
        self.write_pos = (self.write_pos + 1) % CONTEXT_FRAMES;
        if self.count < CONTEXT_FRAMES {
            self.count += 1;
        }
    }

    /// Check if buffer has enough context.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= CONTEXT_FRAMES
    }

    /// Get current count of stored frames.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Extract context in `[CONTROL_SLOTS, CONTEXT_FRAMES]` layout for ONNX.
    /// This matches the ONNX input shape `[1, CONTROL_SLOTS, L]`.
    /// RT-safe: writes to pre-allocated output buffer.
    #[inline]
    pub fn get_context(&self, output: &mut [i64]) {
        debug_assert_eq!(output.len(), CONTROL_SLOTS * CONTEXT_FRAMES);

        for slot in 0..CONTROL_SLOTS {
            for frame in 0..CONTEXT_FRAMES {
                let read_pos = (self.write_pos + frame) % CONTEXT_FRAMES;
                let src_val = self.buffer[read_pos][slot];
                let dst_idx = slot * CONTEXT_FRAMES + frame;
                output[dst_idx] = src_val;
            }
        }
    }

    /// Reset buffer to initial state.
    pub fn reset(&mut self) {
        for row in &mut self.buffer {
            *row = [0i64; CONTROL_SLOTS];
        }
        self.write_pos = 0;
        self.count = 0;
    }
}

impl Default for ControlTokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined token context for UCLM streaming inference.
///
/// Pre-allocates all necessary buffers at construction time.
/// RT-safe: all operations are allocation-free.
pub struct TokenContext {
    pub ctx_a: AcousticTokenBuffer,
    pub ctx_b: ControlTokenBuffer,

    ctx_a_output: [i64; N_CODEBOOKS * CONTEXT_FRAMES],
    ctx_b_output: [i64; CONTROL_SLOTS * CONTEXT_FRAMES],
}

impl TokenContext {
    pub fn new() -> Self {
        Self {
            ctx_a: AcousticTokenBuffer::new(),
            ctx_b: ControlTokenBuffer::new(),
            ctx_a_output: [0i64; N_CODEBOOKS * CONTEXT_FRAMES],
            ctx_b_output: [0i64; CONTROL_SLOTS * CONTEXT_FRAMES],
        }
    }

    /// Push new tokens for current frame (both A and B).
    #[inline]
    pub fn push(&mut self, a_t: &[i64; N_CODEBOOKS], b_t: &[i64; CONTROL_SLOTS]) {
        self.ctx_a.push(a_t);
        self.ctx_b.push(b_t);
    }

    /// Push source acoustic token A_src to ctx_A.
    #[inline]
    pub fn push_a(&mut self, a_t: &[i64; N_CODEBOOKS]) {
        self.ctx_a.push(a_t);
    }

    /// Push target control token B_t to ctx_B.
    #[inline]
    pub fn push_b(&mut self, b_t: &[i64; CONTROL_SLOTS]) {
        self.ctx_b.push(b_t);
    }

    /// Check if ctx_A has enough context.
    #[inline]
    pub fn is_ctx_a_ready(&self) -> bool {
        self.ctx_a.is_full()
    }

    /// Check if both buffers have enough context.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ctx_a.is_full() && self.ctx_b.is_full()
    }

    /// Get context A for ONNX input.
    /// Returns slice of pre-allocated buffer `[N_CODEBOOKS * CONTEXT_FRAMES]`.
    #[inline]
    pub fn get_ctx_a(&mut self) -> &[i64] {
        self.ctx_a.get_context(&mut self.ctx_a_output);
        &self.ctx_a_output
    }

    /// Get context B for ONNX input.
    /// Returns slice of pre-allocated buffer `[CONTROL_SLOTS * CONTEXT_FRAMES]`.
    #[inline]
    pub fn get_ctx_b(&mut self) -> &[i64] {
        self.ctx_b.get_context(&mut self.ctx_b_output);
        &self.ctx_b_output
    }

    /// Get single-frame context A for initial frames (before buffer is full).
    /// Returns slice to a temporary single-frame buffer.
    pub fn get_ctx_a_single(&mut self, tokens: &[i64; N_CODEBOOKS]) -> &[i64] {
        self.ctx_a_output[..N_CODEBOOKS].copy_from_slice(tokens);
        &self.ctx_a_output[..N_CODEBOOKS]
    }

    /// Reset all buffers.
    pub fn reset(&mut self) {
        self.ctx_a.reset();
        self.ctx_b.reset();
        self.ctx_a_output = [0i64; N_CODEBOOKS * CONTEXT_FRAMES];
        self.ctx_b_output = [0i64; CONTROL_SLOTS * CONTEXT_FRAMES];
    }
}

impl Default for TokenContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acoustic_token_buffer() {
        let mut buf = AcousticTokenBuffer::new();

        assert!(!buf.is_full());
        assert_eq!(buf.count(), 0);

        for i in 0..CONTEXT_FRAMES {
            buf.push(&[i as i64; N_CODEBOOKS]);
        }

        assert!(buf.is_full());
        assert_eq!(buf.count(), CONTEXT_FRAMES);

        let mut output = [0i64; N_CODEBOOKS * CONTEXT_FRAMES];
        buf.get_context(&mut output);

        for cb in 0..N_CODEBOOKS {
            for frame in 0..CONTEXT_FRAMES {
                let idx = cb * CONTEXT_FRAMES + frame;
                assert_eq!(output[idx], frame as i64);
            }
        }
    }

    #[test]
    fn test_control_token_buffer() {
        let mut buf = ControlTokenBuffer::new();

        assert!(!buf.is_full());

        for i in 0..CONTEXT_FRAMES {
            buf.push(&[i as i64, (i + 1) as i64, (i + 2) as i64, (i + 3) as i64]);
        }

        assert!(buf.is_full());

        let mut output = [0i64; CONTROL_SLOTS * CONTEXT_FRAMES];
        buf.get_context(&mut output);

        assert!(output.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_token_context() {
        let mut ctx = TokenContext::new();

        assert!(!ctx.is_ready());

        for i in 0..CONTEXT_FRAMES {
            ctx.push(&[i as i64; N_CODEBOOKS], &[i as i64; CONTROL_SLOTS]);
        }

        assert!(ctx.is_ready());

        let ctx_a = ctx.get_ctx_a();
        assert_eq!(ctx_a.len(), N_CODEBOOKS * CONTEXT_FRAMES);

        let ctx_b = ctx.get_ctx_b();
        assert_eq!(ctx_b.len(), CONTROL_SLOTS * CONTEXT_FRAMES);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut buf = AcousticTokenBuffer::new();

        for i in 0..CONTEXT_FRAMES + 50 {
            buf.push(&[i as i64; N_CODEBOOKS]);
        }

        assert!(buf.is_full());
        assert_eq!(buf.count(), CONTEXT_FRAMES);

        let mut output = [0i64; N_CODEBOOKS * CONTEXT_FRAMES];
        buf.get_context(&mut output);

        let first_val = output[0];
        assert!(first_val >= 50);
    }
}
