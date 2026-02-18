use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free SPSC ring buffer for f32 audio samples.
///
/// Single producer, single consumer. No mutex.
/// Capacity must be a power of 2.
pub struct SpscRingBuffer {
    buffer: Box<[f32]>,
    capacity: usize,
    mask: usize,
    read_pos: AtomicUsize,
    write_pos: AtomicUsize,
}

// SAFETY: SpscRingBuffer is designed for single-producer single-consumer
// across threads, with atomic read/write positions providing synchronization.
unsafe impl Send for SpscRingBuffer {}
unsafe impl Sync for SpscRingBuffer {}

impl SpscRingBuffer {
    /// Create a new ring buffer. `capacity` must be a power of 2.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "capacity must be a power of 2");
        Self {
            buffer: vec![0.0f32; capacity].into_boxed_slice(),
            capacity,
            mask: capacity - 1,
            read_pos: AtomicUsize::new(0),
            write_pos: AtomicUsize::new(0),
        }
    }

    /// Number of samples available to read.
    pub fn available(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// Number of samples that can be written.
    pub fn free_space(&self) -> usize {
        self.capacity - self.available()
    }

    /// Write samples into the buffer. Returns number of samples actually written.
    ///
    /// SAFETY: Only one thread should call write (producer).
    pub fn write(&self, data: &[f32]) -> usize {
        let free = self.free_space();
        let n = data.len().min(free);
        if n == 0 {
            return 0;
        }

        let w = self.write_pos.load(Ordering::Relaxed);
        let start = w & self.mask;

        // Get a mutable pointer to the buffer data.
        // SAFETY: single producer guarantees exclusive write access to the
        // region between write_pos and write_pos + n.
        let buf_ptr = self.buffer.as_ptr() as *mut f32;

        if start + n <= self.capacity {
            // Contiguous write
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buf_ptr.add(start), n);
            }
        } else {
            // Wrap-around write
            let first = self.capacity - start;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buf_ptr.add(start), first);
                std::ptr::copy_nonoverlapping(data.as_ptr().add(first), buf_ptr, n - first);
            }
        }

        self.write_pos.store(w.wrapping_add(n), Ordering::Release);
        n
    }

    /// Read samples from the buffer. Returns number of samples actually read.
    ///
    /// SAFETY: Only one thread should call read (consumer).
    pub fn read(&self, out: &mut [f32]) -> usize {
        let avail = self.available();
        let n = out.len().min(avail);
        if n == 0 {
            return 0;
        }

        let r = self.read_pos.load(Ordering::Relaxed);
        let start = r & self.mask;

        if start + n <= self.capacity {
            out[..n].copy_from_slice(&self.buffer[start..start + n]);
        } else {
            let first = self.capacity - start;
            out[..first].copy_from_slice(&self.buffer[start..self.capacity]);
            out[first..n].copy_from_slice(&self.buffer[..n - first]);
        }

        self.read_pos.store(r.wrapping_add(n), Ordering::Release);
        n
    }

    /// Reset the buffer to empty state.
    pub fn reset(&self) {
        self.read_pos.store(0, Ordering::Release);
        self.write_pos.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read() {
        let rb = SpscRingBuffer::new(8);
        let data = [1.0, 2.0, 3.0];
        assert_eq!(rb.write(&data), 3);
        assert_eq!(rb.available(), 3);

        let mut out = [0.0f32; 3];
        assert_eq!(rb.read(&mut out), 3);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        assert_eq!(rb.available(), 0);
    }

    #[test]
    fn test_wrap_around() {
        let rb = SpscRingBuffer::new(4);
        let data = [1.0, 2.0, 3.0];
        rb.write(&data);

        let mut out = [0.0f32; 2];
        rb.read(&mut out); // consume 2

        let data2 = [4.0, 5.0, 6.0];
        assert_eq!(rb.write(&data2), 3); // wraps around

        let mut out2 = [0.0f32; 4];
        assert_eq!(rb.read(&mut out2), 4);
        assert_eq!(out2, [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_full_buffer() {
        let rb = SpscRingBuffer::new(4);
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        // Can only fit 4 (capacity)
        assert_eq!(rb.write(&data), 4);
        assert_eq!(rb.free_space(), 0);
    }
}
