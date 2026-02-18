//! Generate a mel spectrogram thumbnail as a PNG image.
//!
//! Produces a 100×100 (width × height) RGB PNG from mel data using
//! an inferno-inspired colormap.

use crate::constants::N_MELS;

const THUMB_WIDTH: usize = 100;
const THUMB_HEIGHT: usize = 100;

/// Inferno-inspired colormap with 256 entries.
/// Each entry is [R, G, B] in 0..255.
fn inferno_lut() -> [[u8; 3]; 256] {
    let mut lut = [[0u8; 3]; 256];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        // Piecewise linear approximation of inferno colormap
        let (r, g, b) = if t < 0.25 {
            let s = t / 0.25;
            (0.0 + s * 0.34, 0.0 + s * 0.01, 0.01 + s * 0.34)
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25;
            (0.34 + s * 0.48, 0.01 + s * 0.11, 0.35 + s * (-0.05))
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25;
            (0.82 + s * 0.14, 0.12 + s * 0.50, 0.30 + s * (-0.24))
        } else {
            let s = (t - 0.75) / 0.25;
            (0.96 + s * 0.02, 0.62 + s * 0.35, 0.06 + s * 0.87)
        };
        lut[i] = [
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        ];
    }
    lut
}

/// Generate a mel spectrogram thumbnail as a PNG byte vector.
///
/// `mel_data` is row-major `[N_MELS, num_frames]` (mel bin × time).
/// The output is a 100×100 RGB PNG.
pub fn generate_mel_thumbnail(mel_data: &[f32], num_frames: usize) -> Vec<u8> {
    assert!(
        mel_data.len() >= N_MELS * num_frames,
        "mel_data too short: expected {} elements, got {}",
        N_MELS * num_frames,
        mel_data.len()
    );

    let lut = inferno_lut();

    // Find min/max for normalisation
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in &mel_data[..N_MELS * num_frames] {
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }
    let range = if (vmax - vmin).abs() < 1e-8 {
        1.0
    } else {
        vmax - vmin
    };

    // Build 100×100 RGB image (row-major, top row = highest mel bin)
    let mut rgb = vec![0u8; THUMB_WIDTH * THUMB_HEIGHT * 3];

    for y in 0..THUMB_HEIGHT {
        // Flip vertically: top row = highest mel bin
        // Map pixel row y to fractional mel bin (interpolate N_MELS → THUMB_HEIGHT)
        let mel_frac = if N_MELS <= 1 {
            0.0
        } else {
            (THUMB_HEIGHT - 1 - y) as f64 * (N_MELS - 1) as f64 / (THUMB_HEIGHT - 1) as f64
        };
        let m0 = mel_frac as usize;
        let m1 = (m0 + 1).min(N_MELS - 1);
        let mel_alpha = (mel_frac - m0 as f64) as f32;

        for x in 0..THUMB_WIDTH {
            // Map x to fractional time position with linear interpolation
            let t_frac = if num_frames <= 1 {
                0.0
            } else {
                x as f64 * (num_frames - 1) as f64 / (THUMB_WIDTH - 1) as f64
            };
            let t0 = t_frac as usize;
            let t1 = (t0 + 1).min(num_frames - 1);
            let alpha = (t_frac - t0 as f64) as f32;

            // Bilinear interpolation: mel bin × time
            let v00 = mel_data[m0 * num_frames + t0];
            let v01 = mel_data[m0 * num_frames + t1];
            let v10 = mel_data[m1 * num_frames + t0];
            let v11 = mel_data[m1 * num_frames + t1];
            let v_top = v00 + alpha * (v01 - v00);
            let v_bot = v10 + alpha * (v11 - v10);
            let v = v_top + mel_alpha * (v_bot - v_top);

            // Normalise to [0, 1] then map to LUT index
            let norm = ((v - vmin) / range).clamp(0.0, 1.0);
            let idx = (norm * 255.0) as usize;
            let [r, g, b] = lut[idx.min(255)];

            let pixel = (y * THUMB_WIDTH + x) * 3;
            rgb[pixel] = r;
            rgb[pixel + 1] = g;
            rgb[pixel + 2] = b;
        }
    }

    // Encode as PNG
    let mut png_buf: Vec<u8> = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png_buf, THUMB_WIDTH as u32, THUMB_HEIGHT as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("PNG header write failed");
        writer
            .write_image_data(&rgb)
            .expect("PNG data write failed");
    }

    png_buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_valid_png() {
        // Create simple mel data: gradient along time
        let num_frames = 50;
        let mut mel_data = vec![0.0f32; N_MELS * num_frames];
        for m in 0..N_MELS {
            for t in 0..num_frames {
                mel_data[m * num_frames + t] = (m as f32 + t as f32) / (N_MELS + num_frames) as f32;
            }
        }

        let png_bytes = generate_mel_thumbnail(&mel_data, num_frames);

        // Check PNG magic
        assert!(png_bytes.len() > 8);
        assert_eq!(
            &png_bytes[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn handles_single_frame() {
        let mel_data = vec![1.0f32; N_MELS];
        let png_bytes = generate_mel_thumbnail(&mel_data, 1);
        assert!(png_bytes.len() > 8);
        assert_eq!(
            &png_bytes[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn handles_constant_values() {
        let num_frames = 20;
        let mel_data = vec![42.0f32; N_MELS * num_frames];
        let png_bytes = generate_mel_thumbnail(&mel_data, num_frames);
        assert!(png_bytes.len() > 8);
    }
}
