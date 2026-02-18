use std::path::Path;

use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader};

/// Read a WAV file and return mono f32 samples normalized to [-1.0, 1.0],
/// along with the original sample rate.
///
/// Supports 16-bit, 24-bit, 32-bit integer and 32-bit float formats.
/// Multi-channel audio is downmixed to mono by averaging all channels.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader =
        WavReader::open(path).with_context(|| format!("Failed to open WAV: {:?}", path))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    if channels == 0 {
        bail!("WAV file has 0 channels");
    }

    let samples_f32: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let scale = match bits {
                16 => 1.0 / 32768.0,
                24 => 1.0 / 8388608.0,
                32 => 1.0 / 2147483648.0,
                _ => bail!("Unsupported bit depth: {}", bits),
            };
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .context("Failed to read integer samples")?
        }
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .context("Failed to read float samples")?,
    };

    // Downmix to mono
    let num_frames = samples_f32.len() / channels;
    let mono = if channels == 1 {
        samples_f32
    } else {
        let inv_ch = 1.0 / channels as f32;
        (0..num_frames)
            .map(|i| {
                let start = i * channels;
                let sum: f32 = samples_f32[start..start + channels].iter().sum();
                sum * inv_ch
            })
            .collect()
    };

    Ok((mono, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};

    fn write_test_wav(path: &Path, sample_rate: u32, channels: u16, samples: &[i16]) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn read_mono_16bit() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mono16.wav");

        // Full-scale sine: 1000 samples
        let samples: Vec<i16> = (0..1000)
            .map(|i| {
                (32767.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24000.0).sin()) as i16
            })
            .collect();
        write_test_wav(&path, 24000, 1, &samples);

        let (mono, sr) = read_wav(&path).unwrap();
        assert_eq!(sr, 24000);
        assert_eq!(mono.len(), 1000);
        // Check normalization: max absolute value should be close to 1.0
        let max_abs = mono.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_abs > 0.9 && max_abs <= 1.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_stereo_downmix() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_stereo.wav");

        // Stereo: L=+16384, R=-16384 → mono should be ~0
        let mut samples = Vec::new();
        for _ in 0..500 {
            samples.push(16384i16);
            samples.push(-16384i16);
        }
        write_test_wav(&path, 48000, 2, &samples);

        let (mono, sr) = read_wav(&path).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(mono.len(), 500);
        // After averaging, all values should be ~0
        assert!(mono.iter().all(|v| v.abs() < 0.001));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_float32_wav() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_float32.wav");

        let spec = WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        for i in 0..240 {
            let v = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 24000.0).sin();
            writer.write_sample(v).unwrap();
        }
        writer.finalize().unwrap();

        let (mono, sr) = read_wav(&path).unwrap();
        assert_eq!(sr, 24000);
        assert_eq!(mono.len(), 240);
        assert!(mono.iter().all(|v| v.is_finite()));

        let _ = std::fs::remove_file(&path);
    }
}
