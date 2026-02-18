use std::path::Path;

use anyhow::{bail, Result};
use base64::{engine::general_purpose::STANDARD, Engine};

use crate::constants::*;
use crate::dsp;
use crate::mel_thumbnail;
use crate::resampler::PolyphaseResampler;
use crate::speaker::{SpeakerFile, SpeakerMetadata};
use crate::speaker_encoder::SpeakerEncoderSession;
use crate::style::{StyleFile, StyleMetadata};
use crate::wav_reader;

/// Progress updates from the profile creation pipeline.
pub enum ProfileProgress {
    LoadingAudio(usize, usize),
    ComputingMel,
    RunningEncoder,
    Saving,
    Done(String),
    Error(String),
}

/// Create a voice profile (.tmrvc_speaker) from one or more WAV files.
///
/// Pipeline:
/// 1. Read WAV files
/// 2. Resample to 24kHz if needed
/// 3. Compute mel spectrogram
/// 4. Concatenate mel from all files along time axis
/// 5. Run speaker_encoder.onnx
/// 6. Save as .tmrvc_speaker (embedding-only mode: lora_delta = zeros)
pub fn create_voice_profile(
    encoder: &mut SpeakerEncoderSession,
    audio_paths: &[&Path],
    output_path: &Path,
    embedding_only: bool,
    profile_name: &str,
    author_name: &str,
    co_author_name: &str,
    licence_url: &str,
    progress_fn: &mut dyn FnMut(ProfileProgress),
) -> Result<()> {
    if audio_paths.is_empty() {
        bail!("No audio files provided");
    }

    // Initialize mel filterbank
    let mut filterbank = vec![0.0f32; N_MELS * N_FREQ_BINS];
    dsp::init_mel_filterbank(&mut filterbank);

    // Collect mel spectrograms from all files
    let mut all_mel: Vec<f32> = Vec::new();
    let mut total_frames: usize = 0;

    for (idx, &path) in audio_paths.iter().enumerate() {
        progress_fn(ProfileProgress::LoadingAudio(idx + 1, audio_paths.len()));

        let (samples, sample_rate) = wav_reader::read_wav(path)?;

        // Resample to 24kHz if needed
        let samples_24k = if sample_rate == SAMPLE_RATE as u32 {
            samples
        } else {
            let mut resampler = PolyphaseResampler::new(sample_rate, SAMPLE_RATE as u32);
            let max_out =
                PolyphaseResampler::max_output_len(samples.len(), sample_rate, SAMPLE_RATE as u32);
            let mut output = vec![0.0f32; max_out];
            let n_out = resampler.process(&samples, &mut output);
            output.truncate(n_out);
            output
        };

        if samples_24k.is_empty() {
            continue;
        }

        progress_fn(ProfileProgress::ComputingMel);

        let (mel_data, num_frames) = dsp::compute_mel_offline(&samples_24k, &filterbank);
        if num_frames == 0 {
            continue;
        }

        // Accumulate: we need to merge [N_MELS, num_frames] matrices along time
        if all_mel.is_empty() {
            all_mel = mel_data;
            total_frames = num_frames;
        } else {
            // Expand: for each mel bin, append the new frames
            let old_frames = total_frames;
            let new_total = old_frames + num_frames;
            let mut merged = vec![0.0f32; N_MELS * new_total];
            for m in 0..N_MELS {
                // Copy existing frames for this mel bin
                merged[m * new_total..m * new_total + old_frames]
                    .copy_from_slice(&all_mel[m * old_frames..(m + 1) * old_frames]);
                // Append new frames
                merged[m * new_total + old_frames..m * new_total + new_total]
                    .copy_from_slice(&mel_data[m * num_frames..(m + 1) * num_frames]);
            }
            all_mel = merged;
            total_frames = new_total;
        }
    }

    if total_frames == 0 {
        bail!("No valid audio frames found in the provided files");
    }

    // Generate thumbnail from mel data and encode as base64
    let thumbnail_png = mel_thumbnail::generate_mel_thumbnail(&all_mel, total_frames);
    let thumbnail_b64 = STANDARD.encode(&thumbnail_png);

    // Run speaker encoder
    progress_fn(ProfileProgress::RunningEncoder);
    let (spk_embed, mut lora_delta) = encoder.run(&all_mel, total_frames)?;

    // Embedding-only mode: zero out lora_delta
    if embedding_only {
        lora_delta.fill(0.0);
    }

    // Ensure lora_delta has correct size
    lora_delta.resize(LORA_DELTA_SIZE, 0.0);

    // Build metadata
    let profile_name = if profile_name.is_empty() {
        output_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string()
    } else {
        profile_name.to_string()
    };

    let source_audio_files: Vec<String> = audio_paths
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();

    let total_samples: u64 = audio_paths
        .iter()
        .filter_map(|p| wav_reader::read_wav(p).ok())
        .map(|(samples, _)| samples.len() as u64)
        .sum();

    // ISO 8601 UTC timestamp (basic: no chrono dependency)
    let created_at = {
        let d = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = d.as_secs();
        // Simple UTC formatting: days since epoch -> Y-M-D H:M:S
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Calculate year/month/day from days since 1970-01-01
        let (year, month, day) = days_to_ymd(days);
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            year, month, day, hours, minutes, seconds
        )
    };

    let metadata = SpeakerMetadata {
        profile_name,
        author_name: author_name.to_string(),
        co_author_name: co_author_name.to_string(),
        licence_url: licence_url.to_string(),
        thumbnail_b64,
        created_at,
        description: String::new(),
        source_audio_files,
        source_sample_count: total_samples,
        training_mode: "embedding".to_string(),
        checkpoint_name: String::new(),
    };

    // Save
    progress_fn(ProfileProgress::Saving);
    let speaker_file = SpeakerFile {
        spk_embed,
        lora_delta,
        metadata,
    };
    speaker_file.save(output_path)?;

    progress_fn(ProfileProgress::Done(output_path.display().to_string()));

    Ok(())
}

/// Create a style profile (.tmrvc_style) from one or more WAV files.
///
/// Style profile captures lightweight utterance controls used at runtime:
/// - target_log_f0 (prosody target)
/// - target_articulation (high-frequency articulation proxy)
/// - voiced_ratio
pub fn create_style_profile(
    audio_paths: &[&Path],
    output_path: &Path,
    progress_fn: &mut dyn FnMut(ProfileProgress),
) -> Result<()> {
    if audio_paths.is_empty() {
        bail!("No audio files provided");
    }

    // Initialize mel filterbank
    let mut filterbank = vec![0.0f32; N_MELS * N_FREQ_BINS];
    dsp::init_mel_filterbank(&mut filterbank);

    let mut total_frames: usize = 0;
    let mut voiced_frames: usize = 0;
    let mut sum_log_f0 = 0.0f32;
    let mut sum_articulation = 0.0f32;
    let mut total_samples: u64 = 0;

    for (idx, &path) in audio_paths.iter().enumerate() {
        progress_fn(ProfileProgress::LoadingAudio(idx + 1, audio_paths.len()));

        let (samples, sample_rate) = wav_reader::read_wav(path)?;

        // Resample to 24kHz if needed
        let samples_24k = if sample_rate == SAMPLE_RATE as u32 {
            samples
        } else {
            let mut resampler = PolyphaseResampler::new(sample_rate, SAMPLE_RATE as u32);
            let max_out =
                PolyphaseResampler::max_output_len(samples.len(), sample_rate, SAMPLE_RATE as u32);
            let mut output = vec![0.0f32; max_out];
            let n_out = resampler.process(&samples, &mut output);
            output.truncate(n_out);
            output
        };

        if samples_24k.is_empty() {
            continue;
        }

        total_samples += samples_24k.len() as u64;

        // Per-frame F0 analysis (causal window, same hop as runtime)
        let mut context = vec![0.0f32; WINDOW_LENGTH];
        let n_hops = (samples_24k.len() + HOP_LENGTH - 1) / HOP_LENGTH;
        for frame_idx in 0..n_hops {
            let start = frame_idx * HOP_LENGTH;
            let end = (start + HOP_LENGTH).min(samples_24k.len());
            let mut hop = [0.0f32; HOP_LENGTH];
            let copy_len = end.saturating_sub(start);
            if copy_len > 0 {
                hop[..copy_len].copy_from_slice(&samples_24k[start..start + copy_len]);
            }
            dsp::update_context_buffer(&mut context, &hop);
            let log_f0 = dsp::estimate_log_f0_autocorr(&context);
            if log_f0 > 0.0 {
                voiced_frames += 1;
                sum_log_f0 += log_f0;
            }
            total_frames += 1;
        }

        // Articulation analysis from mel
        progress_fn(ProfileProgress::ComputingMel);
        let (mel_data, n_mel_frames) = dsp::compute_mel_offline(&samples_24k, &filterbank);
        if n_mel_frames > 0 {
            for t in 0..n_mel_frames {
                let mut mel_frame = [0.0f32; N_MELS];
                for m in 0..N_MELS {
                    mel_frame[m] = mel_data[m * n_mel_frames + t];
                }
                sum_articulation += dsp::mel_articulation_proxy(&mel_frame);
            }
        }
    }

    if total_frames == 0 {
        bail!("No valid audio frames found in the provided files");
    }

    let target_log_f0 = if voiced_frames > 0 {
        sum_log_f0 / voiced_frames as f32
    } else {
        0.0
    };
    let target_articulation = sum_articulation / total_frames as f32;
    let voiced_ratio = voiced_frames as f32 / total_frames as f32;

    let display_name = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let source_audio_files: Vec<String> = audio_paths
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();

    // ISO 8601 UTC timestamp (basic: no chrono dependency)
    let created_at = {
        let d = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = d.as_secs();
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        let (year, month, day) = days_to_ymd(days);
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            year, month, day, hours, minutes, seconds
        )
    };

    let metadata = StyleMetadata {
        display_name,
        created_at,
        description: String::new(),
        source_audio_files,
        source_sample_count: total_samples,
    };

    progress_fn(ProfileProgress::Saving);
    let style = StyleFile {
        target_log_f0,
        target_articulation,
        voiced_ratio,
        metadata,
    };
    style.save(output_path)?;

    progress_fn(ProfileProgress::Done(output_path.display().to_string()));

    Ok(())
}
/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Shift to March-based year (simplifies leap year handling)
    let mut year = 1970u64;

    // Fast-forward through 400-year cycles
    let cycles_400 = days / 146097;
    year += cycles_400 * 400;
    days %= 146097;

    // 100-year cycles
    let mut cycles_100 = days / 36524;
    if cycles_100 == 4 {
        cycles_100 = 3;
    }
    year += cycles_100 * 100;
    days -= cycles_100 * 36524;

    // 4-year cycles
    let cycles_4 = days / 1461;
    year += cycles_4 * 4;
    days %= 1461;

    // Remaining single years
    let mut years_rem = days / 365;
    if years_rem == 4 {
        years_rem = 3;
    }
    year += years_rem;
    days -= years_rem * 365;

    // days is now the 0-based day-of-year
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_days: [u64; 12] = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }
    if month == 0 {
        month = 12;
    }
    let day = days + 1;

    (year, month, day)
}
