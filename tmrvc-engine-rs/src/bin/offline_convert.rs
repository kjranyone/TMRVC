use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use tmrvc_engine_rs::constants::{D_VOICE_STATE, FRAME_SIZE, SAMPLE_RATE};
use tmrvc_engine_rs::processor::{FrameParams, SharedStatus, StreamingEngine};
use tmrvc_engine_rs::resampler::PolyphaseResampler;
use tmrvc_engine_rs::wav_reader::read_wav;

fn parse_arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

fn parse_arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    parse_arg(args, key)
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn parse_arg_usize(args: &[String], key: &str, default: usize) -> usize {
    parse_arg(args, key)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn normalize_peak(samples: &mut [f32], target_peak: f32) -> f32 {
    let peak = samples
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, |a, b| a.max(b));
    if peak <= 1e-9 || peak <= target_peak {
        return 1.0;
    }
    let scale = target_peak / peak;
    for s in samples {
        *s *= scale;
    }
    scale
}

fn write_wav_24k(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("failed to create output wav: {}", path.display()))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

fn maybe_resample_to_24k(input: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE as u32 {
        return input.to_vec();
    }
    let mut rs = PolyphaseResampler::new(src_rate, SAMPLE_RATE as u32);
    let mut out =
        vec![0.0; PolyphaseResampler::max_output_len(input.len(), src_rate, SAMPLE_RATE as u32)];
    let n = rs.process(input, &mut out);
    out.truncate(n);
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3
        || parse_arg(&args, "--input").is_none()
        || parse_arg(&args, "--output").is_none()
    {
        eprintln!(
            "Usage: cargo run -p tmrvc-engine-rs --bin offline_convert -- \\\n\
             --input <in.wav> --output <out.wav> [--model-dir models/fp32] \\\n\
             [--speaker models/test_speaker.tmrvc_speaker] \\\n\
             [--temperature-a 1.0] [--temperature-b 1.0] \\\n\
             [--top-k-a 50] [--top-k-b 20] \\\n\
             [--cfg-scale 1.0] [--dry-wet 1.0]"
        );
        std::process::exit(2);
    }

    let input_path = PathBuf::from(parse_arg(&args, "--input").unwrap());
    let output_path = PathBuf::from(parse_arg(&args, "--output").unwrap());
    let model_dir =
        PathBuf::from(parse_arg(&args, "--model-dir").unwrap_or_else(|| "models/fp32".to_string()));
    let speaker_path = PathBuf::from(
        parse_arg(&args, "--speaker")
            .unwrap_or_else(|| "models/test_speaker.tmrvc_speaker".to_string()),
    );
    let temperature_a = parse_arg_f32(&args, "--temperature-a", 1.0).max(0.01);
    let temperature_b = parse_arg_f32(&args, "--temperature-b", 1.0).max(0.01);
    let top_k_a = parse_arg_usize(&args, "--top-k-a", 50).max(1);
    let top_k_b = parse_arg_usize(&args, "--top-k-b", 20).max(1);
    let cfg_scale = parse_arg_f32(&args, "--cfg-scale", 1.0).max(0.1);
    let dry_wet = parse_arg_f32(&args, "--dry-wet", 1.0).clamp(0.0, 1.0);

    let (raw, sr) = read_wav(&input_path)
        .with_context(|| format!("failed to read input wav: {}", input_path.display()))?;
    let input_24k = maybe_resample_to_24k(&raw, sr);

    let status = Arc::new(SharedStatus::new());
    let mut engine = StreamingEngine::new(Some(status.clone()));

    engine
        .load_models(&model_dir)
        .with_context(|| format!("failed to load models: {}", model_dir.display()))?;
    engine
        .load_speaker(&speaker_path)
        .with_context(|| format!("failed to load speaker: {}", speaker_path.display()))?;
    engine.start();

    let frame_params = FrameParams {
        dry_wet,
        output_gain: 1.0,
        alpha_timbre: 1.0,
        beta_prosody: 0.0,
        gamma_articulation: 0.0,
        voice_source_alpha: 0.0,
        latency_quality_q: 0.0,
        pitch_shift: 0.0,
        cfg_scale,
        temperature_a,
        temperature_b,
        top_k_a,
        top_k_b,
        voice_state: [0.5; D_VOICE_STATE],
    };

    let mut out = Vec::with_capacity(input_24k.len() + FRAME_SIZE);
    let mut frame_in = [0.0f32; FRAME_SIZE];
    let mut frame_out = [0.0f32; FRAME_SIZE];

    for chunk in input_24k.chunks(FRAME_SIZE) {
        frame_in.fill(0.0);
        frame_in[..chunk.len()].copy_from_slice(chunk);
        engine.process_one_frame(&frame_in, &mut frame_out, &frame_params);
        out.extend_from_slice(&frame_out[..chunk.len()]);
    }

    let norm_scale = normalize_peak(&mut out, 0.95);
    write_wav_24k(&output_path, &out)?;

    let timing = engine.timing();
    println!(
        "done: {} ({} samples @24kHz, norm_scale={:.4}, temp_a={:.2}, temp_b={:.2}, cfg={:.2})",
        output_path.display(),
        out.len(),
        norm_scale,
        temperature_a,
        temperature_b,
        cfg_scale
    );
    println!(
        "timing: avg={:.2}us, max={:.2}us, frames={}, overruns={}",
        timing
            .avg_frame_us
            .load(std::sync::atomic::Ordering::SeqCst),
        timing
            .max_frame_us
            .load(std::sync::atomic::Ordering::SeqCst),
        timing.frame_count.load(std::sync::atomic::Ordering::SeqCst),
        timing
            .overrun_count
            .load(std::sync::atomic::Ordering::SeqCst),
    );
    Ok(())
}
