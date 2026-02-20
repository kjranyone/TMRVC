use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use tmrvc_engine_rs::constants::{HOP_LENGTH, SAMPLE_RATE};
use tmrvc_engine_rs::processor::{FrameParams, StreamingEngine};
use tmrvc_engine_rs::resampler::PolyphaseResampler;
use tmrvc_engine_rs::wav_reader::read_wav;

fn parse_arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].clone())
}

fn parse_arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    parse_arg(args, key)
        .and_then(|v| v.parse::<f32>().ok())
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
    let mut out = vec![0.0; PolyphaseResampler::max_output_len(input.len(), src_rate, SAMPLE_RATE as u32)];
    let n = rs.process(input, &mut out);
    out.truncate(n);
    out
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || parse_arg(&args, "--input").is_none() || parse_arg(&args, "--output").is_none() {
        eprintln!(
            "Usage: cargo run -p tmrvc-engine-rs --bin offline_convert -- \\\n+  --input <in.wav> --output <out.wav> [--model-dir models/fp32] [--speaker models/test_speaker.tmrvc_speaker] \\\n+  [--dry-wet 0.85] [--output-gain 0.7] [--alpha-timbre 1.0] [--latency-q 1.0]"
        );
        std::process::exit(2);
    }

    let input_path = PathBuf::from(parse_arg(&args, "--input").unwrap());
    let output_path = PathBuf::from(parse_arg(&args, "--output").unwrap());
    let model_dir = PathBuf::from(parse_arg(&args, "--model-dir").unwrap_or_else(|| "models/fp32".to_string()));
    let speaker_path = PathBuf::from(
        parse_arg(&args, "--speaker").unwrap_or_else(|| "models/test_speaker.tmrvc_speaker".to_string()),
    );
    let dry_wet = parse_arg_f32(&args, "--dry-wet", 0.85).clamp(0.0, 1.0);
    let output_gain = parse_arg_f32(&args, "--output-gain", 0.7).max(0.0);
    let alpha_timbre = parse_arg_f32(&args, "--alpha-timbre", 1.0).clamp(0.0, 1.0);
    let latency_q = parse_arg_f32(&args, "--latency-q", 1.0).clamp(0.0, 1.0);

    let (raw, sr) = read_wav(&input_path)
        .with_context(|| format!("failed to read input wav: {}", input_path.display()))?;
    let input_24k = maybe_resample_to_24k(&raw, sr);

    let mut engine = StreamingEngine::new(None);
    engine
        .load_models(&model_dir)
        .with_context(|| format!("failed to load models: {}", model_dir.display()))?;
    engine
        .load_speaker(&speaker_path)
        .with_context(|| format!("failed to load speaker: {}", speaker_path.display()))?;

    let params = FrameParams {
        dry_wet,
        output_gain,
        alpha_timbre,
        beta_prosody: 0.0,
        gamma_articulation: 0.0,
        latency_quality_q: latency_q,
        voice_source_alpha: 0.0,
    };

    let mut out = Vec::with_capacity(input_24k.len() + HOP_LENGTH);
    let mut frame_in = [0.0f32; HOP_LENGTH];
    let mut frame_out = [0.0f32; HOP_LENGTH];

    for chunk in input_24k.chunks(HOP_LENGTH) {
        frame_in.fill(0.0);
        frame_in[..chunk.len()].copy_from_slice(chunk);
        engine.process_one_frame(&frame_in, &mut frame_out, &params);
        out.extend_from_slice(&frame_out[..chunk.len()]);
    }

    let norm_scale = normalize_peak(&mut out, 0.95);
    write_wav_24k(&output_path, &out)?;
    println!(
        "done: {} ({} samples @24kHz, norm_scale={:.4}, dry_wet={:.2}, gain={:.2}, q={:.2})",
        output_path.display(),
        out.len(),
        norm_scale,
        dry_wet,
        output_gain,
        latency_q
    );
    Ok(())
}
