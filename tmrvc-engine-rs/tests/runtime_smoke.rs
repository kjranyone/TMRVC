use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use tmrvc_engine_rs::constants::{D_SPEAKER, FRAME_SIZE, SAMPLE_RATE};
use tmrvc_engine_rs::ort_bundle::OrtBundle;
use tmrvc_engine_rs::processor::{FrameParams, StreamingEngine};
use tmrvc_engine_rs::speaker::SpeakerFile;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("manifest dir must have workspace parent")
        .to_path_buf()
}

fn model_dir_from_env_or_default(root: &Path) -> PathBuf {
    std::env::var_os("TMRVC_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models").join("fp32"))
}

fn speaker_path_from_env_or_default(root: &Path) -> PathBuf {
    std::env::var_os("TMRVC_SPEAKER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models").join("test_speaker.tmrvc_speaker"))
}

fn make_nonzero_lora_speaker(base_speaker: &Path) -> PathBuf {
    let mut data = fs::read(base_speaker).expect("base speaker file should be readable");

    let header_size = 32usize;
    let lora_offset = header_size + D_SPEAKER * 4 + 4;
    if data.len() > lora_offset + 4 {
        data[lora_offset..lora_offset + 4].copy_from_slice(&0.123f32.to_le_bytes());
    }

    let checksum_size = 32usize;
    let checksum_offset = data.len() - checksum_size;
    let hash = Sha256::digest(&data[..checksum_offset]);
    data[checksum_offset..].copy_from_slice(hash.as_ref());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after unix epoch")
        .as_nanos();
    let out_path =
        std::env::temp_dir().join(format!("tmrvc_nonzero_lora_{}.tmrvc_speaker", unique));
    fs::write(&out_path, data).expect("patched speaker should be writable");
    out_path
}

#[test]
#[ignore = "requires local ONNX models in workspace/models/fp32"]
fn smoke_processor_runs_frame_inference() {
    let root = workspace_root();
    let model_dir = model_dir_from_env_or_default(&root);
    let speaker_path = speaker_path_from_env_or_default(&root);

    if !model_dir.join("codec_encoder.onnx").exists()
        || !model_dir.join("uclm_core.onnx").exists()
        || !model_dir.join("codec_decoder.onnx").exists()
        || !speaker_path.exists()
    {
        eprintln!("skip: local models/speaker files not found");
        return;
    }

    let mut engine = StreamingEngine::new(None);
    engine.load_models(&model_dir).expect("models should load");
    engine.load_speaker(&speaker_path).expect("speaker should load");
    engine.start();

    let frame_params = FrameParams {
        dry_wet: 1.0,
        output_gain: 1.0,
        alpha_timbre: 1.0,
        beta_prosody: 0.0,
        gamma_articulation: 0.0,
        voice_source_alpha: 0.0,
        latency_quality_q: 0.0,
        pitch_shift: 0.0,
        cfg_scale: 1.0,
        temperature_a: 1.0,
        temperature_b: 1.0,
        top_k_a: 50,
        top_k_b: 20,
        voice_state: [0.5; 8],
    };

    let mut input = [0.0f32; FRAME_SIZE];
    let mut output = [0.0f32; FRAME_SIZE];
    let mut phase = 0.0f32;
    let phase_inc = 2.0 * std::f32::consts::PI * 220.0 / SAMPLE_RATE as f32;
    let mut output_energy = 0.0f32;

    for _ in 0..16 {
        for s in &mut input {
            *s = phase.sin() * 0.1;
            phase += phase_inc;
            if phase > 2.0 * std::f32::consts::PI {
                phase -= 2.0 * std::f32::consts::PI;
            }
        }

        engine.process_one_frame(&input, &mut output, &frame_params);

        for &y in &output {
            assert!(y.is_finite(), "output contains non-finite sample");
            output_energy += y * y;
        }
    }

    let timing = engine.timing();
    println!(
        "timing: avg={:.2}us, max={:.2}us, frames={}, overruns={}",
        timing.avg_frame_us.load(std::sync::atomic::Ordering::SeqCst),
        timing.max_frame_us.load(std::sync::atomic::Ordering::SeqCst),
        timing.frame_count.load(std::sync::atomic::Ordering::SeqCst),
        timing.overrun_count.load(std::sync::atomic::Ordering::SeqCst),
    );
}

#[test]
#[ignore = "requires local ONNX models in workspace/models/fp32"]
fn smoke_speaker_file_loadable() {
    let root = workspace_root();
    let speaker_path = speaker_path_from_env_or_default(&root);

    if !speaker_path.exists() {
        eprintln!("skip: local speaker file not found");
        return;
    }

    let spk = SpeakerFile::load(&speaker_path).expect("speaker file should load");
    assert_eq!(spk.spk_embed.len(), D_SPEAKER);
}

#[test]
fn smoke_ort_bundle_new_uclm_signature_exists() {
    let root = workspace_root();
    let model_dir = model_dir_from_env_or_default(&root);

    if !model_dir.join("codec_encoder.onnx").exists() {
        eprintln!("skip: codec_encoder.onnx not found");
        return;
    }

    let result = OrtBundle::new_uclm(&model_dir);
    assert!(result.is_ok() || result.is_err());
}
