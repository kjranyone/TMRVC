use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use tmrvc_engine_rs::constants::{D_SPEAKER, HOP_LENGTH, LORA_DELTA_SIZE, SAMPLE_RATE};
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

    let header_size = 16usize;
    let lora_offset = header_size + D_SPEAKER * 4;
    data[lora_offset..lora_offset + 4].copy_from_slice(&0.123f32.to_le_bytes());

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
fn smoke_streaming_engine_runs_onehop_inference() {
    let root = workspace_root();
    let model_dir = model_dir_from_env_or_default(&root);
    let speaker_path = speaker_path_from_env_or_default(&root);

    if !model_dir.join("content_encoder.onnx").exists()
        || !model_dir.join("converter.onnx").exists()
        || !model_dir.join("vocoder.onnx").exists()
        || !model_dir.join("ir_estimator.onnx").exists()
        || !speaker_path.exists()
    {
        eprintln!("skip: local models/speaker files not found");
        return;
    }

    let mut engine = StreamingEngine::new(None);
    engine.load_models(&model_dir).expect("models should load");
    engine
        .load_speaker(&speaker_path)
        .expect("speaker should load");
    assert!(
        engine.is_ready(),
        "engine must be ready after model/speaker load"
    );

    let params = FrameParams {
        dry_wet: 1.0,
        output_gain: 1.0,
        alpha_timbre: 1.0,
        beta_prosody: 0.0,
        gamma_articulation: 0.0,
        latency_quality_q: 0.0,
    };

    // Feed several hops to warm up internal states and validate steady execution.
    let mut input = [0.0f32; HOP_LENGTH];
    let mut output = [0.0f32; HOP_LENGTH];
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

        engine.process_one_frame(&input, &mut output, &params);

        for &y in &output {
            assert!(y.is_finite(), "output contains non-finite sample");
            output_energy += y * y;
        }
    }

    assert!(output_energy > 0.0, "output energy should be positive");
}

#[test]
#[ignore = "requires local ONNX models in workspace/models/fp32"]
fn smoke_lora_contract_is_loadable_with_local_assets() {
    let root = workspace_root();
    let model_dir = model_dir_from_env_or_default(&root);
    let speaker_path = speaker_path_from_env_or_default(&root);

    if !model_dir.join("converter.onnx").exists() || !speaker_path.exists() {
        eprintln!("skip: local converter/speaker files not found");
        return;
    }

    let bundle = OrtBundle::load(&model_dir).expect("onnx bundle should load");
    let spk = SpeakerFile::load(&speaker_path).expect("speaker file should load");
    assert_eq!(spk.lora_delta.len(), LORA_DELTA_SIZE);

    let nonzero_speaker = make_nonzero_lora_speaker(&speaker_path);
    let patched = SpeakerFile::load(&nonzero_speaker).expect("patched speaker should load");
    assert!(
        patched.lora_delta.iter().any(|v| v.abs() > 1e-8),
        "patched speaker should contain non-zero LoRA"
    );

    let mut engine = StreamingEngine::new(None);
    engine.load_models(&model_dir).expect("models should load");
    engine
        .load_speaker(&nonzero_speaker)
        .expect("patched speaker should be loadable by engine");

    let params = FrameParams {
        dry_wet: 1.0,
        output_gain: 1.0,
        alpha_timbre: 1.0,
        beta_prosody: 0.0,
        gamma_articulation: 0.0,
        latency_quality_q: 0.0,
    };
    let input = [0.0f32; HOP_LENGTH];
    let mut output = [0.0f32; HOP_LENGTH];
    engine.process_one_frame(&input, &mut output, &params);
    assert!(output.iter().all(|v| v.is_finite()));

    assert!(
        bundle.converter_accepts_lora(),
        "converter.onnx must expose lora_delta input"
    );
    if bundle.has_hq_converter() {
        assert!(
            bundle.converter_hq_accepts_lora(),
            "converter_hq.onnx must expose lora_delta input"
        );
    }

    let _ = fs::remove_file(nonzero_speaker);
}
