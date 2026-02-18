use std::sync::Arc;

use nih_plug::prelude::*;
use parking_lot::Mutex;

#[derive(Params)]
pub struct TmrvcParams {
    #[id = "dry_wet"]
    pub dry_wet: FloatParam,

    #[id = "output_gain"]
    pub output_gain: FloatParam,

    /// Latency-Quality trade-off: 0.0 = Live (low latency), 1.0 = Quality (HQ mode).
    #[id = "latency_quality"]
    pub latency_quality_q: FloatParam,

    /// Path to ONNX model directory — persisted in DAW project state.
    #[persist = "models_dir"]
    pub models_dir: Mutex<String>,

    /// Path to .tmrvc_speaker file — persisted in DAW project state.
    #[persist = "speaker_path"]
    pub speaker_path: Mutex<String>,

    // --- NAM (Neural Amp Modeler) ---
    #[id = "nam_enabled"]
    pub nam_enabled: BoolParam,

    #[id = "nam_mix"]
    pub nam_mix: FloatParam,

    /// Path to .nam profile file — persisted in DAW project state.
    #[persist = "nam_profile_path"]
    pub nam_profile_path: Mutex<String>,
}

impl Default for TmrvcParams {
    fn default() -> Self {
        Self {
            dry_wet: FloatParam::new("Dry/Wet", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(20.0))
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(1))
                .with_string_to_value(formatters::s2v_f32_percentage()),

            output_gain: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-60.0),
                    max: util::db_to_gain(12.0),
                    factor: FloatRange::gain_skew_factor(-60.0, 12.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),

            latency_quality_q: FloatParam::new(
                "Latency-Quality",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_value_to_string(Arc::new(|v| {
                if v < 0.3 {
                    format!("{:.0}% (Live)", v * 100.0)
                } else {
                    format!("{:.0}% (HQ)", v * 100.0)
                }
            })),

            models_dir: Mutex::new(String::new()),
            speaker_path: Mutex::new(String::new()),

            nam_enabled: BoolParam::new("NAM Enabled", false),

            nam_mix: FloatParam::new("NAM Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Linear(20.0))
                .with_unit("%")
                .with_value_to_string(formatters::v2s_f32_percentage(1))
                .with_string_to_value(formatters::s2v_f32_percentage()),

            nam_profile_path: Mutex::new(String::new()),
        }
    }
}
