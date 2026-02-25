//! TTS (Text-to-Speech) pipeline for offline batch synthesis.
//!
//! Runs the TTS front-end (TextEncoder → DurationPredictor → LengthRegulate →
//! F0Predictor + ContentSynthesizer) and then feeds the resulting content frames
//! through the VC streaming backend (Converter + Vocoder + iSTFT + OLA).
//!
//! Unlike the VC `StreamingEngine` which processes one hop at a time from live
//! audio, the TTS pipeline:
//! 1. Runs the front-end models on the full utterance (variable-length batch)
//! 2. Iterates over content frames, feeding them one-by-one through the
//!    streaming Converter + Vocoder to produce audio.

use anyhow::{bail, Result};

use crate::constants::*;
use crate::ort_bundle::OrtBundle;

/// Result of the TTS front-end: per-frame content, F0, and voiced probability.
pub struct TTSFrontEndResult {
    /// Content features, shape [D_CONTENT * T] (row-major [D_CONTENT, T]).
    pub content: Vec<f32>,
    /// F0 in Hz per frame, length T.
    pub f0: Vec<f32>,
    /// Voiced probability per frame, length T.
    pub voiced: Vec<f32>,
    /// Number of output frames.
    pub n_frames: usize,
}

/// Language ID mapping (matches Python's N_LANGUAGES=4 convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Japanese = 0,
    English = 1,
    Chinese = 2,
    Korean = 3,
}

impl Language {
    pub fn from_str(s: &str) -> Self {
        match s {
            "ja" | "jpn" | "japanese" => Language::Japanese,
            "en" | "eng" | "english" => Language::English,
            "zh" | "zho" | "chinese" => Language::Chinese,
            "ko" | "kor" | "korean" => Language::Korean,
            _ => Language::Japanese, // default
        }
    }

    pub fn id(self) -> i64 {
        self as i64
    }
}

/// Expand phoneme-level features to frame-level via predicted durations.
///
/// Given `text_features` of shape [C, L] and `durations` of length L (float, in frames),
/// produces expanded features of shape [C, T] where T = sum(round(durations)).
///
/// Returns (expanded_features, total_frames).
pub fn length_regulate(
    text_features: &[f32],
    channels: usize,
    l: usize,
    durations: &[f32],
) -> (Vec<f32>, usize) {
    assert_eq!(text_features.len(), channels * l);
    assert_eq!(durations.len(), l);

    // Round durations to integers, clamp to >= 1
    let int_durs: Vec<usize> = durations
        .iter()
        .map(|&d| (d.round() as usize).max(1))
        .collect();
    let total_frames: usize = int_durs.iter().sum();

    let mut expanded = vec![0.0f32; channels * total_frames];
    let mut t = 0;
    for (phone_idx, &dur) in int_durs.iter().enumerate() {
        for _ in 0..dur {
            // Copy column phone_idx to column t
            for c in 0..channels {
                expanded[c * total_frames + t] = text_features[c * l + phone_idx];
            }
            t += 1;
        }
    }
    assert_eq!(t, total_frames);

    (expanded, total_frames)
}

/// Run the full TTS front-end pipeline.
///
/// phoneme_ids → TextEncoder → DurationPredictor → LengthRegulate →
/// F0Predictor + ContentSynthesizer → TTSFrontEndResult
pub fn run_tts_frontend(
    bundle: &mut OrtBundle,
    phoneme_ids: &[i64],
    language: Language,
    style: &[f32; D_STYLE],
) -> Result<TTSFrontEndResult> {
    if !bundle.has_tts() {
        bail!("TTS front-end models not loaded");
    }

    let l = phoneme_ids.len();
    if l == 0 {
        bail!("Empty phoneme sequence");
    }

    // 1. TextEncoder: phoneme_ids[1,L] → text_features[1,256,L]
    let text_features = bundle.run_text_encoder(phoneme_ids, language.id())?;
    assert_eq!(text_features.len(), D_TEXT_ENCODER * l);

    // 2. DurationPredictor: text_features + style → durations[1,L]
    let durations = bundle.run_duration_predictor(&text_features, l, style)?;

    // 3. LengthRegulate: expand from L phonemes to T frames
    let (expanded_features, t) = length_regulate(&text_features, D_TEXT_ENCODER, l, &durations);

    // 4. F0Predictor: expanded_features + style → f0[T], voiced[T]
    let (f0, voiced) = bundle.run_f0_predictor(&expanded_features, t, style)?;

    // 5. ContentSynthesizer: expanded_features → content[1,256,T]
    let content = bundle.run_content_synthesizer(&expanded_features, t)?;

    Ok(TTSFrontEndResult {
        content,
        f0,
        voiced,
        n_frames: t,
    })
}

/// Extract the content vector for frame `t` from a TTSFrontEndResult.
///
/// Returns a slice of D_CONTENT floats in the provided buffer.
pub fn extract_frame_content(result: &TTSFrontEndResult, frame: usize, out: &mut [f32]) {
    assert!(frame < result.n_frames);
    assert_eq!(out.len(), D_CONTENT);
    // content is [D_CONTENT, T] row-major
    for c in 0..D_CONTENT {
        out[c] = result.content[c * result.n_frames + frame];
    }
}

/// Extract F0 value for frame `t` as linear Hz suitable for model input.
///
/// Returns log(f0 + 1) if voiced, 0.0 otherwise.
pub fn extract_frame_f0(result: &TTSFrontEndResult, frame: usize) -> f32 {
    assert!(frame < result.n_frames);
    let f0_hz = result.f0[frame];
    let voiced = result.voiced[frame];
    if voiced > 0.5 && f0_hz > 0.0 {
        (f0_hz + 1.0).ln()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_regulate_basic() {
        // 2 channels, 3 phonemes, durations [2, 1, 3] → T=6
        let features = vec![
            // channel 0: [a, b, c]
            1.0, 2.0, 3.0, // channel 1: [d, e, f]
            4.0, 5.0, 6.0,
        ];
        let durations = vec![2.0, 1.0, 3.0];

        let (expanded, t) = length_regulate(&features, 2, 3, &durations);
        assert_eq!(t, 6);
        assert_eq!(expanded.len(), 2 * 6);

        // Channel 0: [a, a, b, c, c, c]
        assert_eq!(&expanded[0..6], &[1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
        // Channel 1: [d, d, e, f, f, f]
        assert_eq!(&expanded[6..12], &[4.0, 4.0, 5.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn length_regulate_clamps_to_one() {
        let features = vec![1.0, 2.0];
        let durations = vec![0.0, 0.1]; // both round to 0, clamped to 1
        let (expanded, t) = length_regulate(&features, 1, 2, &durations);
        assert_eq!(t, 2); // each phoneme gets at least 1 frame
        assert_eq!(expanded, vec![1.0, 2.0]);
    }

    #[test]
    fn language_from_str() {
        assert_eq!(Language::from_str("ja"), Language::Japanese);
        assert_eq!(Language::from_str("en"), Language::English);
        assert_eq!(Language::from_str("zh"), Language::Chinese);
        assert_eq!(Language::from_str("ko"), Language::Korean);
        assert_eq!(Language::from_str("unknown"), Language::Japanese);
    }

    #[test]
    fn extract_frame_f0_voiced() {
        let result = TTSFrontEndResult {
            content: vec![0.0; D_CONTENT * 3],
            f0: vec![200.0, 0.0, 300.0],
            voiced: vec![0.9, 0.1, 0.8],
            n_frames: 3,
        };
        // Frame 0: voiced, f0=200 → log(201)
        let f0_0 = extract_frame_f0(&result, 0);
        assert!((f0_0 - (201.0f32).ln()).abs() < 1e-5);
        // Frame 1: unvoiced → 0.0
        let f0_1 = extract_frame_f0(&result, 1);
        assert_eq!(f0_1, 0.0);
        // Frame 2: voiced, f0=300 → log(301)
        let f0_2 = extract_frame_f0(&result, 2);
        assert!((f0_2 - (301.0f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn extract_frame_content_basic() {
        let t = 3;
        let mut content = vec![0.0f32; D_CONTENT * t];
        // Set channel 0 of frame 1 to 42.0
        content[0 * t + 1] = 42.0;
        // Set last channel of frame 2 to 99.0
        content[(D_CONTENT - 1) * t + 2] = 99.0;

        let result = TTSFrontEndResult {
            content,
            f0: vec![0.0; t],
            voiced: vec![0.0; t],
            n_frames: t,
        };

        let mut out = vec![0.0f32; D_CONTENT];
        extract_frame_content(&result, 1, &mut out);
        assert_eq!(out[0], 42.0);

        extract_frame_content(&result, 2, &mut out);
        assert_eq!(out[D_CONTENT - 1], 99.0);
    }
}
