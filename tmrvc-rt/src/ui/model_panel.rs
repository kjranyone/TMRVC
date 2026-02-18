use std::path::PathBuf;

use egui::Ui;

/// Model / Speaker / Style selection state.
pub struct ModelPanelState {
    pub onnx_dir: String,
    pub speaker_path: String,
    pub style_path: String,
}

impl ModelPanelState {
    pub fn new() -> Self {
        let (onnx_dir, speaker_path, style_path) = detect_default_model_paths();
        Self {
            onnx_dir,
            speaker_path,
            style_path,
        }
    }
}

fn detect_default_model_paths() -> (String, String, String) {
    let onnx_from_env = std::env::var("TMRVC_ONNX_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let speaker_from_env = std::env::var("TMRVC_SPEAKER_PATH")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let style_from_env = std::env::var("TMRVC_STYLE_PATH")
        .ok()
        .filter(|s| !s.trim().is_empty());

    let roots = candidate_roots();

    let onnx_dir = onnx_from_env.unwrap_or_else(|| {
        for root in &roots {
            let p = root.join("models").join("fp32");
            if p.is_dir() {
                return p.display().to_string();
            }
        }
        String::new()
    });

    let speaker_path = speaker_from_env.unwrap_or_else(|| {
        let candidates = [
            ["models", "demo_fewshot.tmrvc_speaker"],
            ["models", "test_speaker_ft.tmrvc_speaker"],
            ["models", "test_speaker.tmrvc_speaker"],
        ];
        for root in &roots {
            for parts in candidates {
                let p = root.join(parts[0]).join(parts[1]);
                if p.is_file() {
                    return p.display().to_string();
                }
            }
        }
        String::new()
    });

    let style_path = style_from_env.unwrap_or_else(|| {
        let candidates = [
            ["models", "demo_style.tmrvc_style"],
            ["models", "test_style.tmrvc_style"],
        ];
        for root in &roots {
            for parts in candidates {
                let p = root.join(parts[0]).join(parts[1]);
                if p.is_file() {
                    return p.display().to_string();
                }
            }
        }
        String::new()
    });

    (onnx_dir, speaker_path, style_path)
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.clone());
        if let Some(parent) = cwd.parent() {
            out.push(parent.to_path_buf());
            if let Some(grand_parent) = parent.parent() {
                out.push(grand_parent.to_path_buf());
            }
        }
    }
    out
}

/// Model load/speaker load/style load events.
pub enum ModelPanelEvent {
    None,
    LoadModels(String),
    LoadSpeaker(String),
    LoadStyle(String),
}

/// Draw the model/speaker/style selection panel.
pub fn draw_model_panel(ui: &mut Ui, state: &mut ModelPanelState) -> ModelPanelEvent {
    let mut event = ModelPanelEvent::None;

    ui.group(|ui| {
        ui.label("Model");
        ui.separator();

        // ONNX directory
        ui.horizontal(|ui| {
            ui.label("ONNX:");
            let display = if state.onnx_dir.is_empty() {
                "(not selected)"
            } else {
                &state.onnx_dir
            };
            ui.add(egui::Label::new(display).truncate());
            if ui.button("Browse...").clicked() {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Select ONNX model directory")
                    .pick_folder()
                {
                    let path = dir.display().to_string();
                    state.onnx_dir = path.clone();
                    event = ModelPanelEvent::LoadModels(path);
                }
            }
        });

        // Speaker file
        ui.horizontal(|ui| {
            ui.label("Speaker:");
            let display = if state.speaker_path.is_empty() {
                "(not selected)"
            } else {
                &state.speaker_path
            };
            ui.add(egui::Label::new(display).truncate());
            if ui.button("Browse...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Select speaker file")
                    .add_filter("TMRVC Speaker", &["tmrvc_speaker"])
                    .pick_file()
                {
                    let p = path.display().to_string();
                    state.speaker_path = p.clone();
                    event = ModelPanelEvent::LoadSpeaker(p);
                }
            }
        });

        // Style file
        ui.horizontal(|ui| {
            ui.label("Style:");
            let display = if state.style_path.is_empty() {
                "(optional)"
            } else {
                &state.style_path
            };
            ui.add(egui::Label::new(display).truncate());
            if ui.button("Browse...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Select style file")
                    .add_filter("TMRVC Style", &["tmrvc_style"])
                    .pick_file()
                {
                    let p = path.display().to_string();
                    state.style_path = p.clone();
                    event = ModelPanelEvent::LoadStyle(p);
                }
            }
        });
    });

    event
}
