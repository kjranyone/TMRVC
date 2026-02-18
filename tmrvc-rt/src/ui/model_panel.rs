use egui::Ui;

/// Model / Speaker selection state.
pub struct ModelPanelState {
    pub onnx_dir: String,
    pub speaker_path: String,
}

impl ModelPanelState {
    pub fn new() -> Self {
        Self {
            onnx_dir: String::new(),
            speaker_path: String::new(),
        }
    }
}

/// Model load/speaker load events.
pub enum ModelPanelEvent {
    None,
    LoadModels(String),
    LoadSpeaker(String),
}

/// Draw the model/speaker selection panel.
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
    });

    event
}
