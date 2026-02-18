use egui::Ui;

/// Audio device selection state.
pub struct DevicePanelState {
    pub input_device_idx: usize,
    pub output_device_idx: usize,
    pub buffer_size_idx: usize,
    pub input_device_names: Vec<String>,
    pub output_device_names: Vec<String>,
}

const BUFFER_SIZES: &[u32] = &[128, 256, 512, 1024];

impl DevicePanelState {
    pub fn new(input_names: Vec<String>, output_names: Vec<String>) -> Self {
        Self {
            input_device_idx: 0,
            output_device_idx: 0,
            buffer_size_idx: 1, // 256 default
            input_device_names: input_names,
            output_device_names: output_names,
        }
    }

    pub fn selected_buffer_size(&self) -> u32 {
        BUFFER_SIZES[self.buffer_size_idx]
    }

    pub fn selected_input_name(&self) -> Option<&str> {
        self.input_device_names
            .get(self.input_device_idx)
            .map(|s| s.as_str())
    }

    pub fn selected_output_name(&self) -> Option<&str> {
        self.output_device_names
            .get(self.output_device_idx)
            .map(|s| s.as_str())
    }
}

/// Draw the audio device selection panel.
///
/// Returns true if any selection changed.
pub fn draw_device_panel(ui: &mut Ui, state: &mut DevicePanelState) -> bool {
    let mut changed = false;

    ui.group(|ui| {
        ui.label("Audio Device");
        ui.separator();

        // Input device
        ui.horizontal(|ui| {
            ui.label("Input:");
            let current = state
                .input_device_names
                .get(state.input_device_idx)
                .cloned()
                .unwrap_or_else(|| "(none)".to_string());
            egui::ComboBox::from_id_salt("input_device")
                .selected_text(&current)
                .show_ui(ui, |ui| {
                    for (i, name) in state.input_device_names.iter().enumerate() {
                        if ui
                            .selectable_value(&mut state.input_device_idx, i, name)
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
        });

        // Output device
        ui.horizontal(|ui| {
            ui.label("Output:");
            let current = state
                .output_device_names
                .get(state.output_device_idx)
                .cloned()
                .unwrap_or_else(|| "(none)".to_string());
            egui::ComboBox::from_id_salt("output_device")
                .selected_text(&current)
                .show_ui(ui, |ui| {
                    for (i, name) in state.output_device_names.iter().enumerate() {
                        if ui
                            .selectable_value(&mut state.output_device_idx, i, name)
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
        });

        // Buffer size
        ui.horizontal(|ui| {
            ui.label("Buffer:");
            let current_label = format!("{} samples", BUFFER_SIZES[state.buffer_size_idx]);
            egui::ComboBox::from_id_salt("buffer_size")
                .selected_text(&current_label)
                .show_ui(ui, |ui| {
                    for (i, &sz) in BUFFER_SIZES.iter().enumerate() {
                        let label = format!("{} samples", sz);
                        if ui
                            .selectable_value(&mut state.buffer_size_idx, i, &label)
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
        });
    });

    changed
}
