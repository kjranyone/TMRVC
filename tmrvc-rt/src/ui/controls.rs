use egui::Ui;

/// Control panel state.
pub struct ControlsState {
    pub dry_wet: f32,            // 0.0 (dry) to 1.0 (wet)
    pub gain_db: f32,            // -60 to +12 dB
    pub alpha_timbre: f32,       // 0.0 to 1.0
    pub beta_prosody: f32,       // 0.0 to 1.0
    pub gamma_articulation: f32, // 0.0 to 1.0
    pub latency_quality_q: f32,  // 0.0 (Live) to 1.0 (Quality)
}

impl ControlsState {
    pub fn new() -> Self {
        Self {
            dry_wet: 1.0,
            gain_db: 0.0,
            alpha_timbre: 1.0,
            beta_prosody: 0.0,
            gamma_articulation: 0.0,
            latency_quality_q: 0.0,
        }
    }

    /// Convert gain_db to linear multiplier.
    pub fn gain_linear(&self) -> f32 {
        10.0f32.powf(self.gain_db / 20.0)
    }
}

/// Draw control sliders.
///
/// Returns true if any value changed.
pub fn draw_controls(ui: &mut Ui, state: &mut ControlsState) -> bool {
    let mut changed = false;

    ui.group(|ui| {
        ui.label("Controls");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Dry/Wet");
            if ui
                .add(egui::Slider::new(&mut state.dry_wet, 0.0..=1.0).show_value(false))
                .changed()
            {
                changed = true;
            }
            ui.label(format!("{:.0}%", state.dry_wet * 100.0));
        });

        ui.horizontal(|ui| {
            ui.label("Gain   ");
            if ui
                .add(egui::Slider::new(&mut state.gain_db, -60.0..=12.0).suffix(" dB"))
                .changed()
            {
                changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Timbre ");
            if ui
                .add(egui::Slider::new(&mut state.alpha_timbre, 0.0..=1.0).show_value(false))
                .changed()
            {
                changed = true;
            }
            ui.label(format!("{:.2}", state.alpha_timbre));
        });

        ui.horizontal(|ui| {
            ui.label("Prosody");
            if ui
                .add(egui::Slider::new(&mut state.beta_prosody, 0.0..=1.0).show_value(false))
                .changed()
            {
                changed = true;
            }
            ui.label(format!("{:.2}", state.beta_prosody));
        });

        ui.horizontal(|ui| {
            ui.label("Artic. ");
            if ui
                .add(egui::Slider::new(&mut state.gamma_articulation, 0.0..=1.0).show_value(false))
                .changed()
            {
                changed = true;
            }
            ui.label(format!("{:.2}", state.gamma_articulation));
        });

        ui.horizontal(|ui| {
            ui.label("L-Q    ");
            if ui
                .add(egui::Slider::new(&mut state.latency_quality_q, 0.0..=1.0).show_value(false))
                .changed()
            {
                changed = true;
            }
            ui.label(format!("{:.2}", state.latency_quality_q));
        });
    });

    changed
}
