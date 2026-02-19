use egui::Ui;

/// State for the voice profile creation panel.
pub struct VoiceProfilePanelState {
    pub audio_paths: Vec<String>,
    pub output_path: String,
    pub style_output_path: String,
    pub encoder_available: bool,
    pub creating: bool,
    pub progress_message: String,
    pub python_available: bool,
    pub finetune_mode: bool,
    pub checkpoint_path: String,
    pub profile_name: String,
    pub author_name: String,
    pub co_author_name: String,
    pub licence_url: String,
}

/// Events emitted by the voice profile panel.
pub enum VoiceProfileEvent {
    None,
    CreateEmbeddingProfile {
        audio_paths: Vec<String>,
        output_path: String,
        profile_name: String,
        author_name: String,
        co_author_name: String,
        licence_url: String,
    },
    CreateFinetunedProfile {
        audio_paths: Vec<String>,
        output_path: String,
        checkpoint_path: String,
        profile_name: String,
        author_name: String,
        co_author_name: String,
        licence_url: String,
    },
    CreateStyleProfile {
        audio_paths: Vec<String>,
        output_path: String,
    },
}

impl VoiceProfilePanelState {
    pub fn new(encoder_available: bool, python_available: bool) -> Self {
        Self {
            audio_paths: Vec::new(),
            output_path: String::new(),
            style_output_path: String::new(),
            encoder_available,
            creating: false,
            progress_message: String::new(),
            python_available,
            finetune_mode: false,
            checkpoint_path: String::new(),
            profile_name: String::new(),
            author_name: String::new(),
            co_author_name: String::new(),
            licence_url: String::new(),
        }
    }
}

/// Draw the voice/style profile creation panel.
///
/// Returns a `VoiceProfileEvent` when the user triggers an action.
pub fn draw_voice_profile_panel(
    ui: &mut Ui,
    state: &mut VoiceProfilePanelState,
) -> VoiceProfileEvent {
    if !state.encoder_available {
        return VoiceProfileEvent::None;
    }

    let mut event = VoiceProfileEvent::None;

    ui.group(|ui| {
        ui.label("Profile Creator");
        ui.separator();

        // Audio file list
        ui.horizontal(|ui| {
            ui.label("Audio:");
            if ui
                .add_enabled(!state.creating, egui::Button::new("Add WAV..."))
                .clicked()
            {
                if let Some(paths) = rfd::FileDialog::new()
                    .set_title("Select WAV files")
                    .add_filter("WAV files", &["wav"])
                    .pick_files()
                {
                    for p in paths {
                        state.audio_paths.push(p.display().to_string());
                    }
                }
            }
            if !state.audio_paths.is_empty()
                && ui
                    .add_enabled(!state.creating, egui::Button::new("Clear"))
                    .clicked()
            {
                state.audio_paths.clear();
            }
        });

        if !state.audio_paths.is_empty() {
            ui.indent("audio_list", |ui| {
                let mut remove_idx = None;
                for (i, path) in state.audio_paths.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let display = std::path::Path::new(path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.clone());
                        ui.label(&display);
                        if !state.creating && ui.small_button("X").clicked() {
                            remove_idx = Some(i);
                        }
                    });
                }
                if let Some(idx) = remove_idx {
                    state.audio_paths.remove(idx);
                }
            });
        } else {
            ui.label("  (Add one or more WAV files)");
        }

        // Speaker profile output path
        ui.horizontal(|ui| {
            ui.label("Speaker:");
            let display = if state.output_path.is_empty() {
                "(not selected)"
            } else {
                &state.output_path
            };
            ui.add(egui::Label::new(display).truncate());
            if ui
                .add_enabled(!state.creating, egui::Button::new("Save As..."))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Save speaker profile")
                    .add_filter("TMRVC Speaker", &["tmrvc_speaker"])
                    .save_file()
                {
                    state.output_path = path.display().to_string();
                }
            }
        });

        // Style profile output path
        ui.horizontal(|ui| {
            ui.label("Style:");
            let display = if state.style_output_path.is_empty() {
                "(optional)"
            } else {
                &state.style_output_path
            };
            ui.add(egui::Label::new(display).truncate());
            if ui
                .add_enabled(!state.creating, egui::Button::new("Save As..."))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Save style profile")
                    .add_filter("TMRVC Style", &["tmrvc_style"])
                    .save_file()
                {
                    state.style_output_path = path.display().to_string();
                }
            }
        });

        // Metadata fields
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.add_enabled(
                !state.creating,
                egui::TextEdit::singleline(&mut state.profile_name)
                    .desired_width(200.0)
                    .hint_text("Profile name"),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Author:");
            ui.add_enabled(
                !state.creating,
                egui::TextEdit::singleline(&mut state.author_name)
                    .desired_width(200.0)
                    .hint_text("Author name"),
            );
        });

        // Optional metadata (collapsible)
        egui::CollapsingHeader::new("Optional")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Co-Author:");
                    ui.add_enabled(
                        !state.creating,
                        egui::TextEdit::singleline(&mut state.co_author_name)
                            .desired_width(180.0)
                            .hint_text("(optional)"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Licence URL:");
                    ui.add_enabled(
                        !state.creating,
                        egui::TextEdit::singleline(&mut state.licence_url)
                            .desired_width(180.0)
                            .hint_text("https://..."),
                    );
                });
            });

        // Mode selection (speaker profile)
        if state.python_available {
            ui.horizontal(|ui| {
                ui.label("Mode:");
                ui.radio_value(&mut state.finetune_mode, false, "Embedding only");
                ui.radio_value(&mut state.finetune_mode, true, "Fine-tune (LoRA)");
            });

            if state.finetune_mode {
                ui.horizontal(|ui| {
                    ui.label("Checkpoint:");
                    let display = if state.checkpoint_path.is_empty() {
                        "(not selected)"
                    } else {
                        &state.checkpoint_path
                    };
                    ui.add(egui::Label::new(display).truncate());
                    if ui
                        .add_enabled(!state.creating, egui::Button::new("Browse..."))
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("Select checkpoint")
                            .add_filter("PyTorch Checkpoint", &["pt"])
                            .pick_file()
                        {
                            state.checkpoint_path = path.display().to_string();
                        }
                    }
                });
            }
        }

        let can_create_speaker = !state.creating
            && !state.audio_paths.is_empty()
            && !state.output_path.is_empty()
            && (!state.finetune_mode || !state.checkpoint_path.is_empty());

        let can_create_style =
            !state.creating && !state.audio_paths.is_empty() && !state.style_output_path.is_empty();

        ui.horizontal(|ui| {
            if ui
                .add_enabled(can_create_speaker, egui::Button::new("Create Speaker"))
                .clicked()
            {
                if state.finetune_mode {
                    event = VoiceProfileEvent::CreateFinetunedProfile {
                        audio_paths: state.audio_paths.clone(),
                        output_path: state.output_path.clone(),
                        checkpoint_path: state.checkpoint_path.clone(),
                        profile_name: state.profile_name.clone(),
                        author_name: state.author_name.clone(),
                        co_author_name: state.co_author_name.clone(),
                        licence_url: state.licence_url.clone(),
                    };
                } else {
                    event = VoiceProfileEvent::CreateEmbeddingProfile {
                        audio_paths: state.audio_paths.clone(),
                        output_path: state.output_path.clone(),
                        profile_name: state.profile_name.clone(),
                        author_name: state.author_name.clone(),
                        co_author_name: state.co_author_name.clone(),
                        licence_url: state.licence_url.clone(),
                    };
                }
            }

            if ui
                .add_enabled(can_create_style, egui::Button::new("Create Style"))
                .clicked()
            {
                event = VoiceProfileEvent::CreateStyleProfile {
                    audio_paths: state.audio_paths.clone(),
                    output_path: state.style_output_path.clone(),
                };
            }

            if state.creating {
                ui.spinner();
            }
        });

        if !state.progress_message.is_empty() {
            ui.label(&state.progress_message);
        }
    });

    event
}
