use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use atomic_float::AtomicF32;
use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::audio::stream::{self, AudioConfig, AudioStream};
use crate::ui::controls::{self, ControlsState};
use crate::ui::device_panel::{self, DevicePanelState};
use crate::ui::model_panel::{self, ModelPanelEvent, ModelPanelState};
use crate::ui::monitor;
use eframe::egui;
use tmrvc_engine_rs::constants::*;
use tmrvc_engine_rs::processor::{FrameParams, SharedStatus, StreamingEngine};
use tmrvc_engine_rs::ring_buffer::SpscRingBuffer;

/// Commands sent from GUI thread to processor thread.
pub enum Command {
    LoadModels(String),
    LoadSpeaker(String),
    Start,
    Stop,
    Quit,
}

/// Application state for egui.
pub struct TmrvcApp {
    // UI sub-state
    device_panel: DevicePanelState,
    model_panel: ModelPanelState,
    controls: ControlsState,

    // Shared atomics (GUI ↔ Processor)
    dry_wet: Arc<AtomicF32>,
    output_gain: Arc<AtomicF32>,
    latency_quality_q: Arc<AtomicF32>,
    status: Arc<SharedStatus>,

    // Ring buffers (cpal ↔ Processor)
    input_ring: Arc<SpscRingBuffer>,
    output_ring: Arc<SpscRingBuffer>,
    underrun_count: Arc<AtomicU64>,

    // Command channel (GUI → Processor)
    command_tx: Sender<Command>,

    // Audio stream (owned, kept alive)
    audio_stream: Option<AudioStream>,

    // Status text
    status_text: String,
    is_running: bool,
}

impl TmrvcApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Enumerate audio devices
        let input_devices = stream::list_input_devices();
        let output_devices = stream::list_output_devices();
        let input_names: Vec<String> = input_devices.iter().map(|(n, _)| n.clone()).collect();
        let output_names: Vec<String> = output_devices.iter().map(|(n, _)| n.clone()).collect();

        // Shared state
        let dry_wet = Arc::new(AtomicF32::new(1.0));
        let output_gain = Arc::new(AtomicF32::new(1.0));
        let latency_quality_q = Arc::new(AtomicF32::new(0.0));
        let status = Arc::new(SharedStatus::new());
        let input_ring = Arc::new(SpscRingBuffer::new(RING_BUFFER_CAPACITY));
        let output_ring = Arc::new(SpscRingBuffer::new(RING_BUFFER_CAPACITY));
        let underrun_count = Arc::new(AtomicU64::new(0));

        // Command channel
        let (command_tx, command_rx) = unbounded();

        // Spawn processor thread
        {
            let dry_wet = Arc::clone(&dry_wet);
            let output_gain = Arc::clone(&output_gain);
            let latency_quality_q = Arc::clone(&latency_quality_q);
            let status = Arc::clone(&status);
            let input_ring = Arc::clone(&input_ring);
            let output_ring = Arc::clone(&output_ring);

            thread::Builder::new()
                .name("tmrvc-processor".to_string())
                .spawn(move || {
                    processor_thread(
                        command_rx,
                        dry_wet,
                        output_gain,
                        latency_quality_q,
                        status,
                        input_ring,
                        output_ring,
                    );
                })
                .expect("Failed to spawn processor thread");
        }

        Self {
            device_panel: DevicePanelState::new(input_names, output_names),
            model_panel: ModelPanelState::new(),
            controls: ControlsState::new(),
            dry_wet,
            output_gain,
            latency_quality_q,
            status,
            input_ring,
            output_ring,
            underrun_count,
            command_tx,
            audio_stream: None,
            status_text: "Ready".to_string(),
            is_running: false,
        }
    }
}

impl eframe::App for TmrvcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("TMRVC Realtime VC");
            ui.separator();

            // Device panel
            device_panel::draw_device_panel(ui, &mut self.device_panel);
            ui.add_space(4.0);

            // Model panel
            match model_panel::draw_model_panel(ui, &mut self.model_panel) {
                ModelPanelEvent::LoadModels(dir) => {
                    let _ = self.command_tx.send(Command::LoadModels(dir));
                    self.status_text = "Loading models...".to_string();
                }
                ModelPanelEvent::LoadSpeaker(path) => {
                    let _ = self.command_tx.send(Command::LoadSpeaker(path));
                    self.status_text = "Loading speaker...".to_string();
                }
                ModelPanelEvent::None => {}
            }
            ui.add_space(4.0);

            // Start / Stop buttons
            ui.horizontal(|ui| {
                let start_enabled = !self.is_running;
                let stop_enabled = self.is_running;

                if ui
                    .add_enabled(start_enabled, egui::Button::new("Start VC"))
                    .clicked()
                {
                    self.start_vc();
                }
                if ui
                    .add_enabled(stop_enabled, egui::Button::new("Stop"))
                    .clicked()
                {
                    self.stop_vc();
                }
            });
            ui.add_space(4.0);

            // Monitor
            self.status.underrun_count.store(
                self.underrun_count.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            monitor::draw_monitor(ui, &self.status);
            ui.add_space(4.0);

            // Controls
            if controls::draw_controls(ui, &mut self.controls) {
                self.dry_wet.store(self.controls.dry_wet, Ordering::Relaxed);
                self.output_gain
                    .store(self.controls.gain_linear(), Ordering::Relaxed);
                self.latency_quality_q
                    .store(self.controls.latency_quality_q, Ordering::Relaxed);
            }
            ui.add_space(4.0);

            // Status bar
            ui.separator();
            ui.label(format!("Status: {}", self.status_text));
        });

        // Request repaint for live monitoring (~30 fps)
        if self.is_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
}

impl Drop for TmrvcApp {
    fn drop(&mut self) {
        let _ = self.command_tx.send(Command::Quit);
    }
}

impl TmrvcApp {
    fn start_vc(&mut self) {
        // Reset ring buffers
        self.input_ring.reset();
        self.output_ring.reset();
        self.underrun_count.store(0, Ordering::Relaxed);

        // Start cpal streams
        let config = AudioConfig {
            input_device_name: self
                .device_panel
                .selected_input_name()
                .map(|s| s.to_string()),
            output_device_name: self
                .device_panel
                .selected_output_name()
                .map(|s| s.to_string()),
            buffer_size: self.device_panel.selected_buffer_size(),
        };

        match AudioStream::start(
            &config,
            Arc::clone(&self.input_ring),
            Arc::clone(&self.output_ring),
            Arc::clone(&self.underrun_count),
        ) {
            Ok(stream) => {
                self.audio_stream = Some(stream);
                let _ = self.command_tx.send(Command::Start);
                self.is_running = true;
                self.status_text = "Running".to_string();
            }
            Err(e) => {
                self.status_text = format!("Audio error: {}", e);
                log::error!("Failed to start audio: {}", e);
            }
        }
    }

    fn stop_vc(&mut self) {
        let _ = self.command_tx.send(Command::Stop);
        self.audio_stream = None; // Drop stops streams
        self.is_running = false;
        self.status_text = "Stopped".to_string();
    }
}

const EVAL_WINDOW_FRAMES: usize = 512;
const EVAL_SUMMARY_EVERY_FRAMES: u64 = 50;
const EVAL_LOG_EVERY_FRAMES: u64 = 10;

struct EvalLogger {
    writer: Option<BufWriter<File>>,
    frame_ms_ring: Vec<f32>,
    q_ring: Vec<f32>,
    scratch: Vec<f32>,
    ring_pos: usize,
    ring_len: usize,
    sample_idx: u64,
}

impl EvalLogger {
    fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let csv_path = Path::new(&format!("tmrvc_eval_{}.csv", timestamp)).to_path_buf();
        Self::new_with_path(&csv_path)
    }

    fn new_with_path(csv_path: &Path) -> Self {
        let mut writer = None;

        match File::create(csv_path) {
            Ok(file) => {
                let mut w = BufWriter::new(file);
                if let Err(e) = writeln!(
                    w,
                    "unix_ms,frame_idx,frame_ms,p50_ms,p95_ms,q_current,q_mean,overrun_count"
                ) {
                    log::warn!("Failed to write CSV header: {}", e);
                } else {
                    log::info!("Evaluation CSV logging enabled: {:?}", csv_path);
                    writer = Some(w);
                }
            }
            Err(e) => {
                log::warn!("Evaluation CSV logging disabled (create failed): {}", e);
            }
        }

        Self {
            writer,
            frame_ms_ring: vec![0.0; EVAL_WINDOW_FRAMES],
            q_ring: vec![0.0; EVAL_WINDOW_FRAMES],
            scratch: Vec::with_capacity(EVAL_WINDOW_FRAMES),
            ring_pos: 0,
            ring_len: 0,
            sample_idx: 0,
        }
    }

    fn reset(&mut self, status: &Arc<SharedStatus>) {
        self.frame_ms_ring.fill(0.0);
        self.q_ring.fill(0.0);
        self.ring_pos = 0;
        self.ring_len = 0;
        self.sample_idx = 0;

        status.frame_count.store(0, Ordering::Relaxed);
        status.overrun_count.store(0, Ordering::Relaxed);
        status.inference_ms.store(0.0, Ordering::Relaxed);
        status.inference_p50_ms.store(0.0, Ordering::Relaxed);
        status.inference_p95_ms.store(0.0, Ordering::Relaxed);
    }

    fn record(&mut self, frame_ms: f32, q: f32, status: &Arc<SharedStatus>) {
        self.frame_ms_ring[self.ring_pos] = frame_ms;
        self.q_ring[self.ring_pos] = q;
        self.ring_pos = (self.ring_pos + 1) % EVAL_WINDOW_FRAMES;
        self.ring_len = self.ring_len.saturating_add(1).min(EVAL_WINDOW_FRAMES);
        self.sample_idx = self.sample_idx.saturating_add(1);

        if self.sample_idx % EVAL_SUMMARY_EVERY_FRAMES == 0 {
            let p50 = self.percentile(0.50);
            let p95 = self.percentile(0.95);
            status.inference_p50_ms.store(p50, Ordering::Relaxed);
            status.inference_p95_ms.store(p95, Ordering::Relaxed);
        }

        if self.sample_idx % EVAL_LOG_EVERY_FRAMES == 0 {
            self.write_csv_row(frame_ms, q, status);
        }
    }

    fn percentile(&mut self, q: f32) -> f32 {
        if self.ring_len == 0 {
            return 0.0;
        }

        self.scratch.clear();
        self.scratch
            .extend(self.frame_ms_ring.iter().take(self.ring_len).copied());
        self.scratch
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((self.scratch.len() - 1) as f32 * q)
            .round()
            .clamp(0.0, (self.scratch.len() - 1) as f32) as usize;
        self.scratch[idx]
    }

    fn q_mean(&self) -> f32 {
        if self.ring_len == 0 {
            return 0.0;
        }
        let sum: f32 = self.q_ring.iter().take(self.ring_len).sum();
        sum / self.ring_len as f32
    }

    fn write_csv_row(&mut self, frame_ms: f32, q: f32, status: &Arc<SharedStatus>) {
        let q_mean = self.q_mean();
        let Some(writer) = self.writer.as_mut() else {
            return;
        };

        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let p50 = status.inference_p50_ms.load(Ordering::Relaxed);
        let p95 = status.inference_p95_ms.load(Ordering::Relaxed);
        let overruns = status.overrun_count.load(Ordering::Relaxed);

        if let Err(e) = writeln!(
            writer,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
            unix_ms, self.sample_idx, frame_ms, p50, p95, q, q_mean, overruns
        ) {
            log::warn!("Failed to write evaluation CSV row: {}", e);
            return;
        }

        if self.sample_idx % 100 == 0 {
            if let Err(e) = writer.flush() {
                log::warn!("Failed to flush evaluation CSV: {}", e);
            }
        }
    }

    fn flush(&mut self) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        if let Err(e) = writer.flush() {
            log::warn!("Failed to flush evaluation CSV: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalLogger, SharedStatus};
    use std::fs;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn eval_logger_writes_csv_with_latency_and_q_fields() {
        let status = Arc::new(SharedStatus::new());
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let csv_path = std::env::temp_dir().join(format!("tmrvc_eval_test_{}.csv", stamp));

        let mut logger = EvalLogger::new_with_path(&csv_path);
        logger.reset(&status);

        for i in 0..120u64 {
            let frame_ms = 2.0 + ((i % 7) as f32) * 0.3;
            let q = ((i % 11) as f32) / 10.0;
            if i % 15 == 0 {
                status.overrun_count.fetch_add(1, Ordering::Relaxed);
            }
            logger.record(frame_ms, q, &status);
        }
        logger.flush();

        let csv = fs::read_to_string(&csv_path).expect("failed to read eval csv");
        assert!(csv.contains("p50_ms,p95_ms,q_current,q_mean,overrun_count"));
        assert!(csv.lines().count() >= 3); // header + at least 2 rows
        assert!(csv.contains(",0."));
        assert!(csv.contains(",1."));

        let _ = fs::remove_file(csv_path);
    }
}

/// Processor thread: receives commands, runs per-frame ONNX inference.
fn processor_thread(
    command_rx: Receiver<Command>,
    dry_wet: Arc<AtomicF32>,
    output_gain: Arc<AtomicF32>,
    latency_quality_q: Arc<AtomicF32>,
    status: Arc<SharedStatus>,
    input_ring: Arc<SpscRingBuffer>,
    output_ring: Arc<SpscRingBuffer>,
) {
    let mut engine = StreamingEngine::new(Some(Arc::clone(&status)));
    let mut eval_logger = EvalLogger::new();
    let mut running = false;

    loop {
        // Non-blocking command check
        while let Ok(cmd) = command_rx.try_recv() {
            match cmd {
                Command::LoadModels(dir) => match engine.load_models(Path::new(&dir)) {
                    Ok(()) => log::info!("Models loaded"),
                    Err(e) => log::error!("Failed to load models: {}", e),
                },
                Command::LoadSpeaker(path) => match engine.load_speaker(Path::new(&path)) {
                    Ok(()) => log::info!("Speaker loaded"),
                    Err(e) => log::error!("Failed to load speaker: {}", e),
                },
                Command::Start => {
                    eval_logger.reset(&status);
                    running = true;
                    status.is_running.store(true, Ordering::Relaxed);
                    log::info!("Processor started");
                }
                Command::Stop => {
                    running = false;
                    status.is_running.store(false, Ordering::Relaxed);
                    engine.reset();
                    eval_logger.flush();
                    log::info!("Processor stopped");
                }
                Command::Quit => {
                    eval_logger.flush();
                    log::info!("Processor thread exiting");
                    return;
                }
            }
        }

        if running && input_ring.available() >= HOP_LENGTH {
            // Read one hop from input ring
            let mut input_buf = [0.0f32; HOP_LENGTH];
            input_ring.read(&mut input_buf);

            // Process one frame
            let mut output_buf = [0.0f32; HOP_LENGTH];
            let frame_params = FrameParams {
                dry_wet: dry_wet.load(Ordering::Relaxed),
                output_gain: output_gain.load(Ordering::Relaxed),
                latency_quality_q: latency_quality_q.load(Ordering::Relaxed),
            };
            engine.process_one_frame(&input_buf, &mut output_buf, &frame_params);
            let frame_ms = status.inference_ms.load(Ordering::Relaxed);
            let q = latency_quality_q.load(Ordering::Relaxed);
            status.latency_quality_q.store(q, Ordering::Relaxed);
            eval_logger.record(frame_ms, q, &status);

            // Write to output ring
            output_ring.write(&output_buf);
        } else {
            // Sleep briefly to avoid busy-waiting
            thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
