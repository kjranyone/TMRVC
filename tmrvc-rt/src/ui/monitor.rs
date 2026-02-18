use std::sync::atomic::Ordering;
use std::sync::Arc;

use egui::Ui;

use tmrvc_engine_rs::constants::{HOP_LENGTH, HQ_THRESHOLD_Q, MAX_LOOKAHEAD_HOPS, SAMPLE_RATE};
use tmrvc_engine_rs::processor::SharedStatus;

/// Draw the monitoring panel (levels, inference time, latency, style controls).
pub fn draw_monitor(ui: &mut Ui, status: &Arc<SharedStatus>) {
    let input_db = status.input_level_db.load(Ordering::Relaxed);
    let output_db = status.output_level_db.load(Ordering::Relaxed);
    let inference = status.inference_ms.load(Ordering::Relaxed);
    let inference_p50 = status.inference_p50_ms.load(Ordering::Relaxed);
    let inference_p95 = status.inference_p95_ms.load(Ordering::Relaxed);
    let frames = status.frame_count.load(Ordering::Relaxed);
    let overruns = status.overrun_count.load(Ordering::Relaxed);
    let underruns = status.underrun_count.load(Ordering::Relaxed);
    let q = status.latency_quality_q.load(Ordering::Relaxed);

    let alpha = status.alpha_timbre.load(Ordering::Relaxed);
    let beta = status.beta_prosody.load(Ordering::Relaxed);
    let gamma = status.gamma_articulation.load(Ordering::Relaxed);
    let estimated_log_f0 = status.estimated_log_f0.load(Ordering::Relaxed);
    let style_target_log_f0 = status.style_target_log_f0.load(Ordering::Relaxed);
    let style_target_artic = status.style_target_articulation.load(Ordering::Relaxed);
    let style_loaded = status.style_loaded.load(Ordering::Relaxed);

    let est_f0_hz = if estimated_log_f0 > 0.0 {
        estimated_log_f0.exp() - 1.0
    } else {
        0.0
    };
    let target_f0_hz = if style_target_log_f0 > 0.0 {
        style_target_log_f0.exp() - 1.0
    } else {
        0.0
    };

    let hop_ms = HOP_LENGTH as f32 / SAMPLE_RATE as f32 * 1000.0;

    ui.group(|ui| {
        ui.label("Monitor");
        ui.separator();

        // Input level meter
        ui.horizontal(|ui| {
            ui.label("Input  ");
            draw_level_bar(ui, input_db);
            ui.label(format!("{:>6.1} dB", input_db));
        });

        // Output level meter
        ui.horizontal(|ui| {
            ui.label("Output ");
            draw_level_bar(ui, output_db);
            ui.label(format!("{:>6.1} dB", output_db));
        });

        ui.add_space(4.0);

        // Inference time
        ui.horizontal(|ui| {
            ui.label(format!(
                "Inference: {:.1} ms / {:.1} ms hop",
                inference, hop_ms
            ));
        });
        ui.horizontal(|ui| {
            ui.label(format!(
                "Frame time p50/p95: {:.2} / {:.2} ms",
                inference_p50, inference_p95
            ));
        });

        // Estimated latency based on actual mode
        let hq_active = q > HQ_THRESHOLD_Q as f32;
        let algo_latency_ms = if hq_active {
            2.0 * hop_ms + MAX_LOOKAHEAD_HOPS as f32 * hop_ms // 80ms
        } else {
            2.0 * hop_ms // 20ms
        };
        let mode_label = if hq_active { "HQ" } else { "Live" };
        ui.horizontal(|ui| {
            ui.label(format!(
                "Latency: ~{:.0} ms ({} mode)",
                algo_latency_ms, mode_label
            ));
        });
        ui.horizontal(|ui| {
            ui.label(format!("Latency-Quality q: {:.2}", q));
        });

        ui.add_space(4.0);

        // Style/control observability
        ui.horizontal(|ui| {
            ui.label(format!(
                "alpha/beta/gamma: {:.2} / {:.2} / {:.2}",
                alpha, beta, gamma
            ));
        });
        ui.horizontal(|ui| {
            ui.label(format!(
                "F0 est/target: {:.1} / {:.1} Hz{}",
                est_f0_hz,
                target_f0_hz,
                if style_loaded { "" } else { " (style off)" }
            ));
        });
        ui.horizontal(|ui| {
            ui.label(format!(
                "Style articulation target: {:.3}",
                style_target_artic
            ));
        });

        // Frame count and underruns
        ui.horizontal(|ui| {
            ui.label(format!(
                "Frames: {}    Overruns: {}    Underruns: {}",
                frames, overruns, underruns
            ));
        });
    });
}

/// Draw a simple horizontal level bar.
fn draw_level_bar(ui: &mut Ui, db: f32) {
    // Map dB to 0..1 range: -60 dB = 0, 0 dB = 1
    let normalized = ((db + 60.0) / 60.0).clamp(0.0, 1.0);

    let desired_size = egui::vec2(120.0, 14.0);
    let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));

        // Filled portion
        let fill_width = rect.width() * normalized;
        let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));

        let color = if db > -6.0 {
            egui::Color32::from_rgb(220, 50, 50) // Red (near clipping)
        } else if db > -20.0 {
            egui::Color32::from_rgb(50, 200, 50) // Green (normal)
        } else {
            egui::Color32::from_rgb(50, 150, 50) // Dark green (quiet)
        };

        painter.rect_filled(fill_rect, 2.0, color);
    }
}
