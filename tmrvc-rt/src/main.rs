mod app;
mod audio;
mod ui;

use app::TmrvcApp;

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("TMRVC Realtime VC")
            .with_inner_size([480.0, 600.0])
            .with_min_inner_size([400.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "TMRVC Realtime VC",
        options,
        Box::new(|cc| Ok(Box::new(TmrvcApp::new(cc)))),
    )
}
