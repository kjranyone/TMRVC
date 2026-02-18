use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, Stream, StreamConfig};

use tmrvc_engine_rs::ring_buffer::SpscRingBuffer;

/// Audio stream configuration.
pub struct AudioConfig {
    pub input_device_name: Option<String>,
    pub output_device_name: Option<String>,
    pub buffer_size: u32,
}

/// Manages cpal input and output streams.
pub struct AudioStream {
    _input_stream: Stream,
    _output_stream: Stream,
}

impl AudioStream {
    /// Create and start audio streams.
    ///
    /// - `input_ring`: cpal input callback writes here (producer)
    /// - `output_ring`: cpal output callback reads from here (consumer)
    /// - `underrun_count`: incremented when output buffer underruns
    pub fn start(
        config: &AudioConfig,
        input_ring: Arc<SpscRingBuffer>,
        output_ring: Arc<SpscRingBuffer>,
        underrun_count: Arc<AtomicU64>,
    ) -> Result<Self> {
        let host = cpal::default_host();

        let input_device = find_input_device(&host, config.input_device_name.as_deref())?;
        let output_device = find_output_device(&host, config.output_device_name.as_deref())?;

        let sample_rate = cpal::SampleRate(
            input_device
                .default_input_config()
                .context("No default input config")?
                .sample_rate()
                .0,
        );

        let stream_config = StreamConfig {
            channels: 1,
            sample_rate,
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size),
        };

        // Input stream: microphone → input_ring
        let input_stream = input_device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    input_ring.write(data);
                },
                |err| {
                    log::error!("Input stream error: {}", err);
                },
                None,
            )
            .context("Failed to build input stream")?;

        // Output stream: output_ring → speakers
        let output_stream = output_device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let read = output_ring.read(data);
                    if read < data.len() {
                        // Underrun: zero-fill remaining samples
                        data[read..].fill(0.0);
                        underrun_count.fetch_add(1, Ordering::Relaxed);
                    }
                },
                |err| {
                    log::error!("Output stream error: {}", err);
                },
                None,
            )
            .context("Failed to build output stream")?;

        input_stream
            .play()
            .context("Failed to start input stream")?;
        output_stream
            .play()
            .context("Failed to start output stream")?;

        Ok(Self {
            _input_stream: input_stream,
            _output_stream: output_stream,
        })
    }
}

/// Enumerate available input devices.
pub fn list_input_devices() -> Vec<(String, Device)> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    if let Ok(devs) = host.input_devices() {
        for dev in devs {
            if let Ok(name) = dev.name() {
                // Filter to devices that support f32 mono
                if supports_f32_input(&dev) {
                    devices.push((name, dev));
                }
            }
        }
    }
    devices
}

/// Enumerate available output devices.
pub fn list_output_devices() -> Vec<(String, Device)> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    if let Ok(devs) = host.output_devices() {
        for dev in devs {
            if let Ok(name) = dev.name() {
                if supports_f32_output(&dev) {
                    devices.push((name, dev));
                }
            }
        }
    }
    devices
}

fn find_input_device(host: &Host, name: Option<&str>) -> Result<Device> {
    if let Some(name) = name {
        if let Ok(devs) = host.input_devices() {
            for dev in devs {
                if dev.name().ok().as_deref() == Some(name) {
                    return Ok(dev);
                }
            }
        }
        bail!("Input device '{}' not found", name);
    }
    host.default_input_device()
        .context("No default input device")
}

fn find_output_device(host: &Host, name: Option<&str>) -> Result<Device> {
    if let Some(name) = name {
        if let Ok(devs) = host.output_devices() {
            for dev in devs {
                if dev.name().ok().as_deref() == Some(name) {
                    return Ok(dev);
                }
            }
        }
        bail!("Output device '{}' not found", name);
    }
    host.default_output_device()
        .context("No default output device")
}

fn supports_f32_input(dev: &Device) -> bool {
    dev.supported_input_configs()
        .map(|cfgs| {
            cfgs.into_iter()
                .any(|c| c.sample_format() == SampleFormat::F32)
        })
        .unwrap_or(false)
}

fn supports_f32_output(dev: &Device) -> bool {
    dev.supported_output_configs()
        .map(|cfgs| {
            cfgs.into_iter()
                .any(|c| c.sample_format() == SampleFormat::F32)
        })
        .unwrap_or(false)
}
