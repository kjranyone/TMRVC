use anyhow::{bail, Result};

use super::parse::WaveNetConfig;
use super::NamModel;

// ---------------------------------------------------------------------------
// Dilated causal 1D convolution with ring-buffer state
// ---------------------------------------------------------------------------

struct DilatedConv {
    weight: Vec<f32>,  // [out_ch, in_ch, kernel_size] row-major
    bias: Vec<f32>,    // [out_ch]
    history: Vec<f32>, // [buf_len * in_ch]
    hist_pos: usize,
    buf_len: usize,    // (kernel_size - 1) * dilation + 1
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    dilation: usize,
}

impl DilatedConv {
    fn new(in_ch: usize, out_ch: usize, kernel_size: usize, dilation: usize) -> Self {
        let buf_len = (kernel_size - 1) * dilation + 1;
        Self {
            weight: vec![0.0; out_ch * in_ch * kernel_size],
            bias: vec![0.0; out_ch],
            history: vec![0.0; buf_len * in_ch],
            hist_pos: 0,
            buf_len,
            in_ch,
            out_ch,
            kernel_size,
            dilation,
        }
    }

    /// Process one time step: push `input` [in_ch] into ring buffer, compute output [out_ch].
    ///
    /// Follows NeuralAmpModelerCore convention: kernel tap k=0 reads current sample,
    /// k=K-1 reads the oldest sample at (K-1)*dilation steps ago.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_ch);
        debug_assert_eq!(output.len(), self.out_ch);

        // Write current input into ring buffer
        let start = self.hist_pos * self.in_ch;
        self.history[start..start + self.in_ch].copy_from_slice(input);

        // Compute output
        for o in 0..self.out_ch {
            let mut sum = self.bias[o];
            for k in 0..self.kernel_size {
                let delay = k * self.dilation;
                let read_pos = (self.hist_pos + self.buf_len - delay) % self.buf_len;
                let read_start = read_pos * self.in_ch;
                let w_base = (o * self.in_ch) * self.kernel_size + k;
                for i in 0..self.in_ch {
                    sum += self.weight[w_base + i * self.kernel_size]
                        * self.history[read_start + i];
                }
            }
            output[o] = sum;
        }

        // Advance write position
        self.hist_pos = (self.hist_pos + 1) % self.buf_len;
    }

    fn reset(&mut self) {
        self.history.fill(0.0);
        self.hist_pos = 0;
    }

    /// Number of weight + bias floats consumed from the flat weight array.
    fn weight_count(&self) -> usize {
        self.out_ch * self.in_ch * self.kernel_size + self.out_ch
    }
}

// ---------------------------------------------------------------------------
// WaveNet layer: dilated conv → gated activation → 1×1 (residual + skip)
// ---------------------------------------------------------------------------

struct WaveNetLayer {
    conv: DilatedConv,
    // Input mixin 1×1: raw input (1 sample) → gate_channels
    mixin_weight: Vec<f32>, // [gate_ch]
    mixin_bias: Vec<f32>,   // [gate_ch]
    // 1×1 output: activation (channels) → 2*channels (residual + skip)
    out_weight: Vec<f32>, // [2*ch, ch]
    out_bias: Vec<f32>,   // [2*ch]
    channels: usize,
    gate_channels: usize,
    gated: bool,
    // Pre-allocated scratch
    z: Vec<f32>,             // [gate_ch]
    activation_buf: Vec<f32>, // [ch]
    out_1x1: Vec<f32>,      // [2*ch]
}

impl WaveNetLayer {
    fn new(channels: usize, kernel_size: usize, dilation: usize, gated: bool) -> Self {
        let gate_ch = if gated { 2 * channels } else { channels };
        Self {
            conv: DilatedConv::new(channels, gate_ch, kernel_size, dilation),
            mixin_weight: vec![0.0; gate_ch],
            mixin_bias: vec![0.0; gate_ch],
            out_weight: vec![0.0; 2 * channels * channels],
            out_bias: vec![0.0; 2 * channels],
            channels,
            gate_channels: gate_ch,
            gated,
            z: vec![0.0; gate_ch],
            activation_buf: vec![0.0; channels],
            out_1x1: vec![0.0; 2 * channels],
        }
    }

    /// Process one sample through this layer.
    /// `raw_input`: the original audio sample (scalar).
    /// `condition`: [channels] — modified in-place (residual added).
    /// `skip_accum`: [channels] — skip connection accumulated.
    fn process_sample(
        &mut self,
        raw_input: f32,
        condition: &mut [f32],
        skip_accum: &mut [f32],
    ) {
        let ch = self.channels;

        // 1. Dilated conv on condition
        self.conv.process(condition, &mut self.z);

        // 2. Input mixin: add scaled raw_input to z
        for c in 0..self.gate_channels {
            self.z[c] += self.mixin_weight[c] * raw_input + self.mixin_bias[c];
        }

        // 3. Gated activation
        if self.gated {
            for c in 0..ch {
                let t = self.z[c].tanh();
                let g = sigmoid(self.z[c + ch]);
                self.activation_buf[c] = t * g;
            }
        } else {
            for c in 0..ch {
                self.activation_buf[c] = self.z[c].tanh();
            }
        }

        // 4. 1×1 conv: activation [ch] → out [2*ch]
        for o in 0..2 * ch {
            let mut sum = self.out_bias[o];
            for i in 0..ch {
                sum += self.out_weight[o * ch + i] * self.activation_buf[i];
            }
            self.out_1x1[o] = sum;
        }

        // 5. Residual + skip
        for c in 0..ch {
            condition[c] += self.out_1x1[c];
            skip_accum[c] += self.out_1x1[c + ch];
        }
    }

    fn reset(&mut self) {
        self.conv.reset();
    }
}

// ---------------------------------------------------------------------------
// Layer array: rechannel → N layers
// ---------------------------------------------------------------------------

struct LayerArray {
    rechannel_weight: Vec<f32>, // [channels, input_ch]
    rechannel_bias: Vec<f32>,   // [channels]
    input_ch: usize,
    channels: usize,
    layers: Vec<WaveNetLayer>,
}

impl LayerArray {
    fn new(
        input_ch: usize,
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        gated: bool,
    ) -> Self {
        let layers = dilations
            .iter()
            .map(|&d| WaveNetLayer::new(channels, kernel_size, d, gated))
            .collect();
        Self {
            rechannel_weight: vec![0.0; channels * input_ch],
            rechannel_bias: vec![0.0; channels],
            input_ch,
            channels,
            layers,
        }
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// WaveNet top-level model
// ---------------------------------------------------------------------------

pub(crate) struct WaveNet {
    layer_arrays: Vec<LayerArray>,
    head_rechannel_weight: Vec<f32>, // [head_size, channels]
    head_rechannel_bias: Vec<f32>,   // [head_size]
    head_output_weight: Vec<f32>,    // [1, head_size]
    head_output_bias: Vec<f32>,      // [1] (may be zero if !head_bias)
    head_scale: f32,
    channels: usize,
    head_size: usize,
    sample_rate: u32,
    // Pre-allocated scratch
    condition: Vec<f32>,     // [channels]
    prev_condition: Vec<f32>, // [channels] (for multi-block)
    skip_accum: Vec<f32>,    // [channels]
    head_hidden: Vec<f32>,   // [head_size]
}

impl WaveNet {
    /// Construct a WaveNet from parsed config and flat weight array.
    ///
    /// Weight unpacking follows NeuralAmpModelerCore convention:
    /// ```text
    /// for each block:
    ///   rechannel 1×1: weight [ch, in_ch] + bias [ch]
    ///   for each layer:
    ///     dilated_conv: weight [gate_ch, ch, K] + bias [gate_ch]
    ///     input_mixin:  weight [gate_ch, 1] + bias [gate_ch]
    ///     1×1 out:      weight [2*ch, ch] + bias [2*ch]
    /// head_rechannel: weight [head_size, ch] + bias [head_size]
    /// head_output:    weight [1, head_size] + bias [1] (if head_bias)
    /// head_scale:     1 float
    /// ```
    pub fn from_weights(config: &WaveNetConfig, weights: &[f32], sample_rate: u32) -> Result<Self> {
        let ch = config.channels;
        let head_size = config.head_size;
        let ks = config.kernel_size;
        let gated = config.gated;
        let gate_ch = if gated { 2 * ch } else { ch };

        // Build layer arrays
        let mut layer_arrays = Vec::with_capacity(config.num_blocks);
        for block_idx in 0..config.num_blocks {
            let input_ch = if block_idx == 0 {
                config.condition_size
            } else {
                ch
            };
            layer_arrays.push(LayerArray::new(
                input_ch,
                ch,
                ks,
                &config.dilations,
                gated,
            ));
        }

        // Calculate expected weight count
        let mut expected = 0usize;
        for array in &layer_arrays {
            // rechannel
            expected += array.channels * array.input_ch + array.channels;
            for layer in &array.layers {
                // dilated conv
                expected += layer.conv.weight_count();
                // input mixin
                expected += gate_ch + gate_ch;
                // 1x1 out
                expected += 2 * ch * ch + 2 * ch;
            }
        }
        // head
        expected += head_size * ch + head_size; // head_rechannel
        expected += head_size; // head_output weight
        if config.head_bias {
            expected += 1; // head_output bias
        }
        expected += 1; // head_scale

        if weights.len() < expected {
            bail!(
                "WaveNet: expected at least {} weights, got {}",
                expected,
                weights.len()
            );
        }

        // Unpack weights
        let mut offset = 0;

        for array in &mut layer_arrays {
            // Rechannel 1×1
            let n = array.channels * array.input_ch;
            array.rechannel_weight.copy_from_slice(&weights[offset..offset + n]);
            offset += n;
            array
                .rechannel_bias
                .copy_from_slice(&weights[offset..offset + array.channels]);
            offset += array.channels;

            for layer in &mut array.layers {
                // Dilated conv weight + bias
                let conv = &mut layer.conv;
                let n_w = conv.out_ch * conv.in_ch * conv.kernel_size;
                conv.weight.copy_from_slice(&weights[offset..offset + n_w]);
                offset += n_w;
                conv.bias
                    .copy_from_slice(&weights[offset..offset + conv.out_ch]);
                offset += conv.out_ch;

                // Input mixin 1×1
                layer
                    .mixin_weight
                    .copy_from_slice(&weights[offset..offset + gate_ch]);
                offset += gate_ch;
                layer
                    .mixin_bias
                    .copy_from_slice(&weights[offset..offset + gate_ch]);
                offset += gate_ch;

                // 1×1 out (residual + skip)
                let n_out = 2 * ch * ch;
                layer
                    .out_weight
                    .copy_from_slice(&weights[offset..offset + n_out]);
                offset += n_out;
                layer
                    .out_bias
                    .copy_from_slice(&weights[offset..offset + 2 * ch]);
                offset += 2 * ch;
            }
        }

        // Head rechannel
        let mut head_rechannel_weight = vec![0.0f32; head_size * ch];
        let mut head_rechannel_bias = vec![0.0f32; head_size];
        head_rechannel_weight.copy_from_slice(&weights[offset..offset + head_size * ch]);
        offset += head_size * ch;
        head_rechannel_bias.copy_from_slice(&weights[offset..offset + head_size]);
        offset += head_size;

        // Head output
        let mut head_output_weight = vec![0.0f32; head_size];
        let mut head_output_bias = vec![0.0f32; 1];
        head_output_weight.copy_from_slice(&weights[offset..offset + head_size]);
        offset += head_size;
        if config.head_bias {
            head_output_bias[0] = weights[offset];
            offset += 1;
        }

        // Head scale
        let head_scale = weights[offset];

        Ok(Self {
            layer_arrays,
            head_rechannel_weight,
            head_rechannel_bias,
            head_output_weight,
            head_output_bias,
            head_scale,
            channels: ch,
            head_size,
            sample_rate,
            condition: vec![0.0; ch],
            prev_condition: vec![0.0; ch],
            skip_accum: vec![0.0; ch],
            head_hidden: vec![0.0; head_size],
        })
    }

    /// Process a single audio sample and return the output sample.
    fn process_sample(&mut self, input: f32) -> f32 {
        let ch = self.channels;

        // Reset skip accumulator
        self.skip_accum.fill(0.0);

        // Process layer arrays
        let num_blocks = self.layer_arrays.len();
        for block_idx in 0..num_blocks {
            let prev = if block_idx == 0 {
                None
            } else {
                // Copy condition to prev_condition to avoid aliasing
                self.prev_condition.copy_from_slice(&self.condition);
                Some(self.prev_condition.as_slice())
            };

            // Split borrow: take layer_array mutably, pass skip_accum and condition
            // We need to use raw indexing to avoid borrow checker issues
            let array = &mut self.layer_arrays[block_idx];

            if let Some(prev) = prev {
                // Rechannel from prev_condition
                for o in 0..ch {
                    let mut sum = array.rechannel_bias[o];
                    for i in 0..array.input_ch {
                        sum += array.rechannel_weight[o * array.input_ch + i] * prev[i];
                    }
                    self.condition[o] = sum;
                }
            } else {
                // Rechannel from scalar input
                for o in 0..ch {
                    self.condition[o] = array.rechannel_weight[o * array.input_ch] * input
                        + array.rechannel_bias[o];
                }
            }

            // Process layers
            for layer in &mut array.layers {
                layer.process_sample(input, &mut self.condition, &mut self.skip_accum);
            }
        }

        // Head: rechannel → tanh → output → scale
        for o in 0..self.head_size {
            let mut sum = self.head_rechannel_bias[o];
            for i in 0..ch {
                sum += self.head_rechannel_weight[o * ch + i] * self.skip_accum[i];
            }
            self.head_hidden[o] = sum.tanh();
        }

        let mut output = self.head_output_bias[0];
        for i in 0..self.head_size {
            output += self.head_output_weight[i] * self.head_hidden[i];
        }

        output * self.head_scale
    }
}

impl NamModel for WaveNet {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            output[i] = self.process_sample(input[i]);
        }
    }

    fn reset(&mut self) {
        for array in &mut self.layer_arrays {
            array.reset();
        }
        self.condition.fill(0.0);
        self.prev_condition.fill(0.0);
        self.skip_accum.fill(0.0);
        self.head_hidden.fill(0.0);
    }

    fn expected_sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dilated_conv_basic() {
        // 1 input channel, 1 output channel, kernel_size=2, dilation=1
        // buf_len = (2-1)*1 + 1 = 2
        let mut conv = DilatedConv::new(1, 1, 2, 1);
        // weight[0,0,0]=1.0 (k=0, current), weight[0,0,1]=2.0 (k=1, 1 step ago)
        conv.weight = vec![1.0, 2.0];
        conv.bias = vec![0.5];

        let mut out = vec![0.0];

        // Step 1: input=3.0, history=[0,0] → current=3.0, 1-ago=0.0
        // output = 0.5 + 1.0*3.0 + 2.0*0.0 = 3.5
        conv.process(&[3.0], &mut out);
        assert!((out[0] - 3.5).abs() < 1e-6);

        // Step 2: input=5.0, current=5.0, 1-ago=3.0
        // output = 0.5 + 1.0*5.0 + 2.0*3.0 = 11.5
        conv.process(&[5.0], &mut out);
        assert!((out[0] - 11.5).abs() < 1e-6);
    }

    #[test]
    fn test_dilated_conv_dilation2() {
        // 1 in, 1 out, kernel_size=2, dilation=2
        // buf_len = (2-1)*2 + 1 = 3
        let mut conv = DilatedConv::new(1, 1, 2, 2);
        conv.weight = vec![1.0, 1.0]; // k=0 (current) + k=1 (2 steps ago)
        conv.bias = vec![0.0];

        let mut out = vec![0.0];

        // Step 1: input=1, hist=[0,0,0] → 1 + 0 = 1
        conv.process(&[1.0], &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);

        // Step 2: input=2 → 2 + 0 = 2 (2 steps ago = initial 0)
        conv.process(&[2.0], &mut out);
        assert!((out[0] - 2.0).abs() < 1e-6);

        // Step 3: input=3 → 3 + 1 = 4 (2 steps ago = step 1's input=1)
        conv.process(&[3.0], &mut out);
        assert!((out[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dilated_conv_ring_wrap() {
        // kernel_size=3, dilation=1, buf_len=3
        let mut conv = DilatedConv::new(1, 1, 3, 1);
        conv.weight = vec![1.0, 0.0, 0.0]; // only current sample
        conv.bias = vec![0.0];

        let mut out = vec![0.0];

        // Process many steps to ensure ring buffer wraps correctly
        for i in 0..20 {
            let val = i as f32;
            conv.process(&[val], &mut out);
            assert!(
                (out[0] - val).abs() < 1e-6,
                "step {}: expected {}, got {}",
                i,
                val,
                out[0]
            );
        }
    }

    #[test]
    fn test_wavenet_minimal() {
        // Minimal WaveNet: 1 block, 2 channels, 1 layer, kernel_size=2, gated
        let config = WaveNetConfig {
            condition_size: 1,
            channels: 2,
            head_size: 2,
            kernel_size: 2,
            dilations: vec![1],
            gated: true,
            head_bias: true,
            num_blocks: 1,
        };

        // Count expected weights
        let ch = 2;
        let gate_ch = 4; // gated: 2*ch
        let ks = 2;

        let mut weights = Vec::new();

        // Rechannel: [ch, 1] + [ch] → ch + ch = 4
        weights.extend_from_slice(&[0.1, 0.2]); // weight [2, 1]
        weights.extend_from_slice(&[0.0, 0.0]); // bias [2]

        // Layer 0:
        //   dilated_conv: [gate_ch, ch, ks] + [gate_ch] = 4*2*2 + 4 = 20
        weights.extend(vec![0.01f32; gate_ch * ch * ks]); // weight
        weights.extend(vec![0.0f32; gate_ch]); // bias
        //   input_mixin: [gate_ch] + [gate_ch] = 8
        weights.extend(vec![0.01f32; gate_ch]); // weight
        weights.extend(vec![0.0f32; gate_ch]); // bias
        //   1x1 out: [2*ch, ch] + [2*ch] = 8 + 4 = 12
        weights.extend(vec![0.01f32; 2 * ch * ch]); // weight
        weights.extend(vec![0.0f32; 2 * ch]); // bias

        // Head rechannel: [head_size, ch] + [head_size] = 4 + 2 = 6
        weights.extend(vec![0.1f32; 2 * ch]); // weight
        weights.extend(vec![0.0f32; 2]); // bias

        // Head output: [head_size] + [1] = 3
        weights.extend(vec![0.5f32; 2]); // weight
        weights.push(0.0); // bias

        // Head scale
        weights.push(1.0);

        let mut model = WaveNet::from_weights(&config, &weights, 48000).unwrap();

        // Process a few samples
        let input = vec![0.5f32; 10];
        let mut output = vec![0.0f32; 10];
        model.process(&input, &mut output);

        // Output should not be all zeros (model processes input)
        let sum: f32 = output.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "WaveNet output should not be all zeros");

        // No NaN or inf
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "sample {} is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_wavenet_reset() {
        let config = WaveNetConfig {
            condition_size: 1,
            channels: 2,
            head_size: 2,
            kernel_size: 2,
            dilations: vec![1],
            gated: true,
            head_bias: true,
            num_blocks: 1,
        };

        // All-zero weights for simplicity
        let gate_ch = 4;
        let ch = 2;
        let ks = 2;
        let n_weights = (ch + ch)
            + (gate_ch * ch * ks + gate_ch + gate_ch + gate_ch + 2 * ch * ch + 2 * ch)
            + (2 * ch + 2 + 2 + 1 + 1);
        let weights = vec![0.0f32; n_weights];

        let mut model = WaveNet::from_weights(&config, &weights, 48000).unwrap();

        // Process some samples
        let mut out = vec![0.0f32; 5];
        model.process(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out);

        // Reset
        model.reset();

        // Process again — should produce identical results
        let mut out2 = vec![0.0f32; 5];
        model.process(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut out2);

        for i in 0..5 {
            assert!(
                (out[i] - out2[i]).abs() < 1e-10,
                "Reset did not restore state at sample {}",
                i
            );
        }
    }
}
