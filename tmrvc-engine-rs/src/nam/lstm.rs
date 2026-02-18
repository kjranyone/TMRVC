use anyhow::{bail, Result};

use super::parse::LstmConfig;
use super::NamModel;

// ---------------------------------------------------------------------------
// LSTM cell
// ---------------------------------------------------------------------------

struct LstmCell {
    // Combined weights: W_ih [4H, I] and W_hh [4H, H], stored separately
    w_ih: Vec<f32>, // [4*H, input_size]
    w_hh: Vec<f32>, // [4*H, hidden_size]
    bias: Vec<f32>, // [4*H] (sum of bias_ih + bias_hh)
    input_size: usize,
    hidden_size: usize,
    // State
    h: Vec<f32>, // [H]
    c: Vec<f32>, // [H]
    // Scratch
    gates: Vec<f32>, // [4*H]
}

impl LstmCell {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            w_ih: vec![0.0; 4 * hidden_size * input_size],
            w_hh: vec![0.0; 4 * hidden_size * hidden_size],
            bias: vec![0.0; 4 * hidden_size],
            input_size,
            hidden_size,
            h: vec![0.0; hidden_size],
            c: vec![0.0; hidden_size],
            gates: vec![0.0; 4 * hidden_size],
        }
    }

    /// Process one time step. Input [input_size], updates h and c in place.
    /// Returns a reference to the hidden state h.
    fn process(&mut self, input: &[f32]) -> &[f32] {
        let h_size = self.hidden_size;

        // gates = W_ih * input + W_hh * h + bias
        for g in 0..4 * h_size {
            let mut sum = self.bias[g];
            // W_ih * input
            let w_ih_row = g * self.input_size;
            for i in 0..self.input_size {
                sum += self.w_ih[w_ih_row + i] * input[i];
            }
            // W_hh * h
            let w_hh_row = g * h_size;
            for j in 0..h_size {
                sum += self.w_hh[w_hh_row + j] * self.h[j];
            }
            self.gates[g] = sum;
        }

        // Gate decomposition (PyTorch order: i, f, g, o)
        for j in 0..h_size {
            let i_gate = sigmoid(self.gates[j]);
            let f_gate = sigmoid(self.gates[h_size + j]);
            let g_gate = self.gates[2 * h_size + j].tanh();
            let o_gate = sigmoid(self.gates[3 * h_size + j]);

            self.c[j] = f_gate * self.c[j] + i_gate * g_gate;
            self.h[j] = o_gate * self.c[j].tanh();
        }

        &self.h
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// LSTM model (supports both LSTM and CatLSTM)
// ---------------------------------------------------------------------------

pub(crate) struct Lstm {
    cells: Vec<LstmCell>,
    head_weight: Vec<f32>, // [1, head_input_size]
    head_bias: Vec<f32>,   // [1]
    hidden_size: usize,
    head_input_size: usize, // H for LSTM, num_layers*H for CatLSTM
    sample_rate: u32,
    cat_mode: bool,
    // Scratch
    cat_hidden: Vec<f32>, // [head_input_size] for CatLSTM
}

impl Lstm {
    /// Construct an LSTM from parsed config and flat weight array.
    ///
    /// Weight unpacking order (PyTorch nn.LSTM parameter order):
    /// ```text
    /// for each layer k:
    ///   weight_ih: [4*H, I_k] (I_0=input_size, I_{k>0}=H)
    ///   weight_hh: [4*H, H]
    ///   bias_ih:   [4*H]
    ///   bias_hh:   [4*H]
    /// head_weight: [1, head_input_size]
    /// head_bias:   [1]
    /// ```
    pub fn from_weights(
        config: &LstmConfig,
        weights: &[f32],
        sample_rate: u32,
        cat_mode: bool,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let n_layers = config.num_layers;
        let head_input_size = if cat_mode { n_layers * h } else { h };

        // Calculate expected weight count
        let mut expected = 0usize;
        for k in 0..n_layers {
            let inp = if k == 0 { config.input_size } else { h };
            expected += 4 * h * inp; // w_ih
            expected += 4 * h * h; // w_hh
            expected += 4 * h; // bias_ih
            expected += 4 * h; // bias_hh
        }
        expected += head_input_size; // head_weight
        expected += 1; // head_bias

        if weights.len() < expected {
            bail!(
                "LSTM: expected at least {} weights, got {}",
                expected,
                weights.len()
            );
        }

        let mut cells = Vec::with_capacity(n_layers);
        let mut offset = 0;

        for k in 0..n_layers {
            let inp = if k == 0 { config.input_size } else { h };
            let mut cell = LstmCell::new(inp, h);

            // weight_ih [4*H, inp]
            let n_wih = 4 * h * inp;
            cell.w_ih.copy_from_slice(&weights[offset..offset + n_wih]);
            offset += n_wih;

            // weight_hh [4*H, H]
            let n_whh = 4 * h * h;
            cell.w_hh.copy_from_slice(&weights[offset..offset + n_whh]);
            offset += n_whh;

            // bias_ih [4*H]
            let bias_ih = &weights[offset..offset + 4 * h];
            offset += 4 * h;

            // bias_hh [4*H]
            let bias_hh = &weights[offset..offset + 4 * h];
            offset += 4 * h;

            // Combined bias = bias_ih + bias_hh
            for j in 0..4 * h {
                cell.bias[j] = bias_ih[j] + bias_hh[j];
            }

            cells.push(cell);
        }

        // Head
        let mut head_weight = vec![0.0f32; head_input_size];
        head_weight.copy_from_slice(&weights[offset..offset + head_input_size]);
        offset += head_input_size;

        let head_bias = vec![weights[offset]];

        Ok(Self {
            cells,
            head_weight,
            head_bias,
            hidden_size: h,
            head_input_size,
            sample_rate,
            cat_mode,
            cat_hidden: vec![0.0; head_input_size],
        })
    }

    fn process_sample(&mut self, input: f32) -> f32 {
        let h = self.hidden_size;
        let input_buf = [input];
        let mut layer_input: &[f32] = &input_buf;

        if self.cat_mode {
            // CatLSTM: concatenate hidden states from all layers
            for (k, cell) in self.cells.iter_mut().enumerate() {
                let _h_out = cell.process(layer_input);
                // Copy hidden state to cat_hidden
                let dst_start = k * h;
                self.cat_hidden[dst_start..dst_start + h].copy_from_slice(&cell.h);
                layer_input = &cell.h;
            }

            // Head: linear projection of concatenated hidden states
            let mut output = self.head_bias[0];
            for i in 0..self.head_input_size {
                output += self.head_weight[i] * self.cat_hidden[i];
            }
            output
        } else {
            // Standard LSTM: only use last layer's hidden state
            for cell in self.cells.iter_mut() {
                let _ = cell.process(layer_input);
                layer_input = &cell.h;
            }

            // Head: linear projection of last hidden state
            let last_h = &self.cells.last().unwrap().h;
            let mut output = self.head_bias[0];
            for i in 0..h {
                output += self.head_weight[i] * last_h[i];
            }
            output
        }
    }
}

impl NamModel for Lstm {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        for i in 0..input.len() {
            output[i] = self.process_sample(input[i]);
        }
    }

    fn reset(&mut self) {
        for cell in &mut self.cells {
            cell.reset();
        }
        self.cat_hidden.fill(0.0);
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
    fn test_lstm_cell_basic() {
        // Simple 1-input, 2-hidden cell with known weights
        let mut cell = LstmCell::new(1, 2);
        // Set all weights to small values so gates are near 0.5
        cell.w_ih.fill(0.1);
        cell.w_hh.fill(0.1);
        cell.bias.fill(0.0);

        let out = cell.process(&[1.0]);
        assert_eq!(out.len(), 2);
        // Output should be finite and non-zero
        for &v in out {
            assert!(v.is_finite());
            assert!(v != 0.0);
        }
    }

    #[test]
    fn test_lstm_cell_reset() {
        let mut cell = LstmCell::new(1, 4);
        cell.w_ih.fill(0.01);
        cell.w_hh.fill(0.01);
        cell.bias.fill(0.0);

        // Process a few steps
        for _ in 0..5 {
            cell.process(&[1.0]);
        }

        // Save state
        let h_before: Vec<f32> = cell.h.clone();
        assert!(h_before.iter().any(|&v| v != 0.0));

        // Reset
        cell.reset();
        assert!(cell.h.iter().all(|&v| v == 0.0));
        assert!(cell.c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lstm_model_standard() {
        let config = LstmConfig {
            num_layers: 2,
            input_size: 1,
            hidden_size: 4,
        };

        let h = 4;

        // Build weight array in PyTorch LSTM order
        let mut weights = Vec::new();

        // Layer 0: input_size=1
        weights.extend(vec![0.01f32; 4 * h * 1]); // w_ih
        weights.extend(vec![0.01f32; 4 * h * h]); // w_hh
        weights.extend(vec![0.0f32; 4 * h]); // bias_ih
        weights.extend(vec![0.0f32; 4 * h]); // bias_hh

        // Layer 1: input_size=H
        weights.extend(vec![0.01f32; 4 * h * h]); // w_ih
        weights.extend(vec![0.01f32; 4 * h * h]); // w_hh
        weights.extend(vec![0.0f32; 4 * h]); // bias_ih
        weights.extend(vec![0.0f32; 4 * h]); // bias_hh

        // Head: [1, H]
        weights.extend(vec![0.5f32; h]); // weight
        weights.push(0.0); // bias

        let mut model = Lstm::from_weights(&config, &weights, 48000, false).unwrap();

        let input = vec![0.5f32; 10];
        let mut output = vec![0.0f32; 10];
        model.process(&input, &mut output);

        // Output should not be all zeros
        assert!(output.iter().any(|&v| v != 0.0));
        // All finite
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lstm_cat_mode() {
        let config = LstmConfig {
            num_layers: 2,
            input_size: 1,
            hidden_size: 4,
        };
        let h = 4;

        let mut weights = Vec::new();

        // Layer 0
        weights.extend(vec![0.01f32; 4 * h * 1]);
        weights.extend(vec![0.01f32; 4 * h * h]);
        weights.extend(vec![0.0f32; 4 * h]);
        weights.extend(vec![0.0f32; 4 * h]);
        // Layer 1
        weights.extend(vec![0.01f32; 4 * h * h]);
        weights.extend(vec![0.01f32; 4 * h * h]);
        weights.extend(vec![0.0f32; 4 * h]);
        weights.extend(vec![0.0f32; 4 * h]);

        // Head: [1, num_layers * H] = [1, 8]
        weights.extend(vec![0.5f32; 2 * h]);
        weights.push(0.0);

        let mut model = Lstm::from_weights(&config, &weights, 48000, true).unwrap();
        assert_eq!(model.head_input_size, 8);

        let mut output = vec![0.0f32; 5];
        model.process(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut output);

        assert!(output.iter().any(|&v| v != 0.0));
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lstm_reset_deterministic() {
        let config = LstmConfig {
            num_layers: 1,
            input_size: 1,
            hidden_size: 4,
        };
        let h = 4;
        let mut weights = Vec::new();
        weights.extend(vec![0.05f32; 4 * h * 1]);
        weights.extend(vec![0.05f32; 4 * h * h]);
        weights.extend(vec![0.0f32; 4 * h]);
        weights.extend(vec![0.0f32; 4 * h]);
        weights.extend(vec![1.0f32; h]);
        weights.push(0.0);

        let mut model = Lstm::from_weights(&config, &weights, 48000, false).unwrap();

        let input = [0.5, 1.0, -0.5, 0.3, 0.7];
        let mut out1 = vec![0.0f32; 5];
        model.process(&input, &mut out1);

        model.reset();

        let mut out2 = vec![0.0f32; 5];
        model.process(&input, &mut out2);

        for i in 0..5 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i,
                out1[i],
                out2[i]
            );
        }
    }
}
