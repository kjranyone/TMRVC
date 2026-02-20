"""Tests for ConverterStudent, ConverterStudentGTM, ConverterStudentHQ."""

import torch
import pytest

from tmrvc_core.constants import (
    CONVERTER_STATE_FRAMES,
    D_CONTENT,
    D_CONVERTER_HIDDEN,
    D_SPEAKER,
    D_VOCODER_FEATURES,
    N_ACOUSTIC_PARAMS,
)
from tmrvc_train.models.converter import (
    ConverterStudent,
    ConverterStudentGTM,
    ConverterStudentHQ,
)


class TestConverterStudent:
    """Tests for ConverterStudent."""

    @pytest.fixture
    def model(self):
        return ConverterStudent()

    def test_training_mode_shape(self, model):
        content = torch.randn(2, D_CONTENT, 50)
        spk = torch.randn(2, D_SPEAKER)
        ir = torch.randn(2, N_ACOUSTIC_PARAMS)
        pred, state = model(content, spk, ir)
        assert pred.shape == (2, D_VOCODER_FEATURES, 50)
        assert state is None

    def test_streaming_mode_shape(self, model):
        content = torch.randn(1, D_CONTENT, 1)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)
        state_in = model.init_state()

        pred, state_out = model(content, spk, ir, state_in)
        assert pred.shape == (1, D_VOCODER_FEATURES, 1)
        assert state_out.shape == (1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES)

    def test_state_size_matches_contract(self, model):
        assert model._total_state == CONVERTER_STATE_FRAMES  # 52

    def test_film_conditioning(self, model):
        """Verify that different speaker embeddings produce different outputs."""
        model.eval()
        content = torch.randn(1, D_CONTENT, 5)
        ir = torch.zeros(1, N_ACOUSTIC_PARAMS)

        spk1 = torch.randn(1, D_SPEAKER)
        spk2 = torch.randn(1, D_SPEAKER)

        with torch.no_grad():
            out1, _ = model(content, spk1, ir)
            out2, _ = model(content, spk2, ir)

        assert not torch.allclose(out1, out2)

    def test_streaming_matches_training(self, model):
        model.eval()
        T = 10
        content = torch.randn(1, D_CONTENT, T)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)

        with torch.no_grad():
            out_full, _ = model(content, spk, ir)

        state = model.init_state()
        frames = []
        with torch.no_grad():
            for t in range(T):
                c = content[:, :, t:t+1]
                f, state = model(c, spk, ir, state)
                frames.append(f)

        out_streaming = torch.cat(frames, dim=-1)
        torch.testing.assert_close(out_full, out_streaming, atol=1e-5, rtol=1e-4)


class TestConverterStudentGTM:
    """Tests for ConverterStudentGTM with Global Timbre Memory."""

    @pytest.fixture
    def model(self):
        return ConverterStudentGTM()

    def test_converter_gtm_training_shape(self, model):
        """Output shape should match ConverterStudent."""
        content = torch.randn(2, D_CONTENT, 50)
        spk = torch.randn(2, D_SPEAKER)
        ir = torch.randn(2, N_ACOUSTIC_PARAMS)
        pred, state = model(content, spk, ir)
        assert pred.shape == (2, D_VOCODER_FEATURES, 50)
        assert state is None

    def test_converter_gtm_streaming_shape(self, model):
        """State shape should be [1, 384, 52] matching ConverterStudent."""
        content = torch.randn(1, D_CONTENT, 1)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)
        state_in = model.init_state()

        pred, state_out = model(content, spk, ir, state_in)
        assert pred.shape == (1, D_VOCODER_FEATURES, 1)
        assert state_out.shape == (1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES)

    def test_converter_gtm_streaming_matches_training(self, model):
        """Frame-by-frame streaming should match full-sequence training mode."""
        model.eval()
        T = 10
        content = torch.randn(1, D_CONTENT, T)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)

        with torch.no_grad():
            out_full, _ = model(content, spk, ir)

        state = model.init_state()
        frames = []
        with torch.no_grad():
            for t in range(T):
                c = content[:, :, t:t+1]
                f, state = model(c, spk, ir, state)
                frames.append(f)

        out_streaming = torch.cat(frames, dim=-1)
        torch.testing.assert_close(out_full, out_streaming, atol=1e-5, rtol=1e-4)

    def test_converter_gtm_different_speakers(self, model):
        """Different speaker embeddings should produce different outputs."""
        model.eval()
        content = torch.randn(1, D_CONTENT, 5)
        ir = torch.zeros(1, N_ACOUSTIC_PARAMS)

        spk1 = torch.randn(1, D_SPEAKER)
        spk2 = torch.randn(1, D_SPEAKER)

        with torch.no_grad():
            out1, _ = model(content, spk1, ir)
            out2, _ = model(content, spk2, ir)

        assert not torch.allclose(out1, out2)

    def test_fewshot_adapter_with_gtm(self, model):
        """LoRA should target GTM projection layer."""
        from tmrvc_train.fewshot import FewShotAdapterGTM

        adapter = FewShotAdapterGTM(model)
        params = adapter.get_lora_parameters()
        assert len(params) == 2  # lora_A and lora_B
        delta = adapter.get_lora_delta_flat()
        assert delta.dim() == 1
        assert delta.numel() > 0


class TestConverterStudentHQ:
    """Tests for ConverterStudentHQ (semi-causal HQ mode)."""

    @pytest.fixture
    def model(self):
        return ConverterStudentHQ()

    def test_converter_hq_training_shape(self, model):
        """Training mode: T=50 → T=50 output (padding preserves length)."""
        content = torch.randn(2, D_CONTENT, 50)
        spk = torch.randn(2, D_SPEAKER)
        ir = torch.randn(2, N_ACOUSTIC_PARAMS)
        pred, state = model(content, spk, ir)
        assert pred.shape == (2, D_VOCODER_FEATURES, 50)
        assert state is None

    def test_converter_hq_streaming_shape(self, model):
        """Streaming: T=7 + state[46] → T=1 output."""
        T_in = 1 + model.max_lookahead  # 7
        content = torch.randn(1, D_CONTENT, T_in)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)
        state_in = model.init_state()

        pred, state_out = model(content, spk, ir, state_in)
        assert pred.shape == (1, D_VOCODER_FEATURES, 1)
        assert state_out.shape == (1, D_CONVERTER_HIDDEN, model._total_state)

    def test_converter_hq_state_size(self, model):
        """Total HQ state should be 46 frames."""
        assert model._total_state == 46

    def test_converter_hq_streaming_matches_training(self, model):
        """Frame-by-frame streaming (T=7 window) matches full sequence."""
        model.eval()
        L = model.max_lookahead  # 6
        T = 20
        content = torch.randn(1, D_CONTENT, T)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)

        # Training: full sequence
        with torch.no_grad():
            out_full, _ = model(content, spk, ir)

        # Streaming: slide a window of size 1+L, advancing 1 frame at a time
        state = model.init_state()
        frames = []
        with torch.no_grad():
            for t in range(T - L):
                c = content[:, :, t:t + 1 + L]  # [1, 256, 7]
                f, state = model(c, spk, ir, state)
                frames.append(f)

        out_streaming = torch.cat(frames, dim=-1)
        # Streaming output corresponds to training output at positions [0..T-L-1]
        torch.testing.assert_close(
            out_full[:, :, :T - L], out_streaming, atol=1e-5, rtol=1e-4,
        )

    def test_converter_hq_from_causal(self, model):
        """from_causal() should successfully load causal weights."""
        causal = ConverterStudent()
        hq = ConverterStudentHQ.from_causal(causal)
        # Verify same number of blocks
        assert len(hq.blocks) == len(causal.blocks)
        # Verify weight shapes match
        for hq_block, causal_block in zip(hq.blocks, causal.blocks):
            assert (
                hq_block.conv_block.dwconv.weight.shape
                == causal_block.conv_block.dwconv.weight.shape
            )

    def test_converter_hq_different_from_causal(self):
        """HQ and causal should produce different outputs for same input."""
        torch.manual_seed(42)
        causal = ConverterStudent()
        hq = ConverterStudentHQ.from_causal(causal)
        causal.eval()
        hq.eval()

        content = torch.randn(1, D_CONTENT, 20)
        spk = torch.randn(1, D_SPEAKER)
        ir = torch.randn(1, N_ACOUSTIC_PARAMS)

        with torch.no_grad():
            out_causal, _ = causal(content, spk, ir)
            out_hq, _ = hq(content, spk, ir)

        # Should differ because HQ uses right-context (lookahead)
        assert not torch.allclose(out_causal, out_hq)
