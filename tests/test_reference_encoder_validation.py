"""Worker 06: ReferenceEncoder validation tests.

Tests:
- Speaker-agnosticism: same text, different speakers -> similar prosody latents
- Prosody discriminability: same speaker, different prosody -> different latents
- Output shape [B, d_prosody] and gradient flow
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tmrvc_train.models.reference_encoder import (
    ReferenceEncoder,
    ReferenceEncoderFromWaveform,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_PROSODY = 64
N_MELS = 80
T_MEL = 200  # ~2 seconds of audio at typical hop length


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mel(batch_size: int = 1, t_mel: int = T_MEL) -> torch.Tensor:
    """Create a synthetic mel spectrogram [B, n_mels, T_mel]."""
    return torch.randn(batch_size, N_MELS, t_mel)


def _make_waveform(batch_size: int = 1, duration_samples: int = 24000) -> torch.Tensor:
    """Create a synthetic waveform [B, T_audio]."""
    return torch.randn(batch_size, duration_samples)


# ---------------------------------------------------------------------------
# Output Shape and Basic Contract
# ---------------------------------------------------------------------------

class TestReferenceEncoderOutputShape:
    """ReferenceEncoder must produce [B, d_prosody] outputs."""

    def test_output_shape_single(self):
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        mel = _make_mel(batch_size=1)
        out = enc(mel)
        assert out.shape == (1, D_PROSODY), f"Expected (1, {D_PROSODY}), got {out.shape}"

    def test_output_shape_batched(self):
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        mel = _make_mel(batch_size=4)
        out = enc(mel)
        assert out.shape == (4, D_PROSODY), f"Expected (4, {D_PROSODY}), got {out.shape}"

    def test_output_shape_128(self):
        """Test with d_prosody=128 as mentioned in the plan."""
        enc = ReferenceEncoder(d_prosody=128, n_mels=N_MELS)
        mel = _make_mel(batch_size=2)
        out = enc(mel)
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

    def test_output_finite(self):
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        mel = _make_mel(batch_size=2)
        out = enc(mel)
        assert torch.isfinite(out).all(), "ReferenceEncoder output contains non-finite values"

    def test_4d_input_accepted(self):
        """ReferenceEncoder should accept [B, 1, n_mels, T] 4D input."""
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        mel = _make_mel(batch_size=2).unsqueeze(1)  # [B, 1, n_mels, T]
        out = enc(mel)
        assert out.shape == (2, D_PROSODY)


# ---------------------------------------------------------------------------
# Gradient Flow
# ---------------------------------------------------------------------------

class TestReferenceEncoderGradientFlow:
    """Gradients must flow through the ReferenceEncoder to the ProsodyPredictor."""

    def test_gradient_flows_through_encoder(self):
        """Backward pass from a loss on the encoder output must produce non-zero gradients."""
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        mel = _make_mel(batch_size=2)

        out = enc(mel)
        loss = out.sum()
        loss.backward()

        # Check that gradients are non-zero for at least some parameters
        has_grad = False
        for name, param in enc.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients flowed through ReferenceEncoder"

    def test_gradient_to_cnn_layers(self):
        """CNN layers must receive gradients."""
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        mel = _make_mel(batch_size=1)

        out = enc(mel)
        loss = out.mean()
        loss.backward()

        cnn_grads = []
        for name, param in enc.cnn.named_parameters():
            if param.grad is not None:
                cnn_grads.append(param.grad.abs().sum().item())

        assert len(cnn_grads) > 0, "No CNN parameters received gradients"
        assert any(g > 0 for g in cnn_grads), "All CNN gradients are zero"


# ---------------------------------------------------------------------------
# Speaker-Agnosticism
# ---------------------------------------------------------------------------

class TestSpeakerAgnosticism:
    """Same text (mel), different speakers -> similar prosody latents.

    With random weights this tests the code path, not trained behaviour.
    The structure validates that the encoder processes spectral content
    without an explicit speaker pathway.
    """

    def test_same_mel_produces_identical_latent(self):
        """Identical mel input must produce identical latent (deterministic)."""
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        enc.eval()

        mel = _make_mel(batch_size=1)

        with torch.no_grad():
            out1 = enc(mel)
            out2 = enc(mel)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Same mel input produced different latents"
        )

    def test_similar_mel_produces_similar_latent(self):
        """Mels that differ only slightly should produce similar latents.

        This simulates same-text, different-speaker scenario where
        the spectral envelope differs but prosody is similar.
        """
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        enc.eval()

        mel_base = _make_mel(batch_size=1)
        # Add small perturbation (simulates speaker difference with same prosody)
        mel_perturbed = mel_base + 0.01 * torch.randn_like(mel_base)

        with torch.no_grad():
            out_base = enc(mel_base)
            out_perturbed = enc(mel_perturbed)

        cos_sim = F.cosine_similarity(out_base, out_perturbed, dim=-1).item()
        # Small perturbation should not drastically change the output
        assert cos_sim > 0.5, (
            f"Small mel perturbation caused large latent change (cos_sim={cos_sim:.3f})"
        )

    def test_no_speaker_input_path(self):
        """ReferenceEncoder should not accept speaker embeddings as input.

        The encoder extracts prosody from mel; speaker identity should be
        handled separately by the SpeakerPromptEncoder.
        """
        import inspect
        sig = inspect.signature(ReferenceEncoder.forward)
        params = set(sig.parameters.keys()) - {"self"}
        assert "speaker_embed" not in params, (
            "ReferenceEncoder.forward accepts speaker_embed, which violates "
            "the speaker-agnostic prosody extraction contract"
        )


# ---------------------------------------------------------------------------
# Prosody Discriminability
# ---------------------------------------------------------------------------

class TestProsodyDiscriminability:
    """Same speaker, different prosody -> different latents.

    With random weights, very different mel inputs should produce
    measurably different latents. This validates the code path.
    """

    def test_different_mel_produces_different_latent(self):
        """Substantially different mels must produce different latents."""
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        enc.eval()

        mel1 = _make_mel(batch_size=1)
        mel2 = _make_mel(batch_size=1)  # completely different

        with torch.no_grad():
            out1 = enc(mel1)
            out2 = enc(mel2)

        cos_dist = 1.0 - F.cosine_similarity(out1, out2, dim=-1).item()
        # With random weights, latents may be very similar; we just verify
        # they are not bitwise identical (determinism check already covers that)
        assert cos_dist > 1e-6 or not torch.allclose(out1, out2, atol=1e-7), (
            f"Completely different mels produced bitwise identical latents "
            f"(cosine distance={cos_dist:.8f})"
        )

    def test_prosody_variance_across_diverse_inputs(self):
        """A batch of diverse mels should produce latents with measurable variance."""
        enc = ReferenceEncoder(d_prosody=D_PROSODY, n_mels=N_MELS)
        enc.eval()

        # Generate 8 diverse mel inputs
        mels = _make_mel(batch_size=8)

        with torch.no_grad():
            latents = enc(mels)  # [8, d_prosody]

        # Compute variance across the batch
        variance = latents.var(dim=0).mean().item()
        # With random weights the variance may be very small but should not be zero
        assert variance > 1e-10, (
            f"Latent variance across diverse inputs is exactly zero; "
            f"encoder is producing constant output regardless of input"
        )


# ---------------------------------------------------------------------------
# ReferenceEncoderFromWaveform
# ---------------------------------------------------------------------------

class TestReferenceEncoderFromWaveform:
    """Validate the waveform -> mel -> prosody pipeline."""

    def test_output_shape(self):
        enc = ReferenceEncoderFromWaveform(d_prosody=D_PROSODY, n_mels=N_MELS)
        enc.eval()
        wav = _make_waveform(batch_size=2)
        with torch.no_grad():
            out = enc(wav)
        assert out.shape == (2, D_PROSODY)

    def test_gradient_flow_from_waveform(self):
        """Gradients must flow from loss back through mel computation."""
        enc = ReferenceEncoderFromWaveform(d_prosody=D_PROSODY, n_mels=N_MELS)
        wav = _make_waveform(batch_size=1)

        out = enc(wav)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in enc.parameters()
        )
        assert has_grad, "No gradients flowed through ReferenceEncoderFromWaveform"

    def test_mel_computation_finite(self):
        """compute_mel should produce finite values."""
        enc = ReferenceEncoderFromWaveform(d_prosody=D_PROSODY, n_mels=N_MELS)
        wav = _make_waveform(batch_size=1)
        mel = enc.compute_mel(wav)
        assert torch.isfinite(mel).all(), "compute_mel produced non-finite values"
        assert mel.shape[1] == N_MELS
