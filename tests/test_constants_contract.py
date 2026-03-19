"""Contract tests: verify that hardcoded defaults in model/runtime code
match the single source of truth in configs/constants.yaml.

These tests exist because parallel subagent implementation introduced
value drift between constants.yaml and Python defaults.  Any future
change to constants.yaml that is not propagated to the corresponding
default parameter will be caught here.

Run:  pytest tests/test_constants_contract.py -v
"""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

# ---------------------------------------------------------------------------
# Load constants.yaml once
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).resolve().parent.parent / "configs" / "constants.yaml"


@pytest.fixture(scope="module")
def yaml_constants() -> dict:
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Generated Python constants match YAML
# ---------------------------------------------------------------------------


class TestGeneratedConstants:
    """_generated_constants.py must be in sync with constants.yaml."""

    def test_generated_file_exists(self):
        from tmrvc_core._generated_constants import D_MODEL  # noqa: F401

    def test_d_model(self, yaml_constants):
        from tmrvc_core._generated_constants import D_MODEL

        assert D_MODEL == yaml_constants["d_model"]

    def test_d_speaker(self, yaml_constants):
        from tmrvc_core._generated_constants import D_SPEAKER

        assert D_SPEAKER == yaml_constants["d_speaker"]

    def test_d_voice_state(self, yaml_constants):
        from tmrvc_core._generated_constants import D_VOICE_STATE

        assert D_VOICE_STATE == yaml_constants["d_voice_state"]

    def test_d_prosody(self, yaml_constants):
        from tmrvc_core._generated_constants import D_PROSODY

        assert D_PROSODY == yaml_constants["d_prosody"]

    def test_d_text_encoder(self, yaml_constants):
        from tmrvc_core._generated_constants import D_TEXT_ENCODER

        assert D_TEXT_ENCODER == yaml_constants["d_text_encoder"]

    def test_n_text_encoder_layers(self, yaml_constants):
        from tmrvc_core._generated_constants import N_TEXT_ENCODER_LAYERS

        assert N_TEXT_ENCODER_LAYERS == yaml_constants["n_text_encoder_layers"]

    def test_n_text_encoder_heads(self, yaml_constants):
        from tmrvc_core._generated_constants import N_TEXT_ENCODER_HEADS

        assert N_TEXT_ENCODER_HEADS == yaml_constants["n_text_encoder_heads"]

    def test_text_encoder_ff_dim(self, yaml_constants):
        from tmrvc_core._generated_constants import TEXT_ENCODER_FF_DIM

        assert TEXT_ENCODER_FF_DIM == yaml_constants["text_encoder_ff_dim"]

    def test_n_codebooks(self, yaml_constants):
        from tmrvc_core._generated_constants import N_CODEBOOKS

        assert N_CODEBOOKS == yaml_constants["n_codebooks"]

    def test_rvq_vocab_size(self, yaml_constants):
        from tmrvc_core._generated_constants import RVQ_VOCAB_SIZE

        assert RVQ_VOCAB_SIZE == yaml_constants["rvq_vocab_size"]

    def test_phoneme_vocab_size(self, yaml_constants):
        from tmrvc_core._generated_constants import PHONEME_VOCAB_SIZE

        assert PHONEME_VOCAB_SIZE == yaml_constants["phoneme_vocab_size"]

    def test_n_prompt_summary_tokens(self, yaml_constants):
        from tmrvc_core._generated_constants import N_PROMPT_SUMMARY_TOKENS

        assert N_PROMPT_SUMMARY_TOKENS == yaml_constants["n_prompt_summary_tokens"]

    def test_cfg_scale_default(self, yaml_constants):
        from tmrvc_core._generated_constants import CFG_SCALE_DEFAULT

        assert CFG_SCALE_DEFAULT == yaml_constants["cfg_scale_default"]

    def test_cfg_scale_max(self, yaml_constants):
        from tmrvc_core._generated_constants import CFG_SCALE_MAX

        assert CFG_SCALE_MAX == yaml_constants["cfg_scale_max"]

    def test_max_frames_per_unit(self, yaml_constants):
        from tmrvc_core._generated_constants import MAX_FRAMES_PER_UNIT

        assert MAX_FRAMES_PER_UNIT == yaml_constants["max_frames_per_unit"]

    def test_skip_protection_threshold(self, yaml_constants):
        from tmrvc_core._generated_constants import SKIP_PROTECTION_THRESHOLD

        assert SKIP_PROTECTION_THRESHOLD == yaml_constants["skip_protection_threshold"]

    def test_streaming_latency_budget_ms(self, yaml_constants):
        from tmrvc_core._generated_constants import STREAMING_LATENCY_BUDGET_MS

        assert STREAMING_LATENCY_BUDGET_MS == yaml_constants["streaming_latency_budget_ms"]

    def test_streaming_hardware_class_primary(self, yaml_constants):
        from tmrvc_core._generated_constants import STREAMING_HARDWARE_CLASS_PRIMARY

        assert STREAMING_HARDWARE_CLASS_PRIMARY == yaml_constants["streaming_hardware_class_primary"]

    def test_sample_rate(self, yaml_constants):
        from tmrvc_core._generated_constants import SAMPLE_RATE

        assert SAMPLE_RATE == yaml_constants["sample_rate"]

    def test_hop_length(self, yaml_constants):
        from tmrvc_core._generated_constants import HOP_LENGTH

        assert HOP_LENGTH == yaml_constants["hop_length"]


# ---------------------------------------------------------------------------
# 2. constants.py re-exports do not shadow generated values
# ---------------------------------------------------------------------------


class TestConstantsPyNoShadow:
    """constants.py must not re-declare values that override _generated_constants."""

    def test_no_hardcoded_overrides(self):
        """constants.py must only contain `from _generated_constants import *`.

        If someone adds hand-written constants with the same name,
        Python's last-wins semantics will silently override the YAML values.
        """
        import tmrvc_core.constants as mod
        import tmrvc_core._generated_constants as gen

        # Every key exported by _generated_constants must have the same
        # value when accessed via constants (i.e. no shadow override).
        gen_names = [k for k in dir(gen) if k.isupper() and not k.startswith("_")]
        for name in gen_names:
            gen_val = getattr(gen, name)
            mod_val = getattr(mod, name, None)
            assert mod_val == gen_val, (
                f"constants.py shadows _generated_constants.{name}: "
                f"expected {gen_val!r}, got {mod_val!r}"
            )


# ---------------------------------------------------------------------------
# 3. Model default parameters match constants.yaml
# ---------------------------------------------------------------------------


class TestTextEncoderDefaults:
    """TextEncoder defaults must match constants.yaml."""

    def test_n_layers(self, yaml_constants):
        from tmrvc_train.models.text_encoder import TextEncoder
        import inspect

        sig = inspect.signature(TextEncoder.__init__)
        assert sig.parameters["n_layers"].default == yaml_constants["n_text_encoder_layers"]

    def test_n_heads(self, yaml_constants):
        from tmrvc_train.models.text_encoder import TextEncoder
        import inspect

        sig = inspect.signature(TextEncoder.__init__)
        assert sig.parameters["n_heads"].default == yaml_constants["n_text_encoder_heads"]

    def test_ff_dim(self, yaml_constants):
        from tmrvc_train.models.text_encoder import TextEncoder
        import inspect

        sig = inspect.signature(TextEncoder.__init__)
        assert sig.parameters["ff_dim"].default == yaml_constants["text_encoder_ff_dim"]

    def test_d_model(self, yaml_constants):
        from tmrvc_train.models.text_encoder import TextEncoder
        import inspect

        sig = inspect.signature(TextEncoder.__init__)
        assert sig.parameters["d_model"].default == yaml_constants["d_model"]

    def test_vocab_size(self, yaml_constants):
        from tmrvc_train.models.text_encoder import TextEncoder
        import inspect

        sig = inspect.signature(TextEncoder.__init__)
        assert sig.parameters["vocab_size"].default == yaml_constants["phoneme_vocab_size"]


class TestReferenceEncoderDefaults:
    """ReferenceEncoder defaults must match constants.yaml."""

    def test_d_prosody(self, yaml_constants):
        from tmrvc_train.models.reference_encoder import ReferenceEncoder
        import inspect

        sig = inspect.signature(ReferenceEncoder.__init__)
        assert sig.parameters["d_prosody"].default == yaml_constants["d_prosody"]

    def test_d_prosody_waveform(self, yaml_constants):
        from tmrvc_train.models.reference_encoder import ReferenceEncoderFromWaveform
        import inspect

        sig = inspect.signature(ReferenceEncoderFromWaveform.__init__)
        assert sig.parameters["d_prosody"].default == yaml_constants["d_prosody"]

    def test_n_mels(self, yaml_constants):
        from tmrvc_train.models.reference_encoder import ReferenceEncoder
        import inspect

        sig = inspect.signature(ReferenceEncoder.__init__)
        assert sig.parameters["n_mels"].default == yaml_constants["n_mels"]


class TestDisentangledUCLMDefaults:
    """DisentangledUCLM defaults must match constants.yaml."""

    def test_d_model(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["d_model"].default == yaml_constants["d_model"]

    def test_n_heads(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["n_heads"].default == yaml_constants["uclm_n_heads"]

    def test_n_layers(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["n_layers"].default == yaml_constants["uclm_n_layers"]

    def test_d_speaker(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["d_speaker"].default == yaml_constants["d_speaker"]

    def test_d_explicit(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["d_explicit"].default == yaml_constants["d_voice_state_explicit"]

    def test_d_ssl(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["d_ssl"].default == yaml_constants["d_voice_state_ssl"]

    def test_n_codebooks(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["n_codebooks"].default == yaml_constants["n_codebooks"]

    def test_rvq_vocab_size(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["rvq_vocab_size"].default == yaml_constants["rvq_vocab_size"]

    def test_d_prosody(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["d_prosody"].default == yaml_constants["d_prosody"]

    def test_control_vocab_size(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["control_vocab_size"].default == yaml_constants["control_vocab_size"]

    def test_vq_bins(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        import inspect

        sig = inspect.signature(DisentangledUCLM.__init__)
        assert sig.parameters["vq_bins"].default == yaml_constants["uclm_vq_bins"]


class TestProsodyPredictorDefaults:
    """ProsodyPredictor defaults must match constants.yaml."""

    def test_d_prosody(self, yaml_constants):
        from tmrvc_train.models.uclm_model import ProsodyPredictor
        import inspect

        sig = inspect.signature(ProsodyPredictor.__init__)
        assert sig.parameters["d_prosody"].default == yaml_constants["d_prosody"]

    def test_d_model(self, yaml_constants):
        from tmrvc_train.models.uclm_model import ProsodyPredictor
        import inspect

        sig = inspect.signature(ProsodyPredictor.__init__)
        assert sig.parameters["d_model"].default == yaml_constants["d_model"]


class TestDialogueContextProjectorDefaults:
    """DialogueContextProjector defaults must match constants.yaml."""

    def test_d_prosody(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DialogueContextProjector
        import inspect

        sig = inspect.signature(DialogueContextProjector.__init__)
        assert sig.parameters["d_prosody"].default == yaml_constants["d_prosody"]

    def test_d_model(self, yaml_constants):
        from tmrvc_train.models.uclm_model import DialogueContextProjector
        import inspect

        sig = inspect.signature(DialogueContextProjector.__init__)
        assert sig.parameters["d_model"].default == yaml_constants["d_model"]


# ---------------------------------------------------------------------------
# 4. Instantiated model dimensions match constants (runtime check)
# ---------------------------------------------------------------------------


class TestInstantiatedDimensions:
    """Models instantiated with defaults must produce the correct shapes."""

    def test_text_encoder_layer_count(self, yaml_constants):
        from tmrvc_train.models.text_encoder import TextEncoder

        enc = TextEncoder()
        assert enc.encoder.num_layers == yaml_constants["n_text_encoder_layers"]

    def test_reference_encoder_output_dim(self, yaml_constants):
        from tmrvc_train.models.reference_encoder import ReferenceEncoder

        enc = ReferenceEncoder()
        assert enc.d_prosody == yaml_constants["d_prosody"]

    def test_uclm_text_encoder_layer_count(self, yaml_constants):
        """TextEncoder inside DisentangledUCLM must use the correct n_layers."""
        from tmrvc_train.models.uclm_model import DisentangledUCLM

        model = DisentangledUCLM()
        assert model.text_encoder.encoder.num_layers == yaml_constants["n_text_encoder_layers"]


# ---------------------------------------------------------------------------
# 5. Frame convention constants
# ---------------------------------------------------------------------------


class TestFrameConvention:
    """Frame convention must be consistent across constants."""

    def test_sample_rate(self, yaml_constants):
        assert yaml_constants["sample_rate"] == 24000

    def test_hop_length(self, yaml_constants):
        """Mel frontend hop_length = 240 (10ms @ 24kHz)."""
        assert yaml_constants["hop_length"] == 240

    def test_codec_hop_length(self, yaml_constants):
        """EnCodec hop_length = 320 (75 Hz @ 24kHz)."""
        assert yaml_constants["codec_hop_length"] == 320

    def test_frame_duration_is_10ms(self, yaml_constants):
        """Mel frontend: 10 ms frame step = sample_rate * 0.01 = hop_length."""
        expected_hop = int(yaml_constants["sample_rate"] * 0.01)
        assert yaml_constants["hop_length"] == expected_hop


# ---------------------------------------------------------------------------
# 6. Phoneme vocab capacity margin
# ---------------------------------------------------------------------------


class TestPhonemeVocabCapacity:
    """phoneme_vocab_size must exceed active inventory by >= 20."""

    def test_margin(self, yaml_constants):
        from tmrvc_data.g2p import PHONEME_LIST

        active_count = len(PHONEME_LIST)
        capacity = yaml_constants["phoneme_vocab_size"]
        margin = capacity - active_count
        assert margin >= 20, (
            f"phoneme_vocab_size ({capacity}) - active phones ({active_count}) "
            f"= {margin}, must be >= 20 for append-only expansion"
        )


# ---------------------------------------------------------------------------
# 7. Voice state dimensionality
# ---------------------------------------------------------------------------


class TestVoiceStateDimensionality:
    """voice_state must be 12-D everywhere."""

    def test_yaml_value(self, yaml_constants):
        assert yaml_constants["d_voice_state"] == 12

    def test_explicit_equals_total(self, yaml_constants):
        assert yaml_constants["d_voice_state_explicit"] == yaml_constants["d_voice_state"]

    def test_serve_schema(self):
        from tmrvc_serve.schemas import PhysicalControls

        pc = PhysicalControls()
        assert len(pc.to_list()) == 12
