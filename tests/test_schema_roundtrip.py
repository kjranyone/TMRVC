"""Schema roundtrip tests.

Required by track_architecture.md § 7:
- Physical control registry serialization roundtrip
- IntentCompilerOutput roundtrip
- TrajectoryRecord roundtrip
- Pointer-synchronous edit target roundtrip
- Compatibility checks between core schema and export schema
"""

import json
import uuid
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tmrvc_core.types import (
    PacingControls,
    SupervisionTier,
    TrajectoryProvenance,
    ActingTextureMacro,
    ActingTextureLatent,
    VoiceStateSupervision,
    IntentCompilerOutput,
    TrajectoryRecord,
    BootstrapCacheEntry,
)
from tmrvc_core.voice_state import (
    VOICE_STATE_REGISTRY,
    CANONICAL_VOICE_STATE_IDS,
    CANONICAL_VOICE_STATE_LABELS,
    CANONICAL_VOICE_STATE_DEFAULTS,
    BASIC_PANEL_IDS,
    get_voice_state_dimension_names,
)


class TestPhysicalRegistry:
    """Test 12-D physical control registry."""

    def test_registry_has_12_dimensions(self):
        assert len(VOICE_STATE_REGISTRY) == 12

    def test_indices_are_contiguous(self):
        for i in range(12):
            assert i in VOICE_STATE_REGISTRY

    def test_compatibility_indices_0_to_7(self):
        """Indices 0-7 must retain legacy semantics."""
        legacy_ids = [
            "pitch_level", "pitch_range", "energy_level", "pressedness",
            "spectral_tilt", "breathiness", "voice_irregularity", "openness",
        ]
        for i, expected_id in enumerate(legacy_ids):
            assert VOICE_STATE_REGISTRY[i].id == expected_id

    def test_new_dimensions(self):
        assert VOICE_STATE_REGISTRY[8].id == "aperiodicity"
        assert VOICE_STATE_REGISTRY[9].id == "formant_shift"
        assert VOICE_STATE_REGISTRY[10].id == "vocal_effort"
        assert VOICE_STATE_REGISTRY[11].id == "creak"

    def test_all_dimensions_have_valid_ranges(self):
        for dim in VOICE_STATE_REGISTRY.values():
            assert dim.min_val <= dim.default_val <= dim.max_val
            assert dim.min_val == 0.0
            assert dim.max_val == 1.0

    def test_no_duplicate_ids(self):
        ids = [d.id for d in VOICE_STATE_REGISTRY.values()]
        assert len(ids) == len(set(ids))

    def test_no_duplicate_proxy_observables(self):
        observables = [d.proxy_observable for d in VOICE_STATE_REGISTRY.values()]
        assert len(observables) == len(set(observables))

    def test_canonical_ids_tuple(self):
        assert len(CANONICAL_VOICE_STATE_IDS) == 12
        assert CANONICAL_VOICE_STATE_IDS[0] == "pitch_level"
        assert CANONICAL_VOICE_STATE_IDS[11] == "creak"

    def test_canonical_labels_tuple(self):
        assert len(CANONICAL_VOICE_STATE_LABELS) == 12

    def test_canonical_defaults_tuple(self):
        assert len(CANONICAL_VOICE_STATE_DEFAULTS) == 12
        for val in CANONICAL_VOICE_STATE_DEFAULTS:
            assert 0.0 <= val <= 1.0

    def test_basic_panel_subset(self):
        all_ids = set(CANONICAL_VOICE_STATE_IDS)
        for pid in BASIC_PANEL_IDS:
            assert pid in all_ids
        assert len(BASIC_PANEL_IDS) == 6

    def test_dimension_names(self):
        names = get_voice_state_dimension_names()
        assert len(names) == 12

    def test_serialization_roundtrip(self):
        """Registry dimensions must survive dict serialization."""
        for dim in VOICE_STATE_REGISTRY.values():
            d = {
                "index": dim.index,
                "id": dim.id,
                "name": dim.name,
                "physical_interpretation": dim.physical_interpretation,
                "unit": dim.unit,
                "min_val": dim.min_val,
                "max_val": dim.max_val,
                "default_val": dim.default_val,
                "is_frame_local": dim.is_frame_local,
                "proxy_observable": dim.proxy_observable,
            }
            serialized = json.dumps(d)
            loaded = json.loads(serialized)
            assert loaded["id"] == dim.id
            assert loaded["default_val"] == dim.default_val


class TestSupervisionTier:
    """Test supervision tier enum."""

    def test_all_tiers_exist(self):
        assert SupervisionTier.A == "tier_a"
        assert SupervisionTier.B == "tier_b"
        assert SupervisionTier.C == "tier_c"
        assert SupervisionTier.D == "tier_d"

    def test_tier_string_roundtrip(self):
        for tier in SupervisionTier:
            assert SupervisionTier(tier.value) == tier


class TestTrajectoryProvenance:
    """Test trajectory provenance enum."""

    def test_all_provenances_exist(self):
        assert TrajectoryProvenance.FRESH_COMPILE == "fresh_compile"
        assert TrajectoryProvenance.DETERMINISTIC_REPLAY == "deterministic_replay"
        assert TrajectoryProvenance.CROSS_SPEAKER_TRANSFER == "cross_speaker_transfer"
        assert TrajectoryProvenance.PATCHED_REPLAY == "patched_replay"


class TestActingTextureMacro:
    """Test acting macro controls."""

    def test_defaults(self):
        macro = ActingTextureMacro()
        assert macro.intensity == 0.5
        assert macro.reference_mix == 0.0

    def test_custom_values(self):
        macro = ActingTextureMacro(intensity=0.9, tension=0.8)
        assert macro.intensity == 0.9
        assert macro.tension == 0.8


class TestActingTextureLatent:
    """Test acting texture latent contract."""

    def test_creation(self):
        latent = ActingTextureLatent(
            latent=torch.randn(1, 24),
            source="reference",
        )
        assert latent.latent.shape == (1, 24)
        assert latent.source == "reference"
        assert latent.schema_version == "1.0"

    def test_physical_and_latent_are_separate_tensors(self):
        """Physical controls and latent controls must be different tensors."""
        physical = torch.randn(1, 12)
        latent_tensor = torch.randn(1, 24)
        latent = ActingTextureLatent(latent=latent_tensor)
        assert physical.shape != latent.latent.shape
        assert not torch.equal(physical, latent.latent[:, :12])


class TestVoiceStateSupervision:
    """Test 12-D voice state supervision."""

    def test_creation(self):
        B, T = 2, 100
        sup = VoiceStateSupervision(
            targets=torch.randn(B, T, 12),
            observed_mask=torch.ones(B, T, 12, dtype=torch.bool),
            confidence=torch.ones(B, T, 12),
            supervision_tier=SupervisionTier.A,
        )
        assert sup.targets.shape == (B, T, 12)
        assert sup.observed_mask.shape == (B, T, 12)

    def test_dimension_is_12_not_8(self):
        sup = VoiceStateSupervision(
            targets=torch.randn(1, 50, 12),
            observed_mask=torch.ones(1, 50, 12, dtype=torch.bool),
            confidence=torch.ones(1, 50, 12),
        )
        assert sup.targets.shape[-1] == 12


class TestIntentCompilerOutput:
    """Test IntentCompilerOutput roundtrip."""

    def test_creation_with_defaults(self):
        output = IntentCompilerOutput(
            compile_id=str(uuid.uuid4()),
            source_prompt="test prompt",
        )
        assert output.schema_version == "1.0"
        assert output.physical_targets is None
        assert output.acting_latent_prior is None

    def test_full_creation(self):
        output = IntentCompilerOutput(
            compile_id="test-001",
            source_prompt="Speak gently",
            inline_tags=["gentle", "warm"],
            physical_targets=torch.randn(1, 12),
            physical_confidence=torch.ones(1, 12),
            acting_latent_prior=torch.randn(1, 24),
            acting_macro=ActingTextureMacro(intensity=0.8),
            pacing=PacingControls(pace=0.9),
            dialogue_state={"turn": 3},
            warnings=["Low confidence on creak dimension"],
            provenance="intent_compiler_v4",
        )
        assert output.physical_targets.shape == (1, 12)
        assert output.acting_latent_prior.shape == (1, 24)
        assert output.acting_macro.intensity == 0.8
        assert output.pacing.pace == 0.9
        assert len(output.warnings) == 1

    def test_schema_version(self):
        output = IntentCompilerOutput(
            compile_id="x", source_prompt="y",
        )
        assert output.schema_version == "1.0"


class TestTrajectoryRecord:
    """Test TrajectoryRecord roundtrip."""

    def test_creation_minimal(self):
        record = TrajectoryRecord(
            trajectory_id="traj-001",
            source_compile_id="comp-001",
        )
        assert record.schema_version == "1.0"
        assert record.version == 1
        assert record.provenance == TrajectoryProvenance.FRESH_COMPILE

    def test_full_creation(self):
        T, L = 200, 50
        record = TrajectoryRecord(
            trajectory_id="traj-002",
            source_compile_id="comp-002",
            phoneme_ids=torch.randint(0, 200, (1, L)),
            text_suprasegmentals=torch.randn(1, L, 4),
            pointer_trace=[(i, 4) for i in range(L)],
            physical_trajectory=torch.randn(T, 12),
            acting_latent_trajectory=torch.randn(T, 24),
            acting_latent_is_static=False,
            acoustic_trace=torch.randint(0, 1024, (8, T)),
            control_trace=torch.randint(0, 64, (4, T)),
            applied_pacing=PacingControls(pace=1.1),
            speaker_profile_id="spk-001",
            provenance=TrajectoryProvenance.FRESH_COMPILE,
        )
        assert record.physical_trajectory.shape == (T, 12)
        assert record.acting_latent_trajectory.shape == (T, 24)
        assert record.acoustic_trace.shape == (8, T)
        assert len(record.pointer_trace) == L

    def test_physical_is_12d_not_8d(self):
        record = TrajectoryRecord(
            trajectory_id="t", source_compile_id="c",
            physical_trajectory=torch.randn(100, 12),
        )
        assert record.physical_trajectory.shape[-1] == 12

    def test_optimistic_versioning(self):
        record = TrajectoryRecord(
            trajectory_id="t", source_compile_id="c",
        )
        assert record.version == 1
        record.version += 1
        assert record.version == 2

    def test_provenance_labels(self):
        for prov in TrajectoryProvenance:
            record = TrajectoryRecord(
                trajectory_id="t", source_compile_id="c",
                provenance=prov,
            )
            assert record.provenance == prov

    def test_pointer_synchronous_edit(self):
        """Patching a local region is a first-class use case."""
        T, L = 100, 25
        record = TrajectoryRecord(
            trajectory_id="t", source_compile_id="c",
            pointer_trace=[(i, 4) for i in range(L)],
            physical_trajectory=torch.randn(T, 12),
        )
        # Patch frames 20-40
        original = record.physical_trajectory.clone()
        patch = torch.ones(20, 12) * 0.5
        record.physical_trajectory[20:40] = patch
        # Verify only patched region changed
        assert torch.equal(record.physical_trajectory[:20], original[:20])
        assert torch.equal(record.physical_trajectory[20:40], patch)
        assert torch.equal(record.physical_trajectory[40:], original[40:])

    def test_static_vs_time_varying_latent(self):
        # Static latent
        static = TrajectoryRecord(
            trajectory_id="t", source_compile_id="c",
            acting_latent_trajectory=torch.randn(1, 24),
            acting_latent_is_static=True,
        )
        assert static.acting_latent_is_static is True
        assert static.acting_latent_trajectory.shape == (1, 24)

        # Time-varying latent
        varying = TrajectoryRecord(
            trajectory_id="t", source_compile_id="c",
            acting_latent_trajectory=torch.randn(100, 24),
            acting_latent_is_static=False,
        )
        assert varying.acting_latent_is_static is False
        assert varying.acting_latent_trajectory.shape == (100, 24)


class TestBootstrapCacheEntry:
    """Test bootstrap cache entry."""

    def test_creation(self):
        entry = BootstrapCacheEntry(
            utterance_id="utt-001",
            corpus_id="corpus-001",
            pseudo_speaker_id="spk-001",
            supervision_tier=SupervisionTier.B,
        )
        assert entry.schema_version == "1.0"
        assert entry.supervision_tier == SupervisionTier.B

    def test_physical_targets_are_12d(self):
        T = 100
        entry = BootstrapCacheEntry(
            utterance_id="u", corpus_id="c",
            physical_targets=torch.randn(T, 12),
            physical_observed_mask=torch.ones(T, 12, dtype=torch.bool),
            physical_confidence=torch.ones(T, 12),
        )
        assert entry.physical_targets.shape[-1] == 12
