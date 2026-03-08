"""Tests for Worker 10: Export, Cache Integration, and Tooling.

Covers:
- Cache-ready meta.json with curation provenance
- Bootstrap alignment export with projection provenance
- voice_state export with masks and confidence
- Dialogue context export (model-agnostic)
- Few-shot prompt metadata export
- Artifact package contract (checksums, provenance, retention)
- Bucket-specific export behavior
- Holdout bundle frozen prompt-target pairings
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tmrvc_data.curation.export import (
    ArtifactPackage,
    CurationExporter,
    ExportConfig,
    ARTIFACT_TYPE_HOLDOUT_BUNDLE,
    ARTIFACT_TYPE_TRAINING_BUNDLE,
    RETENTION_DURABLE,
    RETENTION_RELEASE_CANDIDATE,
    VOICE_STATE_DIM_NAMES,
    VOICE_STATE_NDIM,
    _sha256_file,
)
from tmrvc_data.curation.models import (
    CurationRecord,
    PromotionBucket,
    Provenance,
    RecordStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    record_id: str = "rec_001",
    bucket: PromotionBucket = PromotionBucket.TTS_MAINLINE,
    **overrides,
) -> CurationRecord:
    """Create a promoted CurationRecord for testing."""
    defaults = dict(
        record_id=record_id,
        source_path="/data/audio/test.wav",
        audio_hash="sha256_abc123",
        transcript="hello world",
        language="en",
        speaker_cluster="spk_01",
        quality_score=0.92,
        status=RecordStatus.PROMOTED,
        promotion_bucket=bucket,
        source_legality="owned",
        duration_sec=5.0,
        segment_start_sec=0.0,
        segment_end_sec=5.0,
        conversation_id="conv_001",
        turn_index=1,
        prev_record_id="rec_000",
        next_record_id="rec_002",
        context_window_ids=["rec_000"],
        pass_index=2,
        providers={
            "asr": Provenance(
                stage="asr",
                provider="whisper",
                version="3.0",
                timestamp=1000.0,
                confidence=0.95,
            ),
        },
    )
    defaults.update(overrides)
    attrs = defaults.pop("attributes", {})
    r = CurationRecord(**defaults)
    r.attributes = attrs
    return r


def _make_record_with_voice_state(
    record_id: str = "rec_vs_001",
    n_frames: int = 100,
    **kwargs,
) -> CurationRecord:
    """Create a promoted record with voice_state data."""
    rng = np.random.RandomState(42)
    vs = rng.randn(n_frames, VOICE_STATE_NDIM).astype(np.float32)
    mask = rng.rand(n_frames, VOICE_STATE_NDIM) > 0.3
    conf = rng.rand(n_frames, VOICE_STATE_NDIM).astype(np.float32)

    return _make_record(
        record_id=record_id,
        attributes={
            "voice_state": vs,
            "voice_state_observed_mask": mask,
            "voice_state_confidence": conf,
            "voice_state_meta": {
                "has_voice_state": True,
                "has_observed_mask": True,
                "has_confidence": True,
                "estimator_id": "voice_state_estimator_v2",
                "calibration_version": "cal_2026_01",
                "target_source_provenance": "curation_pipeline",
            },
        },
        **kwargs,
    )


def _make_record_with_alignment(
    record_id: str = "rec_align_001",
    **kwargs,
) -> CurationRecord:
    """Create a promoted record with bootstrap alignment data."""
    return _make_record(
        record_id=record_id,
        attributes={
            "word_timestamps": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
            "phoneme_ids_list": [10, 11, 12, 20, 21, 22],
            "word_to_phoneme_map": [(0, 3), (3, 6)],
            "num_samples": 24000,  # 1 second at 24kHz
        },
        **kwargs,
    )


def _make_record_with_prompt_metadata(
    record_id: str = "rec_prompt_001",
    **kwargs,
) -> CurationRecord:
    """Create a promoted record with few-shot prompt metadata."""
    return _make_record(
        record_id=record_id,
        attributes={
            "prompt_eligible": True,
            "prompt_pair_id": "pair_42",
            "prompt_metadata": {
                "speaker_id": "spk_01",
                "prompt_candidate_record_ids": ["rec_002", "rec_003"],
                "prompt_policy_version": "2.0.0",
                "prompt_duration_sec": 4.5,
                "prompt_language": "en",
                "speaker_purity_estimate": 0.97,
                "leakage_policy_flags": {"holdout_safe": True},
            },
        },
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: Cache-ready meta.json with curation provenance
# ---------------------------------------------------------------------------


class TestMetaJsonProvenance:
    def test_meta_contains_curation_record_id(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["curation_record_id"] == "rec_001"

    def test_meta_contains_promotion_bucket(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["promotion_bucket"] == "tts_mainline"

    def test_meta_contains_curation_pass(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["curation_pass"] == 2

    def test_meta_contains_quality_score(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["quality_score"] == 0.92

    def test_meta_contains_provider_provenance(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert "asr" in meta["providers"]
        assert meta["providers"]["asr"]["provider"] == "whisper"
        assert meta["providers"]["asr"]["version"] == "3.0"
        assert meta["providers"]["asr"]["confidence"] == 0.95

    def test_meta_contains_score_components(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(attributes={"score_components": {"snr": 0.9, "transcript": 0.85}})]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["score_components"]["snr"] == 0.9


# ---------------------------------------------------------------------------
# Test: Bootstrap alignment export with projection provenance
# ---------------------------------------------------------------------------


class TestBootstrapAlignmentExport:
    def test_export_bootstrap_alignment_json(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_alignment()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        align_path = tmp_path / "rec_align_001" / "bootstrap_alignment.json"
        assert align_path.exists()

        data = json.loads(align_path.read_text())
        assert "spans" in data
        assert "num_text_units" in data
        assert "num_frames" in data
        assert "provenance" in data
        assert "projection_method" in data

    def test_alignment_preserves_projection_provenance(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_alignment()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        data = json.loads(
            (tmp_path / "rec_align_001" / "bootstrap_alignment.json").read_text()
        )
        # Must have algorithm_version for Worker 03 tracking
        assert "algorithm_version" in data
        # Must have provenance field
        assert data["provenance"] != ""

    def test_alignment_uses_canonical_frame_convention(self, tmp_path):
        """Frame convention: sample_rate=24000, hop_length=240."""
        exporter = CurationExporter()
        records = [_make_record_with_alignment()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        data = json.loads(
            (tmp_path / "rec_align_001" / "bootstrap_alignment.json").read_text()
        )
        # 24000 samples / 240 hop = 100 frames
        assert data["num_frames"] == 100

    def test_alignment_fails_closed_without_required_data(self, tmp_path):
        """Export must not produce alignment when projection data is missing."""
        exporter = CurationExporter()
        # Record without word_timestamps => no alignment exported
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        align_path = tmp_path / "rec_001" / "bootstrap_alignment.json"
        assert not align_path.exists()

    def test_alignment_phoneme_ids_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_alignment()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        phoneme_path = tmp_path / "rec_align_001" / "phoneme_ids.npy"
        assert phoneme_path.exists()
        arr = np.load(phoneme_path)
        assert arr.tolist() == [10, 11, 12, 20, 21, 22]


# ---------------------------------------------------------------------------
# Test: voice_state export
# ---------------------------------------------------------------------------


class TestVoiceStateExport:
    def test_voice_state_npy_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_voice_state()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        vs_path = tmp_path / "rec_vs_001" / "voice_state.npy"
        assert vs_path.exists()
        arr = np.load(vs_path)
        assert arr.shape == (100, 8)
        assert arr.dtype == np.float32

    def test_voice_state_mask_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_voice_state()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        mask_path = tmp_path / "rec_vs_001" / "voice_state_observed_mask.npy"
        assert mask_path.exists()
        arr = np.load(mask_path)
        assert arr.shape == (100, 8)
        assert arr.dtype == bool

    def test_voice_state_confidence_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_voice_state()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        conf_path = tmp_path / "rec_vs_001" / "voice_state_confidence.npy"
        assert conf_path.exists()
        arr = np.load(conf_path)
        assert arr.shape == (100, 8)
        assert arr.dtype == np.float32

    def test_voice_state_meta_json_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_voice_state()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta_path = tmp_path / "rec_vs_001" / "voice_state_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["dimensions"] == list(VOICE_STATE_DIM_NAMES)
        assert meta["n_frames"] == 100
        assert meta["has_observed_mask"] is True
        assert meta["has_confidence"] is True
        assert meta["estimator_id"] == "voice_state_estimator_v2"
        assert meta["calibration_version"] == "cal_2026_01"
        assert meta["sample_rate"] == 24000
        assert meta["hop_length"] == 240

    def test_voice_state_not_exported_when_missing(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]  # No voice_state in attributes
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        vs_path = tmp_path / "rec_001" / "voice_state.npy"
        assert not vs_path.exists()

    def test_voice_state_meta_in_record_meta(self, tmp_path):
        """meta.json should reference voice_state supervision status."""
        exporter = CurationExporter()
        records = [_make_record_with_voice_state()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_vs_001" / "meta.json").read_text())
        assert "voice_state_supervision" in meta
        assert meta["voice_state_supervision"]["has_voice_state"] is True
        assert meta["voice_state_supervision"]["dimensions"] == list(VOICE_STATE_DIM_NAMES)


# ---------------------------------------------------------------------------
# Test: Dialogue context export
# ---------------------------------------------------------------------------


class TestDialogueContextExport:
    def test_dialogue_graph_fields_in_meta(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["conversation_id"] == "conv_001"
        assert meta["turn_index"] == 1
        assert meta["prev_record_id"] == "rec_000"
        assert meta["next_record_id"] == "rec_002"
        assert meta["context_window_ids"] == ["rec_000"]

    def test_dialogue_context_model_agnostic(self, tmp_path):
        """Dialogue context export must be raw text, model-agnostic."""
        exporter = CurationExporter()
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        ctx = meta["dialogue_context"]
        assert ctx["conversation_id"] == "conv_001"
        assert ctx["raw_text"] == "hello world"
        assert ctx["turn_index"] == 1
        assert ctx["prev_record_id"] == "rec_000"
        assert ctx["next_record_id"] == "rec_002"

    def test_no_dialogue_context_without_conversation(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(conversation_id=None)]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert "dialogue_context" not in meta


# ---------------------------------------------------------------------------
# Test: Few-shot prompt metadata
# ---------------------------------------------------------------------------


class TestPromptMetadataExport:
    def test_prompt_eligible_flag_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_prompt_metadata()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_prompt_001" / "meta.json").read_text())
        assert meta["prompt_eligible"] is True

    def test_prompt_pair_id_exported(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_prompt_metadata()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_prompt_001" / "meta.json").read_text())
        assert meta["prompt_pair_id"] == "pair_42"

    def test_extended_prompt_metadata(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_prompt_metadata()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_prompt_001" / "meta.json").read_text())
        pm = meta["prompt_metadata"]
        assert pm["prompt_policy_version"] == "2.0.0"
        assert pm["speaker_purity_estimate"] == 0.97
        assert pm["leakage_policy_flags"]["holdout_safe"] is True
        assert pm["prompt_candidate_record_ids"] == ["rec_002", "rec_003"]

    def test_default_prompt_eligible_false(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]  # No prompt attrs
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["prompt_eligible"] is False
        assert meta["prompt_pair_id"] is None


# ---------------------------------------------------------------------------
# Test: Artifact package contract
# ---------------------------------------------------------------------------


class TestArtifactPackage:
    def test_artifact_package_has_uuid(self):
        pkg = ArtifactPackage()
        assert len(pkg.artifact_id) == 36  # UUID format

    def test_artifact_package_to_dict(self):
        pkg = ArtifactPackage(
            artifact_type=ARTIFACT_TYPE_TRAINING_BUNDLE,
            retention_class=RETENTION_DURABLE,
            record_count=10,
        )
        d = pkg.to_dict()
        assert d["artifact_type"] == "cache_ready_training_bundle"
        assert d["retention_class"] == "durable"
        assert d["record_count"] == 10
        assert "artifact_id" in d
        assert "created_at" in d

    def test_create_artifact_package_writes_json(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        package = exporter.create_artifact_package(
            records, PromotionBucket.TTS_MAINLINE, tmp_path,
        )
        pkg_path = tmp_path / "artifact_package.json"
        assert pkg_path.exists()

        data = json.loads(pkg_path.read_text())
        assert data["artifact_type"] == "cache_ready_training_bundle"
        assert data["record_count"] == 1
        assert data["checksum"] is not None

    def test_artifact_package_checksum_integrity(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        package = exporter.create_artifact_package(
            records, PromotionBucket.TTS_MAINLINE, tmp_path,
        )
        manifest_path = tmp_path / "manifest.jsonl"
        expected_checksum = _sha256_file(manifest_path)
        assert package.checksum == expected_checksum

    def test_artifact_package_provenance_summary(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record()]
        package = exporter.create_artifact_package(
            records, PromotionBucket.TTS_MAINLINE, tmp_path,
        )
        prov = package.provenance_summary
        assert prov["bucket"] == "tts_mainline"
        assert prov["exported_count"] == 1
        assert "asr" in prov["provider_coverage"]


# ---------------------------------------------------------------------------
# Test: Bucket-specific export behavior
# ---------------------------------------------------------------------------


class TestBucketSpecificExport:
    def test_tts_mainline_full_transcript(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(bucket=PromotionBucket.TTS_MAINLINE)]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["transcript"] == "hello world"

    def test_vc_prior_minimal_text(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(bucket=PromotionBucket.VC_PRIOR)]
        exporter.export_subset(records, PromotionBucket.VC_PRIOR, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert meta["transcript"] == ""
        assert meta["speaker_cluster"] == "spk_01"

    def test_expressive_prior_has_voice_state(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record_with_voice_state(
            bucket=PromotionBucket.EXPRESSIVE_PRIOR,
        )]
        exporter.export_subset(records, PromotionBucket.EXPRESSIVE_PRIOR, tmp_path)

        assert (tmp_path / "rec_vs_001" / "voice_state.npy").exists()
        meta = json.loads((tmp_path / "rec_vs_001" / "meta.json").read_text())
        assert meta["voice_state_supervision"]["has_voice_state"] is True

    def test_holdout_eval_exports_frozen_pairings(self, tmp_path):
        # Need a target and a prompt candidate (same speaker, diff conversation)
        target = _make_record(
            record_id="target_001",
            bucket=PromotionBucket.HOLDOUT_EVAL,
            speaker_cluster="spk_01",
            conversation_id="conv_A",
            duration_sec=5.0,
        )
        prompt_candidate = _make_record(
            record_id="prompt_001",
            bucket=PromotionBucket.TTS_MAINLINE,
            speaker_cluster="spk_01",
            conversation_id="conv_B",
            duration_sec=4.0,
        )
        exporter = CurationExporter()
        package = exporter.export_holdout_eval(
            [target, prompt_candidate], tmp_path,
        )
        pairings_path = tmp_path / "frozen_evaluation_pairings.json"
        assert pairings_path.exists()

        pairings = json.loads(pairings_path.read_text())
        assert len(pairings) == 1
        assert pairings[0]["target_record_id"] == "target_001"
        assert pairings[0]["prompt_record_id"] == "prompt_001"

    def test_holdout_eval_pairings_deterministic(self, tmp_path):
        """Frozen pairings must be reproducible given the same audio_hash."""
        target = _make_record(
            record_id="target_001",
            bucket=PromotionBucket.HOLDOUT_EVAL,
            speaker_cluster="spk_01",
            conversation_id="conv_A",
            audio_hash="fixed_hash_123",
        )
        candidates = [
            _make_record(
                record_id=f"cand_{i}",
                bucket=PromotionBucket.TTS_MAINLINE,
                speaker_cluster="spk_01",
                conversation_id=f"conv_{i}",
                duration_sec=4.0,
            )
            for i in range(5)
        ]
        all_records = [target] + candidates
        exporter = CurationExporter()

        # Export twice to different dirs
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        exporter.export_holdout_eval(all_records, dir1)
        exporter.export_holdout_eval(all_records, dir2)

        p1 = json.loads((dir1 / "frozen_evaluation_pairings.json").read_text())
        p2 = json.loads((dir2 / "frozen_evaluation_pairings.json").read_text())
        assert p1 == p2


# ---------------------------------------------------------------------------
# Test: Bucket-specific convenience exporters
# ---------------------------------------------------------------------------


class TestBucketExporters:
    def test_export_tts_mainline_returns_package(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(bucket=PromotionBucket.TTS_MAINLINE)]
        package = exporter.export_tts_mainline(records, tmp_path)
        assert isinstance(package, ArtifactPackage)
        assert package.artifact_type == ARTIFACT_TYPE_TRAINING_BUNDLE
        assert package.retention_class == RETENTION_DURABLE

    def test_export_vc_prior_returns_package(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(bucket=PromotionBucket.VC_PRIOR)]
        package = exporter.export_vc_prior(records, tmp_path)
        assert isinstance(package, ArtifactPackage)

    def test_export_expressive_prior_returns_package(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(bucket=PromotionBucket.EXPRESSIVE_PRIOR)]
        package = exporter.export_expressive_prior(records, tmp_path)
        assert isinstance(package, ArtifactPackage)

    def test_export_holdout_eval_returns_package(self, tmp_path):
        exporter = CurationExporter()
        records = [_make_record(bucket=PromotionBucket.HOLDOUT_EVAL)]
        package = exporter.export_holdout_eval(records, tmp_path)
        assert isinstance(package, ArtifactPackage)
        assert package.artifact_type == ARTIFACT_TYPE_HOLDOUT_BUNDLE
        assert package.retention_class == RETENTION_RELEASE_CANDIDATE


# ---------------------------------------------------------------------------
# Test: Export does not drop provenance (guardrail)
# ---------------------------------------------------------------------------


class TestProvenanceGuardrails:
    def test_export_preserves_provider_provenance(self, tmp_path):
        """Guardrail: do not drop provenance during export."""
        exporter = CurationExporter()
        record = _make_record(
            providers={
                "asr": Provenance(
                    stage="asr", provider="whisper", version="3.0",
                    timestamp=1000.0, confidence=0.95,
                ),
                "quality": Provenance(
                    stage="quality", provider="scorer_v2", version="2.1",
                    timestamp=1001.0, confidence=0.88,
                ),
            },
        )
        exporter.export_subset([record], PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert "asr" in meta["providers"]
        assert "quality" in meta["providers"]
        assert meta["providers"]["quality"]["version"] == "2.1"

    def test_export_skips_review_items(self, tmp_path):
        """Guardrail: do not export review items into train buckets."""
        exporter = CurationExporter()
        record = _make_record(status=RecordStatus.REVIEW)
        summary = exporter.export_subset(
            [record], PromotionBucket.TTS_MAINLINE, tmp_path,
        )
        assert summary["exported"] == 0

    def test_voice_state_not_exported_without_mask_provenance(self, tmp_path):
        """Guardrail: voice_state supervision must include masks and provenance."""
        exporter = CurationExporter()
        # voice_state without mask/confidence -- still exports the array
        # but meta.json records the absence
        record = _make_record(
            record_id="rec_partial_vs",
            attributes={
                "voice_state": np.zeros((50, 8), dtype=np.float32),
                "voice_state_meta": {
                    "has_voice_state": True,
                    "has_observed_mask": False,
                    "has_confidence": False,
                    "estimator_id": "est_v1",
                    "calibration_version": "cal_1",
                },
            },
        )
        exporter.export_subset([record], PromotionBucket.TTS_MAINLINE, tmp_path)

        vs_meta = json.loads(
            (tmp_path / "rec_partial_vs" / "voice_state_meta.json").read_text()
        )
        assert vs_meta["has_observed_mask"] is False
        assert vs_meta["has_confidence"] is False


# ---------------------------------------------------------------------------
# Test: Export config toggles
# ---------------------------------------------------------------------------


class TestExportConfigToggles:
    def test_disable_voice_state_export(self, tmp_path):
        config = ExportConfig(export_voice_state=False)
        exporter = CurationExporter(config)
        records = [_make_record_with_voice_state()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        assert not (tmp_path / "rec_vs_001" / "voice_state.npy").exists()

    def test_disable_prompt_metadata_export(self, tmp_path):
        config = ExportConfig(export_prompt_metadata=False)
        exporter = CurationExporter(config)
        records = [_make_record_with_prompt_metadata()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_prompt_001" / "meta.json").read_text())
        assert "prompt_eligible" not in meta

    def test_disable_dialogue_context_export(self, tmp_path):
        config = ExportConfig(export_dialogue_context=False)
        exporter = CurationExporter(config)
        records = [_make_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        meta = json.loads((tmp_path / "rec_001" / "meta.json").read_text())
        assert "dialogue_context" not in meta


# ---------------------------------------------------------------------------
# Test: Text suprasegmentals export
# ---------------------------------------------------------------------------


class TestSuprasegmentalsExport:
    def test_text_suprasegmentals_exported(self, tmp_path):
        exporter = CurationExporter()
        supra = np.random.randn(6, 4).astype(np.float32)
        records = [_make_record(attributes={"text_suprasegmentals": supra})]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)

        path = tmp_path / "rec_001" / "text_suprasegmentals.npy"
        assert path.exists()
        loaded = np.load(path)
        np.testing.assert_array_almost_equal(loaded, supra)
