"""Tests for the AI Curation System (Workers 07-11)."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from tmrvc_data.curation.models import (
    CurationRecord,
    RecordStatus,
    PromotionBucket,
    Provenance,
    LegalityStatus,
)
from tmrvc_data.curation.orchestrator import CurationOrchestrator
from tmrvc_data.curation.scoring import (
    QualityScoringEngine,
    ScoringConfig,
    BucketThresholds,
    DEFAULT_THRESHOLDS,
)
from tmrvc_data.curation.export import CurationExporter, ExportConfig
from tmrvc_data.curation.validation import CurationValidator, ValidationConfig
from tmrvc_data.curation.providers import (
    ProviderRegistry,
    TranscriptRefiner,
    create_default_registry,
    ProviderOutput,
)


# ---------------------------------------------------------------------------
# Worker 07: Models and Orchestrator
# ---------------------------------------------------------------------------


class TestCurationRecord:
    def test_round_trip_serialization(self):
        record = CurationRecord(
            record_id="test_001",
            source_path="/tmp/audio.wav",
            audio_hash="abc123",
            duration_sec=3.5,
            transcript="hello world",
            transcript_confidence=0.95,
            source_legality="owned",
            conversation_id="conv_001",
            turn_index=2,
            prev_record_id="test_000",
            next_record_id="test_002",
            context_window_ids=["test_000"],
        )
        d = record.to_dict()
        restored = CurationRecord.from_dict(d)
        assert restored.record_id == "test_001"
        assert restored.source_legality == "owned"
        assert restored.conversation_id == "conv_001"
        assert restored.turn_index == 2
        assert restored.prev_record_id == "test_000"
        assert restored.next_record_id == "test_002"
        assert restored.context_window_ids == ["test_000"]

    def test_default_legality_is_unknown(self):
        record = CurationRecord(
            record_id="r", source_path="p", audio_hash="h"
        )
        assert record.source_legality == "unknown"

    def test_status_enum_values(self):
        assert RecordStatus.INGESTED.value == "ingested"
        assert RecordStatus.PROMOTED.value == "promoted"
        assert RecordStatus.REJECTED.value == "rejected"

    def test_promotion_bucket_values(self):
        assert PromotionBucket.TTS_MAINLINE.value == "tts_mainline"
        assert PromotionBucket.HOLDOUT_EVAL.value == "holdout_eval"

    def test_legality_status_values(self):
        assert LegalityStatus.OWNED.value == "owned"
        assert LegalityStatus.UNKNOWN.value == "unknown"

    def test_provenance_serialization(self):
        record = CurationRecord(
            record_id="r", source_path="p", audio_hash="h",
            providers={
                "asr": Provenance(
                    stage="asr", provider="whisper",
                    version="1.0", timestamp=1000.0,
                    confidence=0.9, metadata={"model": "large-v3"},
                )
            },
        )
        d = record.to_dict()
        assert d["providers"]["asr"]["provider"] == "whisper"
        restored = CurationRecord.from_dict(d)
        assert restored.providers["asr"].confidence == 0.9


class TestOrchestrator:
    def test_save_and_load_manifest(self, tmp_path):
        orch = CurationOrchestrator(tmp_path)
        record = CurationRecord(
            record_id="r1", source_path="/a.wav", audio_hash="h1",
            source_legality="owned", transcript="test",
        )
        orch.update_record(record)
        orch.save_manifest()

        orch2 = CurationOrchestrator(tmp_path)
        assert "r1" in orch2.records
        assert orch2.records["r1"].source_legality == "owned"

    def test_summary_updated_on_save(self, tmp_path):
        orch = CurationOrchestrator(tmp_path)
        orch.update_record(CurationRecord(
            record_id="r1", source_path="p", audio_hash="h",
        ))
        orch.save_manifest()
        summary = json.loads(orch.summary_path.read_text())
        assert summary["total_records"] == 1

    def test_run_stage_skips_processed(self, tmp_path):
        orch = CurationOrchestrator(tmp_path)
        record = CurationRecord(
            record_id="r1", source_path="p", audio_hash="h",
            providers={"test_stage": Provenance(
                stage="test_stage", provider="p", version="1", timestamp=0,
            )},
        )
        orch.update_record(record)
        call_count = [0]

        def processor(r):
            call_count[0] += 1
            return r

        orch.run_stage("test_stage", processor, force=False)
        assert call_count[0] == 0  # skipped

        orch.run_stage("test_stage", processor, force=True)
        assert call_count[0] == 1  # forced


# ---------------------------------------------------------------------------
# Worker 08: Providers
# ---------------------------------------------------------------------------


class TestProviders:
    def test_create_default_registry(self):
        registry = create_default_registry()
        assert len(registry.get_providers("asr")) >= 1
        assert len(registry.get_providers("transcript_refinement")) >= 1

    def test_transcript_refiner_majority_vote(self):
        refiner = TranscriptRefiner()
        assert refiner.is_available()
        output = refiner.process(
            CurationRecord(record_id="r", source_path="p", audio_hash="h"),
            asr_outputs=[
                {"text": "hello", "confidence": 0.9},
                {"text": "hello", "confidence": 0.8},
                {"text": "helo", "confidence": 0.7},
            ],
        )
        assert output.fields["transcript"] == "hello"
        assert output.confidence > 0.5

    def test_transcript_refiner_empty_outputs(self):
        refiner = TranscriptRefiner()
        output = refiner.process(
            CurationRecord(record_id="r", source_path="p", audio_hash="h"),
            asr_outputs=[],
        )
        assert output.confidence == 0.0
        assert len(output.warnings) > 0

    def test_provider_registry_primary_fallback(self):
        registry = ProviderRegistry()
        refiner = TranscriptRefiner()
        registry.register(refiner)
        primary = registry.get_primary("transcript_refinement")
        assert primary is refiner
        assert registry.get_fallback("transcript_refinement") is None
        assert registry.get_primary("nonexistent") is None


# ---------------------------------------------------------------------------
# Worker 09: Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def _make_record(self, **overrides):
        defaults = dict(
            record_id="r1", source_path="p", audio_hash="h",
            transcript="hello", transcript_confidence=0.95,
            diarization_confidence=0.9, duration_sec=5.0,
            source_legality="owned", language="ja",
            quality_score=0.0,
            attributes={
                "snr_db": 30.0, "refinement_agreement": 0.95,
                "pause_events": [{"type": "pause"}],
            },
        )
        defaults.update(overrides)
        attrs = defaults.pop("attributes", {})
        r = CurationRecord(**defaults)
        r.attributes = attrs
        return r

    def test_compute_score_high_quality(self):
        engine = QualityScoringEngine()
        record = self._make_record()
        score = engine.compute_score(record)
        assert 0.5 < score <= 1.0

    def test_hard_reject_empty_transcript(self):
        engine = QualityScoringEngine()
        record = self._make_record(transcript=None)
        reasons = engine.check_hard_reject(record)
        assert "transcript_empty" in reasons

    def test_hard_reject_invalid_duration(self):
        engine = QualityScoringEngine()
        record = self._make_record(duration_sec=0.0)
        reasons = engine.check_hard_reject(record)
        assert "invalid_duration" in reasons

    def test_review_marginal_confidence(self):
        engine = QualityScoringEngine()
        record = self._make_record(transcript_confidence=0.5)
        reasons = engine.check_review(record)
        assert "marginal_transcript_confidence" in reasons

    def test_promote_to_tts_mainline(self):
        engine = QualityScoringEngine()
        record = self._make_record()
        result = engine.score_and_decide(record)
        assert result.status == RecordStatus.PROMOTED
        assert result.promotion_bucket == PromotionBucket.TTS_MAINLINE

    def test_reject_blocks_promotion(self):
        engine = QualityScoringEngine()
        record = self._make_record(transcript=None)
        result = engine.score_and_decide(record)
        assert result.status == RecordStatus.REJECTED

    def test_unknown_legality_blocks_mainline(self):
        engine = QualityScoringEngine()
        record = self._make_record(source_legality="unknown")
        bucket = engine.determine_bucket(record)
        assert bucket != PromotionBucket.TTS_MAINLINE

    def test_holdout_anti_contamination(self):
        config = ScoringConfig(holdout_record_ids={"r1"})
        engine = QualityScoringEngine(config)
        record = self._make_record()
        bucket = engine.determine_bucket(record)
        assert bucket == PromotionBucket.HOLDOUT_EVAL

    def test_generate_report(self):
        engine = QualityScoringEngine()
        records = [engine.score_and_decide(self._make_record(record_id=f"r{i}"))
                   for i in range(5)]
        report = engine.generate_report(records)
        assert report["total"] == 5
        assert "status_distribution" in report
        assert "bucket_distribution" in report

    def test_default_thresholds_present(self):
        assert "tts_mainline" in DEFAULT_THRESHOLDS
        assert "vc_prior" in DEFAULT_THRESHOLDS
        assert "expressive_prior" in DEFAULT_THRESHOLDS
        assert "holdout_eval" in DEFAULT_THRESHOLDS


# ---------------------------------------------------------------------------
# Worker 10: Export
# ---------------------------------------------------------------------------


class TestExport:
    def _make_promoted_record(self, record_id="r1", bucket=PromotionBucket.TTS_MAINLINE):
        r = CurationRecord(
            record_id=record_id, source_path="/a.wav", audio_hash="h",
            transcript="hello", language="ja", quality_score=0.9,
            status=RecordStatus.PROMOTED, promotion_bucket=bucket,
            source_legality="owned",
            conversation_id="conv1", turn_index=1,
        )
        r.attributes["phoneme_ids_list"] = [10, 11, 12, 20, 21]
        return r

    def test_export_subset_writes_manifest(self, tmp_path):
        exporter = CurationExporter()
        records = [self._make_promoted_record()]
        summary = exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)
        assert summary["exported"] == 1
        assert (tmp_path / "manifest.jsonl").exists()

    def test_export_subset_writes_meta_json(self, tmp_path):
        exporter = CurationExporter()
        records = [self._make_promoted_record()]
        exporter.export_subset(records, PromotionBucket.TTS_MAINLINE, tmp_path)
        meta = json.loads((tmp_path / "r1" / "meta.json").read_text())
        assert meta["transcript"] == "hello"
        assert meta["conversation_id"] == "conv1"

    def test_export_skips_non_promoted(self, tmp_path):
        exporter = CurationExporter()
        record = CurationRecord(
            record_id="r1", source_path="p", audio_hash="h",
            status=RecordStatus.REVIEW,
        )
        summary = exporter.export_subset([record], PromotionBucket.TTS_MAINLINE, tmp_path)
        assert summary["exported"] == 0

    def test_export_all_buckets(self, tmp_path):
        exporter = CurationExporter()
        records = [
            self._make_promoted_record("r1", PromotionBucket.TTS_MAINLINE),
            self._make_promoted_record("r2", PromotionBucket.VC_PRIOR),
        ]
        results = exporter.export_all_buckets(records, tmp_path)
        assert results["tts_mainline"]["exported"] == 1
        assert results["vc_prior"]["exported"] == 1


# ---------------------------------------------------------------------------
# Worker 11: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def _make_records(self):
        return [
            CurationRecord(
                record_id=f"r{i}", source_path="p", audio_hash="h",
                status=RecordStatus.PROMOTED,
                promotion_bucket=PromotionBucket.TTS_MAINLINE,
                source_legality="owned",
                quality_score=0.9,
                providers={"asr": Provenance(
                    stage="asr", provider="w", version="1", timestamp=0,
                )},
            )
            for i in range(10)
        ]

    def test_promotion_distribution_passes(self):
        validator = CurationValidator()
        result = validator.validate_promotion_distribution(self._make_records())
        assert result["pass"] is True

    def test_promotion_distribution_fails_empty(self):
        validator = CurationValidator()
        result = validator.validate_promotion_distribution([])
        assert result["pass"] is False

    def test_legality_gating_passes(self):
        validator = CurationValidator()
        result = validator.validate_legality_gating(self._make_records())
        assert result["pass"] is True

    def test_legality_gating_catches_unknown(self):
        validator = CurationValidator()
        records = self._make_records()
        records[0].source_legality = "unknown"
        result = validator.validate_legality_gating(records)
        assert result["pass"] is False
        assert len(result["violations"]) == 1

    def test_holdout_integrity_passes(self):
        validator = CurationValidator()
        result = validator.validate_holdout_integrity(self._make_records())
        assert result["pass"] is True

    def test_holdout_leak_detected(self):
        validator = CurationValidator()
        records = self._make_records()
        result = validator.validate_holdout_integrity(
            records, holdout_ids={"r0"}
        )
        assert result["pass"] is False
        assert "r0" in result["leaks"]

    def test_provenance_completeness_passes(self):
        validator = CurationValidator()
        result = validator.validate_provenance_completeness(self._make_records())
        assert result["pass"] is True

    def test_provenance_incomplete_detected(self):
        validator = CurationValidator()
        records = self._make_records()
        records[0].providers = {}
        result = validator.validate_provenance_completeness(records)
        assert result["pass"] is False

    def test_run_all(self):
        validator = CurationValidator()
        report = validator.run_all(self._make_records())
        assert report["overall"]["pass"] is True
        assert "promotion_distribution" in report
        assert "legality_gating" in report

    def test_save_report(self, tmp_path):
        validator = CurationValidator()
        report = validator.run_all(self._make_records())
        path = tmp_path / "report.json"
        validator.save_report(report, path)
        loaded = json.loads(path.read_text())
        assert loaded["overall"]["pass"] is True


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCurationCLI:
    def test_cli_parser_builds(self):
        from tmrvc_data.cli.curate import build_parser
        parser = build_parser()
        args = parser.parse_args(["--output-dir", "/tmp/test", "summary"])
        assert args.command == "summary"

    def test_cli_ingest_args(self):
        from tmrvc_data.cli.curate import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "ingest", "--input-dir", "/tmp/audio",
        ])
        assert args.command == "ingest"

    def test_cli_score_args(self):
        from tmrvc_data.cli.curate import build_parser
        parser = build_parser()
        args = parser.parse_args(["score"])
        assert args.command == "score"

    def test_cli_export_args(self):
        from tmrvc_data.cli.curate import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "export", "--export-dir", "/tmp/out",
        ])
        assert args.command == "export"

    def test_cli_validate_args(self):
        from tmrvc_data.cli.curate import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "validate", "--report-path", "/tmp/report.json",
        ])
        assert args.command == "validate"
