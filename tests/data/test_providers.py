"""Tests for Worker 08: Provider Integration and Model Stack.

Tests the provider adapter layer, registry, voice state estimator,
speaker clustering, and comparison metrics.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from tmrvc_data.curation.models import CurationRecord, Provenance
from tmrvc_data.curation.providers import (
    BaseProvider,
    ProviderOutput,
    ProviderRegistry,
)


# =====================================================================
# Provider base and registry (from existing providers.py)
# =====================================================================


class TestProviderOutput:
    def test_default_values(self):
        out = ProviderOutput(fields={"transcript": "hello"})
        assert out.confidence == 0.0
        assert out.warnings == []
        assert out.provenance is None
        assert out.fields["transcript"] == "hello"

    def test_full_construction(self):
        prov = Provenance(
            stage="asr",
            provider="test",
            version="1.0",
            timestamp=time.time(),
            confidence=0.9,
        )
        out = ProviderOutput(
            fields={"transcript": "hello", "transcript_confidence": 0.9},
            confidence=0.9,
            warnings=["test warning"],
            provenance=prov,
        )
        assert out.confidence == 0.9
        assert len(out.warnings) == 1
        assert out.provenance.provider == "test"


class TestProviderRegistry:
    def test_register_and_retrieve(self):
        from tmrvc_data.curation.providers import create_default_registry

        registry = create_default_registry()
        asr_providers = registry.get_providers("asr")
        assert len(asr_providers) >= 1

    def test_get_primary_returns_first(self):
        registry = ProviderRegistry()
        from tmrvc_data.curation.providers import TranscriptRefiner

        registry.register(TranscriptRefiner())
        primary = registry.get_primary("transcript_refinement")
        assert primary is not None
        assert primary.name == "multi_asr_refiner"

    def test_get_fallback_returns_second(self):
        registry = ProviderRegistry()
        from tmrvc_data.curation.providers import TranscriptRefiner

        r1 = TranscriptRefiner()
        r2 = TranscriptRefiner()
        r2.name = "secondary_refiner"
        registry.register(r1)
        registry.register(r2)
        fallback = registry.get_fallback("transcript_refinement")
        assert fallback is not None
        assert fallback.name == "secondary_refiner"

    def test_missing_stage_returns_none(self):
        registry = ProviderRegistry()
        assert registry.get_primary("nonexistent") is None
        assert registry.get_fallback("nonexistent") is None


# =====================================================================
# Worker 08: ASR provider
# =====================================================================


class TestQwen3ASRProvider:
    def test_instantiation(self):
        from tmrvc_data.curation.providers.asr import Qwen3ASRProvider

        provider = Qwen3ASRProvider()
        assert provider.name == "qwen3_asr"
        assert provider.version == "1.7b-v1"
        assert provider.stage == "asr"
        assert provider.artifact_id == "Qwen/Qwen3-ASR-1.7B"

    def test_normalize_confidence(self):
        from tmrvc_data.curation.providers.asr import Qwen3ASRProvider

        # exp(-0.5) ~ 0.6065
        c = Qwen3ASRProvider.normalize_confidence(-0.5)
        assert 0.60 < c < 0.61

        # exp(0) = 1.0
        assert Qwen3ASRProvider.normalize_confidence(0.0) == 1.0

        # Very negative should clamp to ~0
        c = Qwen3ASRProvider.normalize_confidence(-10.0)
        assert c >= 0.0
        assert c < 0.001

    def test_build_output_empty(self):
        from tmrvc_data.curation.providers.asr import Qwen3ASRProvider

        provider = Qwen3ASRProvider()
        output = provider._build_output([], detected_language="en")
        assert output.confidence == 0.0
        assert output.fields["transcript"] == ""
        assert "No segments" in output.warnings[0]

    def test_build_output_with_segments(self):
        from tmrvc_data.curation.providers.asr import (
            Qwen3ASRProvider,
            ASRSegment,
        )

        provider = Qwen3ASRProvider()
        segments = [
            ASRSegment(
                text="hello world",
                start_sec=0.0,
                end_sec=1.5,
                confidence=0.9,
                language="en",
                word_timestamps=[
                    {"word": "hello", "start": 0.0, "end": 0.7, "probability": 0.95},
                    {"word": "world", "start": 0.7, "end": 1.5, "probability": 0.85},
                ],
            ),
            ASRSegment(
                text="test",
                start_sec=1.5,
                end_sec=2.0,
                confidence=0.8,
                language="en",
            ),
        ]
        output = provider._build_output(segments, detected_language="en")
        assert output.fields["transcript"] == "hello world test"
        assert 0.0 < output.confidence <= 1.0
        assert output.fields["language"] == "en"
        attrs = output.fields["attributes"]
        assert len(attrs["word_timestamps"]) == 2
        assert len(attrs["asr_segments"]) == 2
        assert output.provenance is not None
        assert output.provenance.provider == "qwen3_asr"

    def test_make_provenance(self):
        from tmrvc_data.curation.providers.asr import Qwen3ASRProvider

        provider = Qwen3ASRProvider()
        prov = provider.make_provenance(confidence=0.85)
        assert prov.stage == "asr"
        assert prov.provider == "qwen3_asr"
        assert prov.confidence == 0.85


# =====================================================================
# Worker 08: Diarization provider
# =====================================================================


class TestPyAnnoteDiarizationProvider:
    def test_instantiation(self):
        from tmrvc_data.curation.providers.diarization import (
            PyAnnoteDiarizationProvider,
        )

        provider = PyAnnoteDiarizationProvider()
        assert provider.name == "pyannote_community"
        assert provider.stage == "diarization"

    def test_build_output_empty(self):
        from tmrvc_data.curation.providers.diarization import (
            PyAnnoteDiarizationProvider,
        )

        provider = PyAnnoteDiarizationProvider()
        output = provider._build_output([])
        assert output.confidence == 0.0
        assert output.fields["speaker_cluster"] == "spk_000"

    def test_build_output_with_segments(self):
        from tmrvc_data.curation.providers.diarization import (
            PyAnnoteDiarizationProvider,
            SpeakerSegment,
        )

        provider = PyAnnoteDiarizationProvider()
        segments = [
            SpeakerSegment("spk_A", 0.0, 2.0, confidence=0.9),
            SpeakerSegment("spk_B", 2.0, 3.0, confidence=0.8),
            SpeakerSegment("spk_A", 3.0, 5.0, confidence=0.85),
        ]
        output = provider._build_output(segments, n_speakers=2)
        assert output.fields["speaker_cluster"] == "spk_A"  # most duration
        assert 0.0 < output.confidence <= 1.0
        attrs = output.fields["attributes"]
        assert attrs["n_speakers_detected"] == 2
        assert len(attrs["speaker_turns"]) == 3


# =====================================================================
# Worker 08: Alignment provider
# =====================================================================


class TestQwen3AlignerProvider:
    def test_instantiation(self):
        from tmrvc_data.curation.providers.alignment import Qwen3AlignerProvider

        provider = Qwen3AlignerProvider()
        assert provider.name == "qwen3_aligner"
        assert provider.stage == "alignment"

    def test_canonical_frame_conversion(self):
        from tmrvc_data.curation.providers.alignment import Qwen3AlignerProvider

        # 1 second = 100 frames at 10ms frame shift
        assert Qwen3AlignerProvider.seconds_to_canonical_frame(1.0) == 100
        assert Qwen3AlignerProvider.seconds_to_canonical_frame(0.0) == 0
        assert Qwen3AlignerProvider.seconds_to_canonical_frame(0.5) == 50

        # Roundtrip
        frame = Qwen3AlignerProvider.seconds_to_canonical_frame(0.123)
        back = Qwen3AlignerProvider.canonical_frame_to_seconds(frame)
        assert abs(back - 0.12) < 0.011  # within one frame

    def test_build_output_empty(self):
        from tmrvc_data.curation.providers.alignment import Qwen3AlignerProvider

        provider = Qwen3AlignerProvider()
        output = provider._build_output([])
        assert output.confidence == 0.0
        assert output.fields["attributes"]["phoneme_alignments"] == []

    def test_build_output_with_phonemes(self):
        from tmrvc_data.curation.providers.alignment import (
            Qwen3AlignerProvider,
            PhonemeAlignment,
        )

        provider = Qwen3AlignerProvider()
        phonemes = [
            PhonemeAlignment("h", 0.0, 0.05, 0, 5, confidence=0.9),
            PhonemeAlignment("eh", 0.05, 0.12, 5, 12, confidence=0.85),
            PhonemeAlignment("l", 0.12, 0.18, 12, 18, confidence=0.88),
        ]
        output = provider._build_output(phonemes, language="en")
        assert output.confidence > 0.0
        attrs = output.fields["attributes"]
        assert len(attrs["phoneme_alignments"]) == 3
        assert attrs["alignment_hop_length"] == 240
        assert attrs["alignment_sample_rate"] == 24000


# =====================================================================
# Worker 08: Voice state estimator
# =====================================================================


class TestVoiceStateEstimator:
    def test_instantiation(self):
        from tmrvc_data.curation.providers.voice_state import VoiceStateEstimator

        est = VoiceStateEstimator()
        assert est.name == "voice_state_estimator"
        assert est.stage == "voice_state_estimation"
        assert est.is_available()

    def test_dimension_names(self):
        from tmrvc_data.curation.providers.voice_state import VoiceStateEstimator

        names = VoiceStateEstimator.dimension_names()
        assert len(names) == 12
        assert "pitch_level" in names
        assert "breathiness" in names
        assert "openness" in names

    def test_dimension_index(self):
        from tmrvc_data.curation.providers.voice_state import VoiceStateEstimator, VOICE_STATE_NAMES

        assert VoiceStateEstimator.dimension_index("pitch_level") == 0
        assert VoiceStateEstimator.dimension_index("openness") == VOICE_STATE_NAMES.index("openness")

        with pytest.raises(ValueError, match="Unknown voice state dimension"):
            VoiceStateEstimator.dimension_index("nonexistent")

    def test_process_returns_valid_output(self):
        from tmrvc_data.curation.providers.voice_state import VoiceStateEstimator

        est = VoiceStateEstimator()
        record = CurationRecord(
            record_id="test_vs",
            source_path="/tmp/test.wav",
            audio_hash="abc",
            duration_sec=3.0,
        )
        output = est.process(record)
        assert 0.0 <= output.confidence <= 1.0
        attrs = output.fields["attributes"]
        assert len(attrs["voice_state"]) == 12
        assert len(attrs["voice_state_observed_mask"]) == 12
        assert len(attrs["voice_state_confidence"]) == 12
        assert len(attrs["voice_state_names"]) == 12
        assert output.provenance is not None

    def test_estimate_frames(self):
        from tmrvc_data.curation.providers.voice_state import VoiceStateEstimator

        est = VoiceStateEstimator(sample_rate=24000, hop_length=240)
        # 1 second of audio at 24kHz
        audio = np.random.randn(24000).astype(np.float32) * 0.1
        values, mask, confidence = est.estimate_frames(audio, sr=24000)

        # Should produce frames at canonical hop rate
        assert values.shape[1] == 12
        assert mask.shape == values.shape
        assert confidence.shape == values.shape
        assert values.shape[0] > 0

        # Energy dimension should be observed
        assert np.any(mask[:, 2])  # energy_level is dim 2

        # Values should be in [0, 1]
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


# =====================================================================
# Worker 08: Speaker clustering
# =====================================================================


class TestCrossFileSpeakerClustering:
    def test_instantiation(self):
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
        )

        sc = CrossFileSpeakerClustering()
        assert sc.name == "cross_file_speaker_clustering"
        assert sc.stage == "speaker_clustering"
        assert sc.is_available()

    def test_process_without_embedding(self):
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
        )

        sc = CrossFileSpeakerClustering()
        record = CurationRecord(
            record_id="test_sc",
            source_path="/tmp/test.wav",
            audio_hash="abc",
        )
        output = sc.process(record)
        assert output.confidence == 0.0
        assert "No speaker_embedding" in output.warnings[0]

    def test_process_with_embedding(self):
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
        )

        sc = CrossFileSpeakerClustering(embedding_dim=32)
        emb = np.random.randn(32).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        record = CurationRecord(
            record_id="test_sc",
            source_path="/tmp/test.wav",
            audio_hash="abc",
            speaker_cluster="spk_000",
            duration_sec=5.0,
            attributes={"speaker_embedding": emb.tolist()},
        )
        output = sc.process(record)
        assert output.confidence > 0.0
        assert "gspk_" in output.fields["speaker_cluster"]

    def test_batch_clustering_same_speaker(self):
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
            SpeakerEmbedding,
        )

        sc = CrossFileSpeakerClustering(similarity_threshold=0.7)

        # Create similar embeddings (same speaker)
        base = np.random.randn(32).astype(np.float32)
        base = base / np.linalg.norm(base)

        embeddings = []
        for i in range(5):
            noise = np.random.randn(32).astype(np.float32) * 0.05
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(SpeakerEmbedding(
                record_id=f"rec_{i}",
                file_local_speaker_id=f"spk_{i}",
                embedding=emb,
                duration_sec=3.0,
            ))

        results = sc.cluster_batch(embeddings)
        assert len(results) == 5

        # All should be in the same cluster
        cluster_ids = set(gid for gid, _ in results)
        assert len(cluster_ids) == 1

    def test_batch_clustering_different_speakers(self):
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
            SpeakerEmbedding,
        )

        sc = CrossFileSpeakerClustering(similarity_threshold=0.8)
        sc.reset()

        # Create two distinct speaker embeddings
        spk_a = np.zeros(32, dtype=np.float32)
        spk_a[0] = 1.0
        spk_b = np.zeros(32, dtype=np.float32)
        spk_b[1] = 1.0  # orthogonal to spk_a

        embeddings = [
            SpeakerEmbedding("rec_0", "local_a", spk_a, 3.0),
            SpeakerEmbedding("rec_1", "local_b", spk_b, 3.0),
            SpeakerEmbedding("rec_2", "local_a2", spk_a * 0.99 + np.random.randn(32).astype(np.float32) * 0.01, 3.0),
        ]

        results = sc.cluster_batch(embeddings)
        cluster_ids = [gid for gid, _ in results]

        # First and third should be same cluster, second should differ
        assert cluster_ids[0] == cluster_ids[2]
        assert cluster_ids[0] != cluster_ids[1]

    def test_reset(self):
        from tmrvc_data.curation.providers.speaker_clustering import (
            CrossFileSpeakerClustering,
        )

        sc = CrossFileSpeakerClustering()
        assert len(sc.get_clusters()) == 0
        # Process one record to create a cluster
        record = CurationRecord(
            record_id="test",
            source_path="/tmp/test.wav",
            audio_hash="abc",
            attributes={"speaker_embedding": np.ones(32).tolist()},
        )
        sc.process(record)
        assert len(sc.get_clusters()) == 1
        sc.reset()
        assert len(sc.get_clusters()) == 0


# =====================================================================
# Worker 08: Registry with mainline providers
# =====================================================================


class TestWorker08Registry:
    def test_create_provider_registry(self):
        from tmrvc_data.curation.providers.registry import create_provider_registry

        registry = create_provider_registry()

        # ASR: Qwen3 primary + faster-whisper fallback
        asr = registry.get_providers("asr")
        assert len(asr) >= 2
        assert asr[0].name == "qwen3_asr"

        # Diarization
        dia = registry.get_providers("diarization")
        assert len(dia) >= 1
        assert dia[0].name == "pyannote_community"

        # Alignment
        align = registry.get_providers("alignment")
        assert len(align) >= 1
        assert align[0].name == "qwen3_aligner"

        # Voice state
        vs = registry.get_providers("voice_state_estimation")
        assert len(vs) >= 1
        assert vs[0].name == "voice_state_estimator"

        # Speaker clustering
        sc = registry.get_providers("speaker_clustering")
        assert len(sc) >= 1

    def test_mainline_provider_specs(self):
        from tmrvc_data.curation.providers.registry import MAINLINE_PROVIDERS

        # Required mainline stages
        assert "asr" in MAINLINE_PROVIDERS
        assert "diarization" in MAINLINE_PROVIDERS
        assert "alignment" in MAINLINE_PROVIDERS
        assert "voice_state_estimation" in MAINLINE_PROVIDERS

        # No "latest" or "main" in artifact IDs
        for key, spec in MAINLINE_PROVIDERS.items():
            assert "latest" not in spec.artifact_id.lower(), (
                f"Mainline spec {key} uses 'latest' in artifact_id"
            )
            assert spec.version != "latest"
            assert spec.version != "main"

    def test_validate_registry(self):
        from tmrvc_data.curation.providers.registry import (
            create_provider_registry,
            validate_registry,
        )

        registry = create_provider_registry()
        warnings = validate_registry(registry)
        # Should be clean with default registry
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_validate_registry_detects_missing(self):
        from tmrvc_data.curation.providers.registry import validate_registry

        empty_registry = ProviderRegistry()
        warnings = validate_registry(empty_registry)
        assert len(warnings) > 0  # should flag missing providers


# =====================================================================
# Worker 08: Provider comparison metrics
# =====================================================================


class TestProviderComparisonMetrics:
    def test_asr_agreement_identical(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        a = ProviderOutput(
            fields={"transcript": "hello world"},
            confidence=0.9,
            provenance=Provenance("asr", "a", "1.0", time.time()),
        )
        b = ProviderOutput(
            fields={"transcript": "hello world"},
            confidence=0.85,
            provenance=Provenance("asr", "b", "1.0", time.time()),
        )
        result = ProviderComparisonMetrics.asr_agreement(a, b)
        assert result.value == 1.0
        assert result.metric_name == "asr_word_agreement"

    def test_asr_agreement_partial(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        a = ProviderOutput(
            fields={"transcript": "hello world"},
            confidence=0.9,
            provenance=Provenance("asr", "a", "1.0", time.time()),
        )
        b = ProviderOutput(
            fields={"transcript": "hello there"},
            confidence=0.8,
            provenance=Provenance("asr", "b", "1.0", time.time()),
        )
        result = ProviderComparisonMetrics.asr_agreement(a, b)
        assert 0.0 < result.value < 1.0

    def test_asr_confidence_uplift(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        a = ProviderOutput(fields={}, confidence=0.7)
        b = ProviderOutput(fields={}, confidence=0.9)
        result = ProviderComparisonMetrics.asr_confidence_uplift(a, b)
        assert abs(result.value - 0.2) < 0.001

    def test_diarization_overlap_identical(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        turns = [
            {"start_sec": 0.0, "end_sec": 2.0},
            {"start_sec": 2.5, "end_sec": 4.0},
        ]
        result = ProviderComparisonMetrics.diarization_segment_overlap(
            turns, turns
        )
        assert result.value == 1.0

    def test_diarization_overlap_partial(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        turns_a = [{"start_sec": 0.0, "end_sec": 2.0}]
        turns_b = [{"start_sec": 1.0, "end_sec": 3.0}]
        result = ProviderComparisonMetrics.diarization_segment_overlap(
            turns_a, turns_b
        )
        # 1s overlap / 3s union ~ 0.333
        assert 0.3 < result.value < 0.4

    def test_voice_state_agreement_identical(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        state = [0.5] * 8
        result = ProviderComparisonMetrics.voice_state_agreement(state, state)
        assert result.value == 0.0  # zero difference

    def test_voice_state_agreement_with_mask(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        state_a = [0.5] * 8
        state_b = [0.9] * 8
        mask_a = [True, True, False, False, False, False, False, False]
        mask_b = [True, False, False, False, False, False, False, False]
        result = ProviderComparisonMetrics.voice_state_agreement(
            state_a, state_b, mask_a, mask_b
        )
        # Only dim 0 is jointly observed; diff = 0.4
        assert result.details["n_compared_dims"] == 1
        assert abs(result.value - 0.4) < 0.001

    def test_alignment_timing_deviation(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        phonemes_a = [
            {"phoneme": "h", "start_sec": 0.0, "end_sec": 0.05},
            {"phoneme": "eh", "start_sec": 0.05, "end_sec": 0.12},
        ]
        phonemes_b = [
            {"phoneme": "h", "start_sec": 0.01, "end_sec": 0.06},
            {"phoneme": "eh", "start_sec": 0.06, "end_sec": 0.13},
        ]
        result = ProviderComparisonMetrics.alignment_timing_deviation(
            phonemes_a, phonemes_b
        )
        assert result.value == 0.01  # consistent 10ms offset

    def test_speaker_clustering_purity_perfect(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        predicted = ["a", "a", "b", "b"]
        reference = ["x", "x", "y", "y"]
        result = ProviderComparisonMetrics.speaker_clustering_purity(
            predicted, reference
        )
        assert result.value == 1.0

    def test_speaker_clustering_purity_impure(self):
        from tmrvc_data.curation.providers.comparison import (
            ProviderComparisonMetrics,
        )

        predicted = ["a", "a", "a", "a"]  # all in one cluster
        reference = ["x", "x", "y", "y"]  # two classes
        result = ProviderComparisonMetrics.speaker_clustering_purity(
            predicted, reference
        )
        assert result.value == 0.5  # max class in single cluster = 2/4


# =====================================================================
# Worker 08: ProviderSpec dataclass
# =====================================================================


class TestProviderSpec:
    def test_provider_spec_fields(self):
        from tmrvc_data.curation.providers.registry import ProviderSpec

        spec = ProviderSpec(
            stage="asr",
            provider_id="test_asr",
            artifact_id="test/asr-model",
            version="1.0.0",
            runtime_backend="transformers",
            supported_languages=["en", "zh"],
            license_status="apache-2.0",
            gated_access=False,
            fallback_policy="downgrade",
        )
        assert spec.stage == "asr"
        assert spec.provider_id == "test_asr"
        assert "en" in spec.supported_languages
        assert spec.fallback_policy == "downgrade"
