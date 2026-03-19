"""Export curated subsets to TMRVC training-compatible cache format.

Implements Worker 10: converts promoted records into trainable assets.

Exports include:
- Cache-ready meta.json with curation provenance
- phoneme_ids.npy, codec tokens
- voice_state.npy, voice_state_observed_mask.npy, voice_state_confidence.npy
- bootstrap_alignment.json projected to canonical phoneme space
- Dialogue context fields for context-conditioned batching
- Few-shot prompt eligibility metadata
- Artifact package contract with checksums and provenance

Frame convention (frozen):
    sample_rate = 24000
    hop_length = 240
    start_frame inclusive, end_frame exclusive
    T = ceil(num_samples / 240)

Voice state dimensions (12-D, canonical order):
    pitch_level, pitch_range, energy_level, pressedness,
    spectral_tilt, breathiness, voice_irregularity, openness,
    aperiodicity, formant_shift, vocal_effort, creak
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tmrvc_data.bootstrap_alignment import (
    BootstrapAlignment,
    project_word_timestamps_acoustic,
    SAMPLE_RATE,
    HOP_LENGTH,
)
from .models import CurationRecord, PromotionBucket, RecordStatus

logger = logging.getLogger(__name__)

# Canonical voice_state dimension names (Worker 03 contract, export order).
VOICE_STATE_DIM_NAMES = (
    "pitch_level",
    "pitch_range",
    "energy_level",
    "pressedness",
    "spectral_tilt",
    "breathiness",
    "voice_irregularity",
    "openness",
    "aperiodicity",
    "formant_shift",
    "vocal_effort",
    "creak",
)
VOICE_STATE_NDIM = len(VOICE_STATE_DIM_NAMES)

# Artifact retention classes
RETENTION_EPHEMERAL = "ephemeral"
RETENTION_DURABLE = "durable"
RETENTION_RELEASE_CANDIDATE = "release_candidate"

# Artifact types
ARTIFACT_TYPE_TRAINING_BUNDLE = "cache_ready_training_bundle"
ARTIFACT_TYPE_HOLDOUT_BUNDLE = "holdout_evaluation_bundle"
ARTIFACT_TYPE_WORKSHOP_BUNDLE = "pinned_workshop_take_bundle"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class ArtifactPackage:
    """Artifact package contract (Worker 10 task 12).

    Every exported package carries identity, provenance, and retention
    metadata so that downstream consumers can audit and manage artifacts.
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: str = ARTIFACT_TYPE_TRAINING_BUNDLE
    created_at: str = field(default_factory=_utc_now_iso)
    source_dataset_ids: List[str] = field(default_factory=list)
    manifest_snapshot_id: Optional[str] = None
    policy_version: str = "1.0.0"
    provenance_summary: Dict[str, Any] = field(default_factory=dict)
    retention_class: str = RETENTION_DURABLE
    checksum: Optional[str] = None
    record_count: int = 0
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at,
            "source_dataset_ids": self.source_dataset_ids,
            "manifest_snapshot_id": self.manifest_snapshot_id,
            "policy_version": self.policy_version,
            "provenance_summary": self.provenance_summary,
            "retention_class": self.retention_class,
            "checksum": self.checksum,
            "record_count": self.record_count,
            "output_dir": self.output_dir,
        }


@dataclass
class ExportConfig:
    """Configuration for curation export."""

    output_dir: Path = field(default_factory=lambda: Path("data/curated_export"))
    export_style_embedding: bool = True
    export_events: bool = True
    export_dialogue_graph: bool = True
    export_bootstrap_alignment: bool = True
    export_voice_state: bool = True
    export_prompt_metadata: bool = True
    export_dialogue_context: bool = True


class CurationExporter:
    """Exports promoted curation subsets to TMRVC cache-compatible format."""

    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.config = config or ExportConfig()

    def export_subset(
        self,
        records: List[CurationRecord],
        bucket: PromotionBucket,
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Export records from a specific bucket to cache format.

        Returns export summary dict.
        """
        out = (
            Path(output_dir)
            if output_dir
            else self.config.output_dir / bucket.value
        )
        out.mkdir(parents=True, exist_ok=True)

        eligible = [
            r
            for r in records
            if r.status == RecordStatus.PROMOTED
            and r.promotion_bucket == bucket
        ]

        if not eligible:
            logger.warning("No records eligible for bucket %s", bucket.value)
            return {"bucket": bucket.value, "exported": 0}

        manifest_entries: List[Dict[str, Any]] = []
        for record in eligible:
            entry = self._record_to_export(record, bucket)
            manifest_entries.append(entry)

            # Write per-record meta.json
            record_dir = out / record.record_id
            record_dir.mkdir(parents=True, exist_ok=True)
            meta_path = record_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)

            # Export bootstrap alignment if enabled
            if self.config.export_bootstrap_alignment:
                self._export_bootstrap_alignment(record, record_dir)

            # Export voice_state supervision artifacts
            if self.config.export_voice_state:
                self._export_voice_state(record, record_dir, bucket)

            # Export phoneme_ids.npy if available (Enforced for tts_mainline)
            self._export_phoneme_ids(record, record_dir, bucket)

        # Write manifest
        manifest_path = out / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        summary: Dict[str, Any] = {
            "bucket": bucket.value,
            "exported": len(manifest_entries),
            "output_dir": str(out),
        }

        summary_path = out / "export_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Exported %d records to %s (bucket=%s)",
            len(manifest_entries),
            out,
            bucket.value,
        )
        return summary

    def _export_bootstrap_alignment(self, record: CurationRecord, record_dir: Path) -> None:
        """Project word timestamps and export bootstrap_alignment.json with acoustic refinement."""
        word_ts = record.attributes.get("word_timestamps")
        phoneme_ids = record.attributes.get("phoneme_ids_list") # list of ints
        w2p_map = record.attributes.get("word_to_phoneme_map")
        num_samples = record.attributes.get("num_samples")
        
        if not all([word_ts, phoneme_ids, w2p_map, num_samples]):
            return

        # Attempt acoustic-aware projection (Worker 10: Boundary Refinement)
        import torch
        ef_path = record_dir / "energy_flux.npy"
        # If energy_flux doesn't exist, we fallback to coarse timestamps
        # Energy flux/Spectral flux is used to sharpen boundaries.
        energy_flux = torch.from_numpy(np.load(ef_path)) if ef_path.exists() else None
        
        alignment = project_word_timestamps_acoustic(
            word_ts, phoneme_ids, w2p_map, num_samples, energy_flux=energy_flux
        )
        
        errors = alignment.validate()
        if errors:
            logger.warning("Alignment validation failed for %s: %s", record.record_id, errors)
            
        align_path = record_dir / "bootstrap_alignment.json"
        with open(align_path, "w", encoding="utf-8") as f:
            # We add versioning to the artifact so Worker 03 can track drift.
            data = alignment.to_dict()
            data["algorithm_version"] = "1.1.0-refined"
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _record_to_export(
        self, record: CurationRecord, bucket: PromotionBucket
    ) -> Dict[str, Any]:
        """Convert a CurationRecord to export-ready dict with provenance.

        Applies bucket-specific export behavior:
        - tts_mainline: full export with all supervision
        - vc_prior: audio + speaker identity, minimal text
        - expressive_prior: audio + prosody + voice_state
        - holdout_eval: full export + frozen prompt-target pairings
        """
        entry: Dict[str, Any] = {
            "record_id": record.record_id,
            "source_path": record.source_path,
            "speaker_cluster": record.speaker_cluster,
            "quality_score": record.quality_score,
            "source_legality": record.source_legality,
            "promotion_bucket": bucket.value,
            "segment_start_sec": record.segment_start_sec,
            "segment_end_sec": record.segment_end_sec,
            "duration_sec": record.duration_sec,
            # Curation provenance (Worker 10 cache-ready contract)
            "curation_record_id": record.record_id,
            "curation_pass": record.pass_index,
            # Provider provenance (Worker 07/09 requirement)
            "providers": {
                k: {
                    "provider": v.provider,
                    "version": v.version,
                    "confidence": v.confidence,
                }
                for k, v in record.providers.items()
            },
            # Score components if available
            "score_components": record.attributes.get("score_components", {}),
        }

        # Bucket-specific text handling
        if bucket == PromotionBucket.VC_PRIOR:
            # VC prior: minimal text -- speaker identity is primary
            entry["transcript"] = ""
            entry["language"] = record.language or ""
        else:
            # All other buckets get full text
            entry["transcript"] = record.transcript or ""
            entry["language"] = record.language or ""

        # Dialogue graph fields
        if self.config.export_dialogue_graph:
            entry["conversation_id"] = record.conversation_id
            entry["turn_index"] = record.turn_index
            entry["prev_record_id"] = record.prev_record_id
            entry["next_record_id"] = record.next_record_id
            entry["context_window_ids"] = record.context_window_ids

        # Dialogue context export (model-agnostic raw text)
        if self.config.export_dialogue_context and record.conversation_id:
            entry["dialogue_context"] = {
                "conversation_id": record.conversation_id,
                "turn_index": record.turn_index,
                "raw_text": record.transcript or "",
                "prev_record_id": record.prev_record_id,
                "next_record_id": record.next_record_id,
                "context_window_ids": record.context_window_ids,
            }

        # Events
        if self.config.export_events:
            entry["pause_events"] = record.attributes.get("pause_events", [])
            entry["breath_events"] = record.attributes.get(
                "breath_events", []
            )

        # Style embedding reference
        if self.config.export_style_embedding:
            entry["has_style_embedding"] = (
                "style_embedding" in record.attributes
            )

        # Few-shot prompt metadata
        if self.config.export_prompt_metadata:
            entry["prompt_eligible"] = record.attributes.get(
                "prompt_eligible", False
            )
            entry["prompt_pair_id"] = record.attributes.get(
                "prompt_pair_id", None
            )
            # Extended prompt metadata when available
            prompt_meta = record.attributes.get("prompt_metadata", {})
            if prompt_meta:
                entry["prompt_metadata"] = {
                    "speaker_id": prompt_meta.get("speaker_id", record.speaker_cluster),
                    "prompt_candidate_record_ids": prompt_meta.get(
                        "prompt_candidate_record_ids", []
                    ),
                    "prompt_policy_version": prompt_meta.get(
                        "prompt_policy_version", "1.0.0"
                    ),
                    "prompt_duration_sec": prompt_meta.get(
                        "prompt_duration_sec", record.duration_sec
                    ),
                    "prompt_language": prompt_meta.get(
                        "prompt_language", record.language
                    ),
                    "speaker_purity_estimate": prompt_meta.get(
                        "speaker_purity_estimate", None
                    ),
                    "leakage_policy_flags": prompt_meta.get(
                        "leakage_policy_flags", {}
                    ),
                }

        # Voice state metadata (reference -- actual arrays exported separately)
        if self.config.export_voice_state:
            vs_meta = record.attributes.get("voice_state_meta", {})
            entry["voice_state_supervision"] = {
                "has_voice_state": "voice_state" in record.attributes
                or vs_meta.get("has_voice_state", False),
                "has_observed_mask": vs_meta.get("has_observed_mask", False),
                "has_confidence": vs_meta.get("has_confidence", False),
                "estimator_id": vs_meta.get("estimator_id", None),
                "calibration_version": vs_meta.get("calibration_version", None),
                "dimensions": list(VOICE_STATE_DIM_NAMES),
            }

        return entry

    # ------------------------------------------------------------------
    # Voice state export (Worker 10 task 10)
    # ------------------------------------------------------------------

    def _export_voice_state(
        self,
        record: CurationRecord,
        record_dir: Path,
        bucket: PromotionBucket,
    ) -> None:
        """Export voice_state supervision artifacts.

        Writes:
        - voice_state.npy: [T_frames, 8] float32
        - voice_state_observed_mask.npy: [T_frames, 8] bool
        - voice_state_confidence.npy: [T_frames, 8] float32
        - voice_state_meta.json: estimator identity and provenance
        """
        vs_data = record.attributes.get("voice_state")
        if vs_data is None:
            return

        vs_array = np.asarray(vs_data, dtype=np.float32)
        if vs_array.ndim != 2 or vs_array.shape[1] != VOICE_STATE_NDIM:
            logger.warning(
                "Invalid voice_state shape %s for %s, expected [T, %d]",
                vs_array.shape, record.record_id, VOICE_STATE_NDIM,
            )
            return

        np.save(record_dir / "voice_state.npy", vs_array)

        # Observed mask
        mask_data = record.attributes.get("voice_state_observed_mask")
        if mask_data is not None:
            mask_array = np.asarray(mask_data, dtype=bool)
            if mask_array.shape == vs_array.shape:
                np.save(record_dir / "voice_state_observed_mask.npy", mask_array)
            else:
                logger.warning(
                    "voice_state_observed_mask shape mismatch for %s: %s vs %s",
                    record.record_id, mask_array.shape, vs_array.shape,
                )

        # Confidence
        conf_data = record.attributes.get("voice_state_confidence")
        if conf_data is not None:
            conf_array = np.asarray(conf_data, dtype=np.float32)
            if conf_array.shape == vs_array.shape:
                np.save(record_dir / "voice_state_confidence.npy", conf_array)
            else:
                logger.warning(
                    "voice_state_confidence shape mismatch for %s: %s vs %s",
                    record.record_id, conf_array.shape, vs_array.shape,
                )

        # Voice state meta
        vs_meta = record.attributes.get("voice_state_meta", {})
        meta = {
            "dimensions": list(VOICE_STATE_DIM_NAMES),
            "n_frames": int(vs_array.shape[0]),
            "has_observed_mask": mask_data is not None,
            "has_confidence": conf_data is not None,
            "estimator_id": vs_meta.get("estimator_id", "unknown"),
            "calibration_version": vs_meta.get("calibration_version", "unknown"),
            "target_source_provenance": vs_meta.get("target_source_provenance", "curation_export"),
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
        }
        with open(record_dir / "voice_state_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Phoneme IDs export (Enforced by Worker 10 / GEMINI.md)
    # ------------------------------------------------------------------

    def _export_phoneme_ids(
        self, record: CurationRecord, record_dir: Path, bucket: PromotionBucket
    ) -> None:
        """Export phoneme_ids.npy from record attributes.

        SOTA Mandate: phoneme_ids.npy is REQUIRED for tts_mainline.
        If missing, we fail fast to prevent G2P skipping during preprocessing.
        """
        phoneme_ids = record.attributes.get("phoneme_ids_list")

        if bucket == PromotionBucket.TTS_MAINLINE and phoneme_ids is None:
            # Check if transcript exists to attempt emergency G2P or fail.
            # Charter says G2P skipping is forbidden, so we stop here.
            msg = f"FATAL: Missing phoneme_ids for tts_mainline record {record.record_id}. " \
                  f"Preprocessing skip-G2P is forbidden by GEMINI.md."
            logger.error(msg)
            raise ValueError(msg)

        if phoneme_ids is not None:
            arr = np.asarray(phoneme_ids, dtype=np.int64)
            np.save(record_dir / "phoneme_ids.npy", arr)

        # Export text_suprasegmentals if present (Optional but recommended)
        supra = record.attributes.get("text_suprasegmentals")
        if supra is not None:
            supra_arr = np.asarray(supra, dtype=np.float32)
            np.save(record_dir / "text_suprasegmentals.npy", supra_arr)

    # ------------------------------------------------------------------
    # Artifact package (Worker 10 task 12)
    # ------------------------------------------------------------------

    def create_artifact_package(
        self,
        records: List[CurationRecord],
        bucket: PromotionBucket,
        output_dir: Union[Path, str],
        artifact_type: str = ARTIFACT_TYPE_TRAINING_BUNDLE,
        retention_class: str = RETENTION_DURABLE,
        source_dataset_ids: Optional[List[str]] = None,
        manifest_snapshot_id: Optional[str] = None,
        policy_version: str = "1.0.0",
    ) -> ArtifactPackage:
        """Export records and wrap in an artifact package with provenance.

        This is the primary entry point for producing auditable export
        packages. It calls export_subset internally, then writes the
        artifact_package.json contract file.
        """
        out = Path(output_dir)
        summary = self.export_subset(records, bucket, out)

        # Compute manifest checksum
        manifest_path = out / "manifest.jsonl"
        checksum = _sha256_file(manifest_path) if manifest_path.exists() else None

        # Provenance summary
        provider_summary: Dict[str, int] = {}
        for r in records:
            if r.status == RecordStatus.PROMOTED and r.promotion_bucket == bucket:
                for k in r.providers:
                    provider_summary[k] = provider_summary.get(k, 0) + 1

        package = ArtifactPackage(
            artifact_type=artifact_type,
            source_dataset_ids=source_dataset_ids or [],
            manifest_snapshot_id=manifest_snapshot_id,
            policy_version=policy_version,
            retention_class=retention_class,
            checksum=checksum,
            record_count=summary.get("exported", 0),
            output_dir=str(out),
            provenance_summary={
                "bucket": bucket.value,
                "provider_coverage": provider_summary,
                "exported_count": summary.get("exported", 0),
            },
        )

        # Write artifact_package.json
        pkg_path = out / "artifact_package.json"
        with open(pkg_path, "w", encoding="utf-8") as f:
            json.dump(package.to_dict(), f, indent=2)

        logger.info(
            "Created artifact package %s (%s, %d records)",
            package.artifact_id, artifact_type, package.record_count,
        )
        return package

    # ------------------------------------------------------------------
    # Bucket-specific export helpers
    # ------------------------------------------------------------------

    def export_tts_mainline(
        self,
        records: List[CurationRecord],
        output_dir: Union[Path, str],
    ) -> ArtifactPackage:
        """Export tts_mainline bucket: full export with all supervision."""
        return self.create_artifact_package(
            records,
            PromotionBucket.TTS_MAINLINE,
            output_dir,
            artifact_type=ARTIFACT_TYPE_TRAINING_BUNDLE,
            retention_class=RETENTION_DURABLE,
        )

    def export_vc_prior(
        self,
        records: List[CurationRecord],
        output_dir: Union[Path, str],
    ) -> ArtifactPackage:
        """Export vc_prior bucket: audio + speaker identity, minimal text."""
        return self.create_artifact_package(
            records,
            PromotionBucket.VC_PRIOR,
            output_dir,
            artifact_type=ARTIFACT_TYPE_TRAINING_BUNDLE,
            retention_class=RETENTION_DURABLE,
        )

    def export_expressive_prior(
        self,
        records: List[CurationRecord],
        output_dir: Union[Path, str],
    ) -> ArtifactPackage:
        """Export expressive_prior bucket: audio + prosody + voice_state."""
        return self.create_artifact_package(
            records,
            PromotionBucket.EXPRESSIVE_PRIOR,
            output_dir,
            artifact_type=ARTIFACT_TYPE_TRAINING_BUNDLE,
            retention_class=RETENTION_DURABLE,
        )

    def export_holdout_eval(
        self,
        records: List[CurationRecord],
        output_dir: Union[Path, str],
    ) -> ArtifactPackage:
        """Export holdout_eval bucket: full export + frozen prompt-target pairings."""
        package = self.create_artifact_package(
            records,
            PromotionBucket.HOLDOUT_EVAL,
            output_dir,
            artifact_type=ARTIFACT_TYPE_HOLDOUT_BUNDLE,
            retention_class=RETENTION_RELEASE_CANDIDATE,
        )

        # Freeze prompt-target pairings
        out = Path(output_dir)
        self._freeze_evaluation_pairings(records, out)
        return package

    def _freeze_evaluation_pairings(
        self,
        records: List[CurationRecord],
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Freeze deterministic prompt-target pairings for holdout evaluation."""
        import random

        pairings = []
        for record in records:
            if (
                record.status == RecordStatus.PROMOTED
                and record.promotion_bucket == PromotionBucket.HOLDOUT_EVAL
            ):
                rng = random.Random(record.audio_hash)
                candidates = [
                    r
                    for r in records
                    if r.speaker_cluster == record.speaker_cluster
                    and r.conversation_id != record.conversation_id
                    and r.duration_sec >= 3.0
                ]
                if candidates:
                    prompt_rec = rng.choice(candidates)
                    pairings.append(
                        {
                            "target_record_id": record.record_id,
                            "prompt_record_id": prompt_rec.record_id,
                            "speaker_cluster": record.speaker_cluster,
                            "prompt_duration": prompt_rec.duration_sec,
                        }
                    )

        with open(output_dir / "frozen_evaluation_pairings.json", "w", encoding="utf-8") as f:
            json.dump(pairings, f, indent=2)
        return pairings

    def export_all_buckets(
        self,
        records: List[CurationRecord],
        output_dir: Optional[Union[Path, str]] = None,
    ) -> Dict[str, Any]:
        """Export all promotion buckets."""
        base = Path(output_dir) if output_dir else self.config.output_dir
        results: Dict[str, Any] = {}
        for bucket in PromotionBucket:
            if bucket == PromotionBucket.NONE:
                continue
            bucket_dir = base / bucket.value
            summary = self.export_subset(records, bucket, bucket_dir)
            results[bucket.value] = summary
        return results

    def export_evaluation_package(
        self,
        records: List[CurationRecord],
        output_dir: Union[Path, str],
    ) -> Dict[str, Any]:
        """Export holdout evaluation subset as a reproducible package with frozen prompt pairings."""
        summary = self.export_subset(
            records, PromotionBucket.HOLDOUT_EVAL, output_dir
        )
        
        # Freeze prompt-target evaluation pairings (Worker 06)
        out = Path(output_dir)
        pairings = []
        import random
        
        for record in records:
            if record.status == RecordStatus.PROMOTED and record.promotion_bucket == PromotionBucket.HOLDOUT_EVAL:
                # Deterministic pseudo-random pairing based on hash
                rng = random.Random(record.audio_hash)
                # Find other records from the same speaker but different conversation
                candidates = [
                    r for r in records
                    if r.speaker_cluster == record.speaker_cluster 
                    and r.conversation_id != record.conversation_id
                    and r.duration_sec >= 3.0
                ]
                
                if candidates:
                    prompt_rec = rng.choice(candidates)
                    pairings.append({
                        "target_record_id": record.record_id,
                        "prompt_record_id": prompt_rec.record_id,
                        "speaker_cluster": record.speaker_cluster,
                        "prompt_duration": prompt_rec.duration_sec
                    })
                    
        with open(out / "frozen_evaluation_pairings.json", "w", encoding="utf-8") as f:
            json.dump(pairings, f, indent=2)
            
        summary["frozen_pairings_count"] = len(pairings)
        return summary

    def export_user_voice_adaptor(
        self,
        profile_id: str,
        base_model_id: str,
        lora_weights_path: Union[Path, str],
        prompt_codec_tokens: np.ndarray,
        speaker_embed: np.ndarray,
        output_dir: Union[Path, str],
        prompt_summary_tokens: Optional[np.ndarray] = None,
        merge_to_onnx: bool = True,
    ) -> Dict[str, Any]:
        """Export a few-shot trained voice identity (LoRA) to a production-ready artifact.
        
        Args:
            profile_id: Unique identifier for the exported profile.
            base_model_id: Checksum/ID of the base model used for training.
            lora_weights_path: Path to the trained LoRA state_dict.
            prompt_codec_tokens: Original 3-10s prompt codec tokens.
            speaker_embed: Global speaker embedding.
            output_dir: Destination directory for the package.
            prompt_summary_tokens: Compressed tokens from PromptResampler.
            merge_to_onnx: If True, merges LoRA weights into a new ONNX model file.
        """
        out = Path(output_dir) / profile_id
        out.mkdir(parents=True, exist_ok=True)
        
        # 1. Save canonical metadata (SpeakerProfile)
        profile_meta = {
            "speaker_profile_id": profile_id,
            "base_model_id": base_model_id,
            "adaptor_id": f"{profile_id}_lora",
            "adaptor_merged": merge_to_onnx,
        }
        with open(out / "profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_meta, f, ensure_ascii=False, indent=2)
            
        # 2. Save prompt artifacts
        np.save(out / "prompt_codec_tokens.npy", prompt_codec_tokens)
        np.save(out / "speaker_embed.npy", speaker_embed)
        if prompt_summary_tokens is not None:
            np.save(out / "prompt_summary_tokens.npy", prompt_summary_tokens)
        
        # 3. Merge weights for VST/Standalone (ONNX constraint)
        if merge_to_onnx:
            merged_model_path = out / f"{profile_id}_merged.onnx"
            logger.info("Merging LoRA weights into ONNX base model: %s", merged_model_path)
            # Stub: Real implementation calls ONNX graph surgery / PyTorch export
            self._merge_lora_to_onnx(base_model_id, lora_weights_path, merged_model_path)
            profile_meta["merged_onnx_path"] = str(merged_model_path)
            
        logger.info("Exported User Voice Adaptor '%s' to %s", profile_id, out)
        return profile_meta
        
    def _merge_lora_to_onnx(self, base_model_id: str, lora_path: Union[Path, str], out_path: Path) -> None:
        """Merge LoRA weights into a base model and export to a single ONNX file."""
        import torch
        from tmrvc_train.models import DisentangledUCLM
        
        logger.info("Starting ONNX export with merged LoRA: %s", out_path)
        
        # 1. Initialize model from base checkpoint
        # (Assuming model architecture is stable across adaptations)
        model = DisentangledUCLM() 
        # Load base weights
        base_ckpt_path = Path("checkpoints/uclm") / f"{base_model_id}.pt"
        if not base_ckpt_path.exists():
            base_ckpt_path = Path("checkpoints/uclm/uclm_latest.pt")
            
        model.load_state_dict(torch.load(base_ckpt_path, map_location="cpu")["model"])
        
        # 2. Load and Apply LoRA state_dict
        lora_sd = torch.load(lora_path, map_location="cpu")
        model.load_state_dict(lora_sd, strict=False) # Only lora params
        
        # 3. Fold LoRA into base weights (optional but recommended for speed)
        # Some LoRA implementations support .merge() or manually adding deltas to weights.
        # This ensures O(1) inference overhead in VST/Standalone.
        if hasattr(model, "merge_lora"):
            model.merge_lora()
            
        # 4. Export to ONNX
        model.eval()
        dummy_input = model.get_dummy_input() # Helper to get tracing inputs
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=17,
            input_names=["phoneme_ids", "language_ids", "pointer_state", "speaker_embed"],
            output_names=["logits_a", "logits_b", "advance_logit", "progress_delta"],
            dynamic_axes={
                "phoneme_ids": {1: "L"},
                "pointer_state": {0: "B"}
            }
        )
        logger.info("Successfully exported merged ONNX model to %s", out_path)
