"""Stage 0: Ingest - Identify and register raw audio assets."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

import soundfile as sf

from ..models import CurationRecord, RecordStatus, Provenance
from ..orchestrator import CurationOrchestrator

logger = logging.getLogger(__name__)


def calculate_hash(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def ingest_directory(
    orchestrator: CurationOrchestrator,
    input_dir: Path | str,
    extension: str = ".wav",
    recursive: bool = True
) -> int:
    """Scan directory for audio files and add new ones to the manifest."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")

    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    files = list(input_path.glob(pattern))
    logger.info("Found %d files with extension %s in %s", len(files), extension, input_path)

    new_count = 0
    for file_path in files:
        # Generate a stable record_id based on relative path
        rel_path = file_path.relative_to(input_path)
        record_id = hashlib.md5(str(rel_path).encode()).hexdigest()[:16]
        
        # Skip if already in manifest
        if record_id in orchestrator.records:
            continue

        try:
            info = sf.info(file_path)
            audio_hash = calculate_hash(file_path)
            
            record = CurationRecord(
                record_id=record_id,
                source_path=str(file_path.absolute()),
                audio_hash=audio_hash,
                segment_start_sec=0.0,
                segment_end_sec=info.duration,
                duration_sec=info.duration,
                status=RecordStatus.INGESTED,
                attributes={
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype
                }
            )
            
            # Initial provenance for Ingest
            record.providers["ingest"] = Provenance(
                stage="ingest",
                provider="local_scanner",
                version="1.0.0",
                timestamp=Path(file_path).stat().st_mtime,
                metadata={"source_dir": str(input_path.absolute())}
            )
            
            orchestrator.update_record(record)
            new_count += 1
            
            if new_count % 100 == 0:
                orchestrator.save_manifest()
                logger.info("Ingested %d new records...", new_count)
                
        except Exception as e:
            logger.error("Failed to ingest %s: %s", file_path, e)

    orchestrator.save_manifest()
    logger.info("Ingest complete. Added %d new records. Total: %d", new_count, len(orchestrator.records))
    return new_count
