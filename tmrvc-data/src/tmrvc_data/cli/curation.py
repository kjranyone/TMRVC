"""``tmrvc-curation`` -- Worker 07 CLI commands.

Provides stage-addressable curation pipeline commands:
  curation-ingest   — ingest raw audio files into SQLite
  curation-run-stage — run a specific stage on pending records
  curation-resume   — resume interrupted processing
  curation-status   — show pipeline status summary
  curation-promote  — apply promotion rules to scored records
  curation-export   — export promoted records to cache format
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Command implementations
# ------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest raw audio files into the curation database."""
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.service import CurationDataService
    from tmrvc_data.curation.stages.ingest import ingest_directory

    orch = CurationOrchestrator(args.output_dir)
    count = ingest_directory(
        orch,
        args.path,
        extension=args.extension,
        recursive=not args.no_recursive,
    )
    print(f"Ingested {count} new records. Total: {len(orch.records)}")

    # Mirror to SQLite if db_path provided
    if args.db:
        svc = CurationDataService(args.db)
        inserted = svc.batch_create(list(orch.records.values()))
        print(f"Synced {inserted} records to SQLite at {args.db}")


def cmd_run_stage(args: argparse.Namespace) -> None:
    """Run a specific stage on pending records."""
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.stage_framework import STAGE_NAMES

    stage_num = args.stage_num

    if stage_num not in STAGE_NAMES:
        print(
            f"Error: stage number must be 0-9. Got {stage_num}",
            file=sys.stderr,
        )
        sys.exit(1)

    orch = CurationOrchestrator(args.output_dir)

    # For stages with built-in processors, use the named-stage API
    stage_name = STAGE_NAMES[stage_num]

    if stage_name == "ingest":
        print(
            "Use 'curation-ingest' for the ingest stage.",
            file=sys.stderr,
        )
        sys.exit(1)

    if stage_name == "score":
        # Use the scoring stage alias
        from tmrvc_data.curation.scoring import QualityScoringEngine

        engine = QualityScoringEngine()
        orch.run_stage(
            "scoring",
            lambda r: engine.score_and_decide(r),
            force=args.force,
        )
    else:
        try:
            orch.run_named_stage(stage_name, force=args.force)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"Stage {stage_num} ({stage_name}) complete.")


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume interrupted processing from the last checkpoint."""
    from tmrvc_data.curation.orchestrator import CurationOrchestrator

    orch = CurationOrchestrator(args.output_dir)

    # Find stages with unprocessed records and run them
    from tmrvc_data.curation.orchestrator import _STAGE_ORDER

    for stage_name in _STAGE_ORDER:
        if stage_name == "ingest":
            continue
        has_pending = any(
            not any(p.stage == stage_name for p in r.providers.values())
            for r in orch.records.values()
        )
        if has_pending:
            print(f"Resuming stage: {stage_name}")
            try:
                orch.run_named_stage(stage_name, force=False)
            except ValueError:
                print(f"  (skipped: no implementation for {stage_name})")

    print("Resume complete.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline status summary."""
    if args.db:
        from tmrvc_data.curation.service import CurationDataService

        svc = CurationDataService(args.db)
        summary = svc.status_summary()
    else:
        from tmrvc_data.curation.orchestrator import CurationOrchestrator

        orch = CurationOrchestrator(args.output_dir)
        status_counts: dict[str, int] = {}
        bucket_counts: dict[str, int] = {}
        for r in orch.records.values():
            status_counts[r.status.value] = (
                status_counts.get(r.status.value, 0) + 1
            )
            bucket_counts[r.promotion_bucket.value] = (
                bucket_counts.get(r.promotion_bucket.value, 0) + 1
            )
        summary = {
            "total_records": len(orch.records),
            "status": status_counts,
            "promotion_buckets": bucket_counts,
        }

    print(json.dumps(summary, indent=2))


def cmd_promote(args: argparse.Namespace) -> None:
    """Apply promotion rules to scored records."""
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.scoring import QualityScoringEngine

    orch = CurationOrchestrator(args.output_dir)
    engine = QualityScoringEngine()

    promoted = 0
    reviewed = 0
    rejected = 0

    for rid, record in list(orch.records.items()):
        if record.status not in (
            RecordStatus.SCORED,
            RecordStatus.REVIEW,
        ):
            continue

        result = engine.score_and_decide(record)
        orch.records[rid] = result

        if result.status == RecordStatus.PROMOTED:
            promoted += 1
        elif result.status == RecordStatus.REVIEW:
            reviewed += 1
        elif result.status == RecordStatus.REJECTED:
            rejected += 1

    orch.save_manifest()
    print(
        f"Promotion complete: {promoted} promoted, "
        f"{reviewed} review, {rejected} rejected"
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export promoted records to cache format."""
    from tmrvc_data.curation.export import CurationExporter
    from tmrvc_data.curation.orchestrator import CurationOrchestrator

    orch = CurationOrchestrator(args.output_dir)
    exporter = CurationExporter()
    results = exporter.export_all_buckets(
        list(orch.records.values()),
        output_dir=args.export_dir,
    )
    print(json.dumps(results, indent=2))


# ------------------------------------------------------------------
# Imports needed by cmd_promote
# ------------------------------------------------------------------
from tmrvc_data.curation.models import RecordStatus


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-curation",
        description="Worker 07: Curation orchestration CLI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/curation"),
        help="Curation working directory.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite database (optional).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # curation-ingest
    p_ingest = sub.add_parser("ingest", help="Ingest raw audio files")
    p_ingest.add_argument("path", type=Path, help="Directory of audio files")
    p_ingest.add_argument("--extension", default=".wav")
    p_ingest.add_argument("--no-recursive", action="store_true")

    # curation-run-stage
    p_stage = sub.add_parser("run-stage", help="Run a specific stage")
    p_stage.add_argument("stage_num", type=int, help="Stage number (0-9)")
    p_stage.add_argument("--force", action="store_true")

    # curation-resume
    sub.add_parser("resume", help="Resume interrupted processing")

    # curation-status
    sub.add_parser("status", help="Show pipeline status summary")

    # curation-promote
    sub.add_parser("promote", help="Apply promotion rules to scored records")

    # curation-export
    p_export = sub.add_parser("export", help="Export promoted records")
    p_export.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/curated_export"),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cmd_map = {
        "ingest": cmd_ingest,
        "run-stage": cmd_run_stage,
        "resume": cmd_resume,
        "status": cmd_status,
        "promote": cmd_promote,
        "export": cmd_export,
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
