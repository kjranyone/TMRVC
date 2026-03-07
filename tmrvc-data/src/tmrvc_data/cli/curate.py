"""``tmrvc-curate`` -- AI Curation System CLI.

Provides commands for ingesting, annotating, scoring, promoting,
exporting, and validating curated audio datasets.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.stages.ingest import ingest_directory

    orch = CurationOrchestrator(args.output_dir)
    count = ingest_directory(
        orch, args.input_dir,
        extension=args.extension,
        recursive=not args.no_recursive,
    )
    print(f"Ingested {count} new records. Total: {len(orch.records)}")


def cmd_score(args: argparse.Namespace) -> None:
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.scoring import QualityScoringEngine

    orch = CurationOrchestrator(args.output_dir)
    engine = QualityScoringEngine()

    for rid, record in orch.records.items():
        orch.records[rid] = engine.score_and_decide(record)

    orch.save_manifest()
    report = engine.generate_report(list(orch.records.values()))
    print(json.dumps(report, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.export import CurationExporter

    orch = CurationOrchestrator(args.output_dir)
    exporter = CurationExporter()
    results = exporter.export_all_buckets(
        list(orch.records.values()),
        output_dir=args.export_dir,
    )
    print(json.dumps(results, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    from tmrvc_data.curation.orchestrator import CurationOrchestrator
    from tmrvc_data.curation.validation import CurationValidator

    orch = CurationOrchestrator(args.output_dir)
    validator = CurationValidator()
    report = validator.run_all(list(orch.records.values()))

    if args.report_path:
        validator.save_report(report, args.report_path)

    print(json.dumps(report, indent=2))

    if not report["overall"]["pass"]:
        sys.exit(1)


def cmd_summary(args: argparse.Namespace) -> None:
    from tmrvc_data.curation.orchestrator import CurationOrchestrator

    orch = CurationOrchestrator(args.output_dir)
    summary_path = orch.summary_path
    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print(f"No summary found at {summary_path}")


def cmd_run_stage(args: argparse.Namespace) -> None:
    from tmrvc_data.curation.orchestrator import CurationOrchestrator

    orch = CurationOrchestrator(args.output_dir)

    stage = args.stage
    if stage == "ingest":
        from tmrvc_data.curation.stages.ingest import ingest_directory
        if not args.input_dir:
            print("Error: --input-dir required for ingest stage", file=sys.stderr)
            sys.exit(1)
        ingest_directory(orch, args.input_dir)
    elif stage == "score":
        from tmrvc_data.curation.scoring import QualityScoringEngine
        engine = QualityScoringEngine()
        orch.run_stage("scoring", lambda r: engine.score_and_decide(r), force=args.force)
    else:
        print(f"Stage '{stage}' is not yet implemented. "
              f"Available: ingest, score", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-curate",
        description="AI Curation System for TMRVC datasets.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/curation"),
        help="Curation working directory (manifest, summary, etc.)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Scan and register raw audio files")
    p_ingest.add_argument("--input-dir", type=Path, required=True)
    p_ingest.add_argument("--extension", default=".wav")
    p_ingest.add_argument("--no-recursive", action="store_true")

    # run-stage
    p_stage = sub.add_parser("run-stage", help="Run a specific curation stage")
    p_stage.add_argument("stage", choices=["ingest", "score"])
    p_stage.add_argument("--input-dir", type=Path, default=None)
    p_stage.add_argument("--force", action="store_true")

    # score
    sub.add_parser("score", help="Score all records and assign decisions")

    # export
    p_export = sub.add_parser("export", help="Export promoted subsets to cache")
    p_export.add_argument("--export-dir", type=Path, default=Path("data/curated_export"))

    # validate
    p_validate = sub.add_parser("validate", help="Run validation checks")
    p_validate.add_argument("--report-path", type=Path, default=None)

    # summary
    sub.add_parser("summary", help="Show curation summary")

    # resume (alias for run-stage with --force)
    p_resume = sub.add_parser("resume", help="Resume a previously interrupted stage")
    p_resume.add_argument("stage", choices=["ingest", "score"])
    p_resume.add_argument("--input-dir", type=Path, default=None)

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
        "score": cmd_score,
        "export": cmd_export,
        "validate": cmd_validate,
        "summary": cmd_summary,
        "resume": lambda a: cmd_run_stage(argparse.Namespace(
            output_dir=a.output_dir, stage=a.stage, input_dir=a.input_dir,
            force=True, verbose=a.verbose,
        )),
    }

    handler = cmd_map.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
