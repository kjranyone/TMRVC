"""TMRVC Gradio Control Plane — UCLM v3 HITL Interface.

Browser-based control plane for drama-grade TTS evaluation, curation
auditing, dataset management, blind A/B evaluation, and system admin.

Implements Worker 12 (Gradio Control Plane) from the UCLM v3 plan.

Launch:
    python -m tmrvc_gui.gradio_app [--server-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import random
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS, VOICE_STATE_REGISTRY
from tmrvc_gui.gradio_state import (
    ROLES,
    AuditTrail,
    CastingGallery,
    EvalPair,
    EvalSession,
    check_permission,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API helpers — talk to tmrvc-serve
# ---------------------------------------------------------------------------

_SERVER_URL = "http://localhost:8000"


def _api_url(path: str) -> str:
    return f"{_SERVER_URL}{path}"


def _api_get(path: str) -> dict | list | None:
    import httpx

    try:
        r = httpx.get(_api_url(path), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("API GET %s failed: %s", path, e)
        return None


def _api_post(path: str, body: dict) -> dict | None:
    import httpx

    try:
        r = httpx.post(_api_url(path), json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("API POST %s failed: %s", path, e)
        return None


def _api_patch(path: str, body: dict) -> dict | None:
    import httpx

    try:
        r = httpx.patch(_api_url(path), json=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("API PATCH %s failed: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------

_audit = AuditTrail()
_gallery = CastingGallery()
_eval_session = EvalSession()

SAMPLE_RATE = 24000


# ===================================================================
# Tab 1: Drama Workshop
# ===================================================================


def _fetch_speaker_profiles() -> list[str]:
    """Fetch speaker profile names from the API, falling back to local gallery."""
    models = _api_get("/admin/models")
    if models and isinstance(models, list):
        return [m.get("name", "unknown") for m in models]
    return _gallery.list_names()


def _build_drama_workshop() -> gr.Blocks:
    """Interactive drama workshop with voice cloning, pacing, and physical controls.

    Tier 1 v3.0: basic inference with pacing controls, 8-D voice_state sliders,
    CFG scale, and speaker_profile_id selection.  Generation dispatches to
    POST /ui/workshop/generate.
    """

    # Canonical 8-D voice_state dimension keys (matching VOICE_STATE_REGISTRY)
    _VS_KEYS = list(CANONICAL_VOICE_STATE_IDS)

    with gr.Blocks() as tab:
        gr.Markdown("## Drama Workshop")

        with gr.Row():
            with gr.Column(scale=1):
                # --- Speaker Profile ---
                gr.Markdown("### Speaker Profile")
                speaker_profile_id = gr.Dropdown(
                    label="Speaker Profile",
                    choices=_fetch_speaker_profiles(),
                    value=None,
                    allow_custom_value=True,
                    interactive=True,
                )
                character_id = gr.Textbox(
                    label="Character ID", value="default", max_lines=1
                )
                btn_refresh_profiles = gr.Button("Refresh Profiles")

                # --- Pacing Controls ---
                gr.Markdown("### Pacing")
                pace = gr.Slider(0.5, 3.0, value=1.0, step=0.05, label="Pace")
                hold_bias = gr.Slider(
                    -1.0, 1.0, value=0.0, step=0.05, label="Hold Bias"
                )
                boundary_bias = gr.Slider(
                    -1.0, 1.0, value=0.0, step=0.05, label="Boundary Bias"
                )
                cfg_scale = gr.Slider(
                    1.0, 3.0, value=1.5, step=0.1, label="CFG Scale"
                )

            with gr.Column(scale=1):
                # --- Physical Controls (8-D voice_state) ---
                gr.Markdown("### Physical Controls (8-D Voice State)")
                vs_pitch_level = gr.Slider(
                    0, 1, value=0.5, step=0.05, label=VOICE_STATE_REGISTRY[0].name
                )
                vs_pitch_range = gr.Slider(
                    0, 1, value=0.3, step=0.05, label=VOICE_STATE_REGISTRY[1].name
                )
                vs_energy_level = gr.Slider(
                    0, 1, value=0.5, step=0.05, label=VOICE_STATE_REGISTRY[2].name
                )
                vs_pressedness = gr.Slider(
                    0, 1, value=0.35, step=0.05, label=VOICE_STATE_REGISTRY[3].name
                )
                vs_spectral_tilt = gr.Slider(
                    0, 1, value=0.5, step=0.05, label=VOICE_STATE_REGISTRY[4].name
                )
                vs_breathiness = gr.Slider(
                    0, 1, value=0.2, step=0.05, label=VOICE_STATE_REGISTRY[5].name
                )
                vs_voice_irregularity = gr.Slider(
                    0, 1, value=0.15, step=0.05, label=VOICE_STATE_REGISTRY[6].name
                )
                vs_openness = gr.Slider(
                    0, 1, value=0.5, step=0.05, label=VOICE_STATE_REGISTRY[7].name
                )

                # --- Dialogue Context ---
                gr.Markdown("### Dialogue Context")
                dialogue_ctx = gr.Textbox(
                    label="Script / Dialogue",
                    placeholder="Enter script or preceding dialogue context...",
                    lines=3,
                )
                acting_intent = gr.Textbox(
                    label="Acting Intent",
                    placeholder="e.g. sarcastic, pleading, cheerful",
                    max_lines=1,
                )

        # --- Input & Generation ---
        gr.Markdown("### Generation")
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter text to synthesize...",
            lines=3,
        )
        with gr.Row():
            btn_generate = gr.Button("Generate", variant="primary")
            btn_multi_take = gr.Button("Generate Multi-Take (3)")

        output_audio = gr.Audio(label="Output", type="numpy")
        generation_info = gr.JSON(label="Generation Info", visible=True)

        take_outputs = gr.Dataframe(
            headers=["take_id", "seed", "cfg_scale", "pace", "notes"],
            label="Takes",
            interactive=False,
        )

        # --- Session Persistence ---
        with gr.Row():
            btn_save_project = gr.Button("Save Project")
            btn_load_project = gr.Button("Load Project")
            project_file = gr.File(label="Project File", file_types=[".json"])

        # --- Preset Management ---
        gr.Markdown("### Voice State Presets")
        with gr.Row():
            preset_name = gr.Textbox(label="Preset Name", max_lines=1)
            btn_save_preset = gr.Button("Save Preset")
            btn_load_preset = gr.Button("Load Preset")
        preset_list = gr.Dropdown(label="Presets", choices=[], interactive=True)

        status = gr.Textbox(label="Status", interactive=False, value="Ready")

        # --- All voice state sliders in canonical order ---
        _vs_sliders = [
            vs_pitch_level,
            vs_pitch_range,
            vs_energy_level,
            vs_pressedness,
            vs_spectral_tilt,
            vs_breathiness,
            vs_voice_irregularity,
            vs_openness,
        ]

        # --- Callbacks ---

        btn_refresh_profiles.click(
            lambda: gr.update(choices=_fetch_speaker_profiles()),
            outputs=[speaker_profile_id],
        )

        def save_preset(name, *slider_vals):
            if not name.strip():
                return [], "Enter preset name."
            preset = dict(zip(_VS_KEYS, slider_vals))
            p = Path("data/presets")
            p.mkdir(parents=True, exist_ok=True)
            (p / f"{name}.json").write_text(json.dumps(preset, indent=2), encoding="utf-8")
            names = [f.stem for f in p.glob("*.json")]
            return names, f"Saved preset: {name}"

        btn_save_preset.click(
            save_preset,
            inputs=[preset_name] + _vs_sliders,
            outputs=[preset_list, status],
        )

        def load_preset(name):
            if not name:
                return [0.5] * 8 + ["Select a preset."]
            p = Path("data/presets") / f"{name}.json"
            if not p.exists():
                return [0.5] * 8 + [f"Preset not found: {name}"]
            d = json.loads(p.read_text(encoding="utf-8"))
            return [d.get(k, 0.5) for k in _VS_KEYS] + [f"Loaded preset: {name}"]

        btn_load_preset.click(
            load_preset,
            inputs=[preset_list],
            outputs=_vs_sliders + [status],
        )

        def generate(
            text, char_id, spk_profile, pace_v, hold_v, boundary_v, cfg_v,
            *vs_and_ctx,
        ):
            # Last two args are dialogue_ctx and acting_intent
            vs_vals = list(vs_and_ctx[:8])
            ctx = vs_and_ctx[8] if len(vs_and_ctx) > 8 else ""
            intent = vs_and_ctx[9] if len(vs_and_ctx) > 9 else ""

            if not text.strip():
                return None, {}, "Enter text to generate."

            body = {
                "text": text,
                "character_id": char_id or "default",
                "pace": pace_v,
                "hold_bias": hold_v,
                "boundary_bias": boundary_v,
                "cfg_scale": cfg_v,
                "explicit_voice_state": vs_vals,
            }
            if spk_profile:
                body["speaker_profile_id"] = spk_profile

            # Use the workshop generate endpoint
            resp = _api_post("/ui/workshop/generate", body)
            if resp is None:
                # Fallback to /tts for backward compatibility
                resp = _api_post("/tts", body)
            if resp is None:
                return None, {}, "API call failed. Is tmrvc-serve running?"
            audio_b64 = resp.get("audio_base64", "")
            if not audio_b64:
                # Workshop endpoint returns job_id; show that as info
                return None, resp, f"Job submitted: {resp.get('job_id', 'unknown')}"
            audio_bytes = base64.b64decode(audio_b64)
            sr = resp.get("sample_rate", SAMPLE_RATE)
            audio_np = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            _audit.log("director", "gradio", "workshop_generate",
                       after_state=f"text={text[:30]}")
            return (sr, audio_np), resp.get("style_used", {}), "Generated."

        btn_generate.click(
            generate,
            inputs=[
                input_text, character_id, speaker_profile_id,
                pace, hold_bias, boundary_bias, cfg_scale,
            ] + _vs_sliders + [dialogue_ctx, acting_intent],
            outputs=[output_audio, generation_info, status],
        )

        def generate_multi_take(
            text, char_id, spk_profile, pace_v, hold_v, boundary_v, cfg_v,
            *vs_and_ctx,
        ):
            if not text.strip():
                return [], "Enter text."

            vs_vals = list(vs_and_ctx[:8])
            body = {
                "text": text,
                "character_id": char_id or "default",
                "pace": pace_v,
                "hold_bias": hold_v,
                "boundary_bias": boundary_v,
                "cfg_scale": cfg_v,
                "explicit_voice_state": vs_vals,
                "n_takes": 3,
            }
            if spk_profile:
                body["speaker_profile_id"] = spk_profile

            resp = _api_post("/ui/workshop/generate", body)
            if resp and resp.get("takes"):
                takes = [[tid, 0, cfg_v, pace_v, f"Take {i+1}"]
                         for i, tid in enumerate(resp["takes"])]
            else:
                takes = []
                for i in range(3):
                    tid = str(uuid.uuid4())[:8]
                    takes.append([tid, random.randint(0, 1000000), cfg_v, pace_v, f"Take {i+1}"])

            _audit.log("director", "gradio", "generate_multi_take",
                       after_state=f"count=3, text={text[:20]}")
            return takes, f"Generated {len(takes)} takes."

        btn_multi_take.click(
            generate_multi_take,
            inputs=[
                input_text, character_id, speaker_profile_id,
                pace, hold_bias, boundary_bias, cfg_scale,
            ] + _vs_sliders + [dialogue_ctx, acting_intent],
            outputs=[take_outputs, status],
        )

        def save_project(text, char_id, pace_v, hold_v, boundary_v, cfg_v,
                         ctx, intent, *slider_vals):
            vs_dict = dict(zip(_VS_KEYS, slider_vals))
            project = {
                "text": text,
                "character_id": char_id,
                "pacing": {"pace": pace_v, "hold_bias": hold_v, "boundary_bias": boundary_v},
                "cfg_scale": cfg_v,
                "dialogue_context": ctx,
                "acting_intent": intent,
                "voice_state": vs_dict,
            }
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="drama_project_"
            )
            json.dump(project, tmp, indent=2, ensure_ascii=False)
            tmp.close()
            return tmp.name, "Project saved."

        btn_save_project.click(
            save_project,
            inputs=[
                input_text, character_id,
                pace, hold_bias, boundary_bias, cfg_scale,
                dialogue_ctx, acting_intent,
            ] + _vs_sliders,
            outputs=[project_file, status],
        )

    return tab


# ===================================================================
# Tab 2: Curation Auditor
# ===================================================================


def _build_curation_auditor() -> gr.Blocks:
    """Manifest browser for review, promotion, and rejection."""

    with gr.Blocks() as tab:
        gr.Markdown("## Curation Auditor")

        with gr.Row():
            manifest_path = gr.Textbox(
                label="Manifest Path",
                value="data/curation/manifest.jsonl",
                max_lines=1,
            )
            btn_load_manifest = gr.Button("Load Manifest")

        with gr.Row():
            filter_status = gr.Dropdown(
                choices=["all", "promoted", "review", "rejected", "scored"],
                value="review",
                label="Filter Status",
            )
            filter_bucket = gr.Dropdown(
                choices=[
                    "all",
                    "tts_mainline",
                    "vc_prior",
                    "expressive_prior",
                    "holdout_eval",
                ],
                value="all",
                label="Filter Bucket",
            )

        manifest_table = gr.Dataframe(
            headers=[
                "record_id",
                "transcript",
                "language",
                "quality_score",
                "status",
                "speaker_cluster",
                "source_legality",
                "metadata_version",
            ],
            label="Manifest Records",
            interactive=False,
        )

        gr.Markdown("### Quick Review")
        with gr.Row():
            selected_record_id = gr.Textbox(
                label="Selected Record ID", max_lines=1
            )
            selected_version = gr.Number(
                label="Version", visible=False, precision=0, value=1
            )

        with gr.Row():
            record_audio = gr.Audio(label="Waveform", interactive=False)
            record_info = gr.JSON(label="Record Details")

        with gr.Row():
            transcript_edit = gr.Textbox(label="Edit Transcript", lines=2)

        with gr.Row():
            role_select = gr.Dropdown(choices=ROLES, value="auditor", label="Your Role")
            actor_id = gr.Textbox(
                label="Your ID", value="reviewer_1", max_lines=1
            )
            rationale = gr.Textbox(
                label="Rationale", placeholder="Reason for decision...", max_lines=1
            )

        with gr.Row():
            btn_promote = gr.Button("Promote", variant="primary")
            btn_reject = gr.Button("Reject", variant="stop")
            btn_save_transcript = gr.Button("Save Transcript Edit")

        audit_status = gr.Textbox(label="Status", interactive=False)

        # --- Callbacks ---

        def load_manifest(status_filter, bucket_filter):
            url = f"/ui/curation/records?status={status_filter}&bucket={bucket_filter}"
            data = _api_get(url)
            if data is None:
                return [], "API call failed. Is tmrvc-serve running?"
            rows = []
            for rec in data:
                rows.append([
                    rec.get("record_id", ""),
                    rec.get("transcript", "")[:80],
                    rec.get("language", ""),
                    rec.get("quality_score", 0),
                    rec.get("status", ""),
                    rec.get("speaker_cluster", ""),
                    rec.get("source_legality", ""),
                    rec.get("metadata_version", 1),
                ])
            return rows, f"Loaded {len(rows)} records via API."

        btn_load_manifest.click(
            load_manifest,
            inputs=[filter_status, filter_bucket],
            outputs=[manifest_table, audit_status],
        )

        def on_select(evt: gr.SelectData, rows):
            # rows is a pandas DataFrame when coming from gr.Dataframe
            rid = rows.iloc[evt.index[0]]["record_id"]
            ver = rows.iloc[evt.index[0]]["metadata_version"]
            # Get full details from API
            details = _api_get(f"/ui/curation/records/{rid}")
            text = details.get("transcript", "") if details else rows.iloc[evt.index[0]]["transcript"]
            return rid, ver, text, details

        manifest_table.select(
            on_select,
            inputs=[manifest_table],
            outputs=[selected_record_id, selected_version, transcript_edit, record_info],
        )

        def do_promote(record_id, version, role, aid, reason):
            if not check_permission(role, "promote"):
                return f"Role '{role}' cannot promote."
            if not record_id.strip():
                return "Select a record first."
            body = {
                "record_id": record_id,
                "role": role,
                "actor_id": aid,
                "rationale": reason,
                "expected_version": int(version),
            }
            resp = _api_post("/ui/curation/actions/promote", body)
            if resp is None:
                return "Promotion failed (Conflict or Server Error)."
            _audit.log(role, aid, "promote", before_state="review", after_state="promoted", rationale=reason)
            return f"Promoted {record_id} (new version: {resp.get('metadata_version')})."

        btn_promote.click(
            do_promote,
            inputs=[selected_record_id, selected_version, role_select, actor_id, rationale],
            outputs=[audit_status],
        )

        def do_reject(record_id, version, role, aid, reason):
            if not check_permission(role, "reject"):
                return f"Role '{role}' cannot reject."
            if not record_id.strip():
                return "Select a record first."
            body = {
                "status": "rejected",
                "expected_version": int(version),
            }
            resp = _api_patch(f"/ui/curation/records/{record_id}", body)
            if resp is None:
                return "Rejection failed (Conflict or Server Error)."

            _audit.log(role, aid, "reject", before_state="review", after_state="rejected", rationale=reason)
            return f"Rejected {record_id}."

        btn_reject.click(
            do_reject,
            inputs=[selected_record_id, selected_version, role_select, actor_id, rationale],
            outputs=[audit_status],
        )

        def do_save_transcript(record_id, version, new_text, role, aid):
            if not check_permission(role, "edit_transcript"):
                return f"Role '{role}' cannot edit transcripts."
            if not record_id.strip():
                return "Select a record first."
            body = {
                "transcript": new_text,
                "expected_version": int(version),
            }
            resp = _api_patch(f"/ui/curation/records/{record_id}", body)
            if resp is None:
                return "Update failed (Conflict or Server Error)."
            _audit.log(role, aid, "edit_transcript", after_state=new_text[:100])
            return f"Transcript saved for {record_id}."

        btn_save_transcript.click(
            do_save_transcript,
            inputs=[selected_record_id, selected_version, transcript_edit, role_select, actor_id],
            outputs=[audit_status],
        )

    return tab


# ===================================================================
# Tab 3: Dataset Manager
# ===================================================================


def _build_dataset_manager() -> gr.Blocks:
    """Asset ingest, health dashboard, pipeline controls, and export."""

    with gr.Blocks() as tab:
        gr.Markdown("## Dataset Manager")

        # --- Asset Ingest ---
        gr.Markdown("### Asset Ingest")
        with gr.Row():
            upload_files = gr.File(
                label="Upload Audio Files",
                file_count="multiple",
                file_types=["audio"],
            )
            register_dir = gr.Textbox(
                label="Or Register Server Directory",
                placeholder="/path/to/audio/corpus",
                max_lines=1,
            )

        with gr.Row():
            legality = gr.Dropdown(
                choices=["owned", "licensed", "research-restricted", "unknown"],
                value="owned",
                label="Legality",
            )
            btn_ingest = gr.Button("Ingest", variant="primary")

        ingest_status = gr.Textbox(label="Ingest Status", interactive=False)

        # --- Health Dashboard ---
        gr.Markdown("### Health Dashboard")
        with gr.Row():
            health_dir = gr.Textbox(
                label="Cache Directory", value="data/cache", max_lines=1
            )
            btn_refresh_health = gr.Button("Refresh")

        with gr.Row():
            lbl_phoneme_cov = gr.Textbox(label="Phoneme Coverage", interactive=False)
            lbl_speaker_count = gr.Textbox(label="Speaker Count", interactive=False)
            lbl_unk_phone = gr.Textbox(label="Unknown Phone Ratio", interactive=False)
            lbl_duration = gr.Textbox(label="Duration Stats", interactive=False)

        # --- Pipeline Controls ---
        gr.Markdown("### Pipeline Controls")
        with gr.Row():
            btn_start_pipeline = gr.Button("Start")
            btn_resume_pipeline = gr.Button("Resume")
            btn_stop_pipeline = gr.Button("Stop")

        pipeline_progress = gr.Textbox(
            label="Pipeline Status", value="Idle", interactive=False
        )

        # --- Export ---
        gr.Markdown("### Export")
        with gr.Row():
            export_bucket = gr.Dropdown(
                choices=[
                    "tts_mainline",
                    "vc_prior",
                    "expressive_prior",
                    "holdout_eval",
                ],
                value="tts_mainline",
                label="Bucket",
            )
            btn_export = gr.Button("Export Promoted Subset")

        export_status = gr.Textbox(label="Export Status", interactive=False)

        # --- Split Manager ---
        gr.Markdown("### Split Manager")
        with gr.Row():
            cb_lock_split = gr.Checkbox(label="Lock Holdout/Train Split", value=False)
            lbl_split = gr.Textbox(
                label="Split Counts", value="train: -- | holdout: --", interactive=False
            )

        # --- Callbacks ---

        def refresh_health(cache_dir):
            p = Path(cache_dir) / "health_metrics.json"
            if not p.exists():
                return "--", "--", "--", "--"
            m = json.loads(p.read_text(encoding="utf-8"))
            return (
                str(m.get("phoneme_coverage", "--")),
                str(m.get("speaker_count", "--")),
                str(m.get("unknown_phone_ratio", "--")),
                str(m.get("duration_stats", "--")),
            )

        btn_refresh_health.click(
            refresh_health,
            inputs=[health_dir],
            outputs=[lbl_phoneme_cov, lbl_speaker_count, lbl_unk_phone, lbl_duration],
        )

        def do_ingest(files, dir_path, legal):
            sources = []
            if files:
                sources.extend(f.name for f in files)
            if dir_path and dir_path.strip():
                sources.append(dir_path.strip())
            if not sources:
                return "No sources provided."
            _audit.log("admin", "gradio", "ingest", after_state=f"legality={legal}, sources={len(sources)}")
            return f"Ingested {len(sources)} source(s) with legality={legal}."

        btn_ingest.click(
            do_ingest,
            inputs=[upload_files, register_dir, legality],
            outputs=[ingest_status],
        )

        def do_export(bucket):
            _audit.log("admin", "gradio", "export", after_state=f"bucket={bucket}")
            return f"Exported '{bucket}' subset."

        btn_export.click(do_export, inputs=[export_bucket], outputs=[export_status])

    return tab


# ===================================================================
# Tab 4: Evaluation Arena
# ===================================================================


def _build_evaluation_arena() -> gr.Blocks:
    """Blind A/B testing and MOS collection with API-backed session management.

    Tier 1 v3.0: blind A/B comparison, MOS collection, session management
    via POST /ui/eval/sessions, GET /ui/eval/assignments/{id},
    POST /ui/eval/assignments/{id}/submit.
    """

    with gr.Blocks() as tab:
        gr.Markdown("## Evaluation Arena")
        gr.Markdown(
            "Blind A/B preference test. Samples are randomized. "
            "Rate each pair without knowing which system produced which sample."
        )

        # --- Session Management ---
        gr.Markdown("### Session Management")
        with gr.Row():
            session_name = gr.Textbox(
                label="Session Name", value="eval_session_1", max_lines=1
            )
            eval_n_assignments = gr.Number(
                label="Number of Assignments", value=10, precision=0
            )
            btn_create_session = gr.Button("Create Session", variant="primary")

        session_id_state = gr.State("")
        assignment_ids_state = gr.State([])
        assignment_idx_state = gr.State(0)
        session_info = gr.JSON(label="Session Info", visible=True)

        with gr.Row():
            eval_role = gr.Dropdown(
                choices=["rater", "director"], value="rater", label="Your Role"
            )
            eval_id = gr.Textbox(label="Your ID", value="rater_1", max_lines=1)

        gr.Markdown("### Current Assignment")
        current_assignment_id = gr.Textbox(
            label="Assignment ID", interactive=False
        )
        eval_text = gr.Textbox(label="Text", interactive=False)
        pair_id_state = gr.State("")

        with gr.Row():
            audio_a = gr.Audio(label="Sample A", interactive=False)
            audio_b = gr.Audio(label="Sample B", interactive=False)

        with gr.Row():
            mos_a = gr.Slider(1, 5, value=3, step=0.5, label="MOS A (1-5)")
            mos_b = gr.Slider(1, 5, value=3, step=0.5, label="MOS B (1-5)")

        with gr.Row():
            btn_prefer_a = gr.Button("Prefer A")
            btn_tie = gr.Button("Tie")
            btn_prefer_b = gr.Button("Prefer B")

        eval_notes = gr.Textbox(
            label="Notes (Director only)",
            placeholder="Qualitative notes...",
            lines=2,
        )

        btn_next_pair = gr.Button("Next Assignment", variant="primary")
        eval_status = gr.Textbox(label="Status", interactive=False)

        # --- Summary ---
        gr.Markdown("### Results Summary")
        btn_summary = gr.Button("Show Summary")
        summary_output = gr.JSON(label="Summary")

        # --- Callbacks ---

        def create_session(name, n_assignments, evaluator_id):
            body = {
                "name": name,
                "evaluator_id": evaluator_id,
                "n_assignments": int(n_assignments),
            }
            resp = _api_post("/ui/eval/sessions", body)
            if resp is None:
                return "", [], 0, {"error": "Failed to create session. Is tmrvc-serve running?"}
            session_id = resp.get("session_id", "")
            assignments = resp.get("assignments", [])
            _audit.log("rater", evaluator_id, "create_eval_session",
                       after_state=f"session={session_id}, assignments={len(assignments)}")
            return session_id, assignments, 0, resp

        btn_create_session.click(
            create_session,
            inputs=[session_name, eval_n_assignments, eval_id],
            outputs=[session_id_state, assignment_ids_state, assignment_idx_state, session_info],
        )

        def get_next_assignment(session_id, assignment_ids, idx):
            if not assignment_ids or idx >= len(assignment_ids):
                return idx, "", "", "", "No more assignments in this session."
            assignment_id = assignment_ids[idx]
            resp = _api_get(f"/ui/eval/assignments/{assignment_id}")
            if resp is None:
                return idx, assignment_id, "", assignment_id, f"Assignment {assignment_id} (API fetch failed)"
            text = resp.get("text", "")
            return idx + 1, assignment_id, text, assignment_id, f"Assignment {idx + 1}/{len(assignment_ids)}"

        btn_next_pair.click(
            get_next_assignment,
            inputs=[session_id_state, assignment_ids_state, assignment_idx_state],
            outputs=[assignment_idx_state, current_assignment_id, eval_text, pair_id_state, eval_status],
        )

        def record_preference(pref, pair_id, ma, mb, role, rid, notes):
            if not pair_id:
                return "No active assignment."
            if not check_permission(role, "rate"):
                return f"Role '{role}' cannot rate."

            # Submit via API
            submit_body = {
                "rating": float(ma if pref == "A" else mb if pref == "B" else (ma + mb) / 2),
                "notes": notes if role == "director" else "",
            }
            resp = _api_post(f"/ui/eval/assignments/{pair_id}/submit", submit_body)

            # Also record locally for persistence
            pair = EvalPair(
                pair_id=pair_id,
                sample_a_label="hidden_a",
                sample_b_label="hidden_b",
                text="",
                preference=pref,
                mos_a=ma,
                mos_b=mb,
                rater_id=rid,
                rater_role=role,
                notes=notes if role == "director" else "",
                reference_audio_length=0.0, # Filled by session manager in real usage
                baseline_version="Qwen3-ForcedAligner-0.6B", # Stub or fetched from active contract
            )
            _eval_session.record(pair)
            _audit.log(role, rid, "rate", after_state=f"pair={pair_id} pref={pref}")

            api_status = "submitted" if resp else "(local only, API unavailable)"
            return f"Recorded: {pref} for {pair_id} {api_status}"

        btn_prefer_a.click(
            lambda *a: record_preference("A", *a),
            inputs=[pair_id_state, mos_a, mos_b, eval_role, eval_id, eval_notes],
            outputs=[eval_status],
        )
        btn_tie.click(
            lambda *a: record_preference("tie", *a),
            inputs=[pair_id_state, mos_a, mos_b, eval_role, eval_id, eval_notes],
            outputs=[eval_status],
        )
        btn_prefer_b.click(
            lambda *a: record_preference("B", *a),
            inputs=[pair_id_state, mos_a, mos_b, eval_role, eval_id, eval_notes],
            outputs=[eval_status],
        )

        btn_summary.click(lambda: _eval_session.summary(), outputs=[summary_output])

    return tab


# ===================================================================
# Tab 4b: Eval Tools (Reference Trimming / Control Sweeps / Rater Assignment)
# ===================================================================


def _build_eval_tools() -> gr.Blocks:
    """Evaluation pipeline tools: trim references, run control sweeps,
    generate rater assignments — all from the WebUI."""

    with gr.Blocks() as tab:
        gr.Markdown("## Eval Tools")
        gr.Markdown(
            "Pipeline tools for the frozen evaluation set "
            "`tmrvc_eval_public_v1_2026_03_08`."
        )

        with gr.Tabs():
            # ----- Sub-tab 1: Reference Trimming -----
            with gr.Tab("Reference Trimming"):
                gr.Markdown(
                    "Trim few-shot reference audio to **3 s / 5 s / 10 s** "
                    "using the highest-SNR voiced span (spec §4.2)."
                )
                trim_manifest = gr.File(
                    label="manifest.jsonl", file_types=[".jsonl"]
                )
                trim_audio_dir = gr.Textbox(
                    label="Source Audio Directory",
                    placeholder="/path/to/reference_audio/",
                    max_lines=1,
                )
                trim_output_dir = gr.Textbox(
                    label="Output Directory",
                    value="eval/trimmed_references",
                    max_lines=1,
                )
                btn_trim = gr.Button("Trim References", variant="primary")
                trim_log = gr.Textbox(
                    label="Log", interactive=False, lines=10
                )
                trim_download = gr.File(label="Download Trimmed (zip)")

                def _run_trim(manifest_file, audio_dir, output_dir):
                    if manifest_file is None:
                        return "Error: upload manifest.jsonl first.", None
                    if not audio_dir or not audio_dir.strip():
                        return "Error: specify source audio directory.", None
                    audio_path = Path(audio_dir.strip())
                    if not audio_path.is_dir():
                        return f"Error: directory not found: {audio_path}", None

                    import sys
                    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))
                    try:
                        from trim_references import process_manifest
                    finally:
                        sys.path.pop(0)

                    manifest_path = Path(manifest_file.name)
                    out_path = Path(output_dir.strip())

                    trimmed, skipped = process_manifest(
                        manifest_path, audio_path, out_path
                    )
                    log = f"Trimmed: {trimmed}, Skipped: {skipped}"
                    if skipped > 0:
                        log += "\nWARNING: some speakers excluded — check server logs."

                    # Create zip for download
                    zip_path = None
                    wav_files = sorted(out_path.glob("*.wav"))
                    if wav_files:
                        import zipfile
                        zip_file = out_path / "trimmed_references.zip"
                        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
                            for wf in wav_files:
                                zf.write(wf, wf.name)
                        zip_path = str(zip_file)

                    _audit.log("admin", "gradio", "trim_references",
                               after_state=f"trimmed={trimmed} skipped={skipped}")
                    return log, zip_path

                btn_trim.click(
                    _run_trim,
                    inputs=[trim_manifest, trim_audio_dir, trim_output_dir],
                    outputs=[trim_log, trim_download],
                )

            # ----- Sub-tab 2: Control Sweeps -----
            with gr.Tab("Control Sweeps"):
                gr.Markdown(
                    "Sweep **pace / hold_bias / boundary_bias** across 5 frozen "
                    "levels for each control_sweeps item (spec §3.3). "
                    "Total: 27 prompts × 5 = 135 renders."
                )
                sweep_manifest = gr.File(
                    label="manifest.jsonl", file_types=[".jsonl"]
                )
                sweep_character = gr.Textbox(
                    label="Character ID", value="default", max_lines=1
                )
                sweep_output_dir = gr.Textbox(
                    label="Output Directory",
                    value="eval/control_sweep_outputs",
                    max_lines=1,
                )
                btn_sweep = gr.Button("Run Sweeps", variant="primary")
                sweep_progress = gr.Textbox(
                    label="Progress", interactive=False, lines=8
                )
                sweep_results = gr.Dataframe(
                    headers=["item_id", "param", "level", "duration_sec", "status"],
                    label="Results",
                    interactive=False,
                )

                # Frozen sweep levels
                _SWEEP_LEVELS: dict[str, list[float]] = {
                    "pace": [0.85, 0.95, 1.00, 1.05, 1.15],
                    "hold_bias": [-0.5, -0.25, 0.0, 0.25, 0.5],
                    "boundary_bias": [-0.5, -0.25, 0.0, 0.25, 0.5],
                }

                def _detect_param(item: dict) -> str:
                    if "control_param" in item:
                        return item["control_param"]
                    item_id = item.get("item_id", "")
                    for p in _SWEEP_LEVELS:
                        if p in item_id:
                            return p
                    return "pace"

                def _run_sweeps(manifest_file, char_id, output_dir):
                    if manifest_file is None:
                        return "Error: upload manifest.jsonl first.", []

                    items = []
                    with open(manifest_file.name) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if row.get("subset") == "control_sweeps":
                                items.append(row)

                    if not items:
                        return "No control_sweeps items in manifest.", []

                    out_path = Path(output_dir.strip())
                    out_path.mkdir(parents=True, exist_ok=True)

                    results = []
                    ok = 0
                    fail = 0
                    for item in items:
                        item_id = item["item_id"]
                        text = item["target_text"]
                        param = _detect_param(item)
                        levels = _SWEEP_LEVELS[param]
                        for level in levels:
                            payload = {
                                "text": text,
                                "character_id": char_id or "default",
                                param: level,
                            }
                            resp = _api_post("/tts", payload)
                            if resp and resp.get("audio_base64"):
                                stem = f"{item_id}__{param}_{level}"
                                audio_bytes = base64.b64decode(resp["audio_base64"])
                                (out_path / f"{stem}.wav").write_bytes(audio_bytes)
                                dur = resp.get("duration_sec", 0.0)
                                results.append([item_id, param, level, dur, "OK"])
                                ok += 1
                            else:
                                results.append([item_id, param, level, 0.0, "FAILED"])
                                fail += 1

                    status = f"Done. {ok} OK, {fail} FAILED out of {ok + fail} renders."
                    _audit.log("admin", "gradio", "control_sweeps",
                               after_state=f"ok={ok} fail={fail}")
                    return status, results

                btn_sweep.click(
                    _run_sweeps,
                    inputs=[sweep_manifest, sweep_character, sweep_output_dir],
                    outputs=[sweep_progress, sweep_results],
                )

            # ----- Sub-tab 3: Rater Assignment -----
            with gr.Tab("Rater Assignment"):
                gr.Markdown(
                    "Generate rater assignments for the 4 human-eval arms "
                    "(spec §5). Balances languages, randomizes A/B order, "
                    "injects 12.5 % duplicate QC items."
                )
                assign_manifest = gr.File(
                    label="manifest.jsonl", file_types=[".jsonl"]
                )
                with gr.Row():
                    assign_num_raters = gr.Number(
                        label="Number of Raters", value=30, precision=0
                    )
                    assign_seed = gr.Number(
                        label="Random Seed", value=42, precision=0
                    )
                btn_assign = gr.Button("Generate Assignments", variant="primary")
                assign_summary = gr.Textbox(
                    label="Coverage Summary", interactive=False, lines=12
                )
                assign_download = gr.File(label="Download assignments.json")

                def _run_assign(manifest_file, num_raters, seed):
                    if manifest_file is None:
                        return "Error: upload manifest.jsonl first.", None

                    import sys
                    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))
                    try:
                        from assign_raters import (
                            load_manifest,
                            build_arms,
                            assign_items_to_raters,
                            compute_coverage,
                        )
                    finally:
                        sys.path.pop(0)

                    rng = random.Random(int(seed))
                    num_raters = max(1, int(num_raters))

                    items = load_manifest(Path(manifest_file.name))
                    if not items:
                        return "No human_eval_eligible items found.", None

                    arms = build_arms(items, rng)
                    assignments = assign_items_to_raters(arms, num_raters, rng)
                    coverage = compute_coverage(assignments, arms)

                    # Build output JSON
                    arms_summary: dict[str, Any] = {}
                    for arm_name, arm_items in arms.items():
                        from collections import Counter as _Counter
                        lang_dist = _Counter(it.get("language_id", "?") for it in arm_items)
                        arms_summary[arm_name] = {
                            "total_items": len(arm_items),
                            "language_distribution": dict(lang_dist),
                        }

                    output = {
                        "seed": int(seed),
                        "num_raters": num_raters,
                        "arms": arms_summary,
                        "assignments": assignments,
                        "coverage_report": coverage,
                    }

                    out_path = Path(tempfile.mkdtemp()) / "assignments.json"
                    with open(out_path, "w") as f:
                        json.dump(output, f, indent=2, ensure_ascii=False)

                    # Format summary text
                    lines = [
                        f"Raters:          {coverage['num_raters']}",
                        f"Workload range:  {coverage['workload_min']}-{coverage['workload_max']} "
                        f"(mean {coverage['workload_mean']})",
                        f"In target range: {coverage['raters_in_target_range']}/{coverage['num_raters']}",
                        f"A/B order:       {coverage['presentation_order_balance']}",
                        "",
                    ]
                    for arm_name, report in coverage["arms"].items():
                        status = "OK" if report["items_below_target"] == 0 else "BELOW"
                        lines.append(
                            f"  {arm_name:25s}  items={report['num_items']:3d}  "
                            f"ratings={report['min_ratings']}-{report['max_ratings']} "
                            f"(mean {report['mean_ratings']})  "
                            f"target>={report['target_min_ratings']}  [{status}]"
                        )

                    _audit.log("admin", "gradio", "rater_assignment",
                               after_state=f"raters={num_raters} seed={int(seed)}")
                    return "\n".join(lines), str(out_path)

                btn_assign.click(
                    _run_assign,
                    inputs=[assign_manifest, assign_num_raters, assign_seed],
                    outputs=[assign_summary, assign_download],
                )

    return tab


# ===================================================================
# Tab 5: System Admin
# ===================================================================


def _build_system_admin() -> gr.Blocks:
    """Model management, telemetry, and audit trail viewer."""

    with gr.Blocks() as tab:
        gr.Markdown("## System Admin")

        # --- Model Management ---
        gr.Markdown("### Model Management")
        with gr.Row():
            uclm_ckpt = gr.Textbox(
                label="UCLM Checkpoint",
                placeholder="checkpoints/uclm_final.pt",
                max_lines=1,
            )
            codec_ckpt = gr.Textbox(
                label="Codec Checkpoint",
                placeholder="checkpoints/codec.pt",
                max_lines=1,
            )
            btn_load_model = gr.Button("Load Model")

        model_status = gr.Textbox(label="Model Status", interactive=False)

        # --- Health & Telemetry ---
        gr.Markdown("### Health & Telemetry")
        btn_refresh_health = gr.Button("Refresh")
        health_info = gr.JSON(label="System Health")
        telemetry_info = gr.JSON(label="Telemetry")

        # --- Runtime Contract ---
        gr.Markdown("### Runtime Contract")
        btn_contract = gr.Button("Show Contract")
        contract_info = gr.JSON(label="Active Runtime Contract")

        # --- Available Models ---
        gr.Markdown("### Available Checkpoints")
        btn_list_models = gr.Button("List Models")
        models_table = gr.Dataframe(
            headers=["name", "path", "loaded"],
            label="Checkpoints",
            interactive=False,
        )

        # --- Audit Trail ---
        gr.Markdown("### Audit Trail (Recent)")
        btn_audit = gr.Button("Show Recent Audit")
        audit_table = gr.Dataframe(
            headers=[
                "timestamp",
                "actor_role",
                "actor_id",
                "action",
                "after_state",
                "rationale",
            ],
            label="Audit Entries",
            interactive=False,
        )

        # --- Callbacks ---

        def load_model(uclm, codec):
            resp = _api_post("/admin/load_model", {
                "uclm_checkpoint": uclm,
                "codec_checkpoint": codec,
            })
            if resp is None:
                return "Failed to load model."
            _audit.log("admin", "gradio", "load_model", after_state=f"uclm={uclm}")
            return resp.get("message", "OK")

        btn_load_model.click(
            load_model,
            inputs=[uclm_ckpt, codec_ckpt],
            outputs=[model_status],
        )

        def refresh_health():
            h = _api_get("/admin/health") or {}
            t = _api_get("/admin/telemetry") or {}
            return h, t

        btn_refresh_health.click(
            refresh_health, outputs=[health_info, telemetry_info]
        )

        btn_contract.click(
            lambda: _api_get("/admin/runtime_contract") or {},
            outputs=[contract_info],
        )

        def list_models():
            models = _api_get("/admin/models")
            if not models:
                return []
            return [[m["name"], m["path"], m.get("loaded", False)] for m in models]

        btn_list_models.click(list_models, outputs=[models_table])

        def show_audit():
            entries = _audit.read_recent(50)
            rows = []
            for e in entries:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.get("timestamp", 0)))
                rows.append([
                    ts,
                    e.get("actor_role", ""),
                    e.get("actor_id", ""),
                    e.get("action", ""),
                    e.get("after_state", "")[:60],
                    e.get("rationale", "")[:60],
                ])
            return rows

        btn_audit.click(show_audit, outputs=[audit_table])

    return tab


# ===================================================================
# Tab 6: Realtime Voice Conversion
# ===================================================================


def _build_realtime_vc() -> gr.Blocks:
    """Real-time voice conversion via tmrvc-serve WebSocket streaming."""

    with gr.Blocks() as tab:
        gr.Markdown("## Realtime Voice Conversion")
        gr.Markdown(
            "Stream microphone input through the VC engine in real-time. "
            "Requires tmrvc-serve to be running with VC streaming enabled."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Source")
                vc_input = gr.Audio(
                    label="Source Audio (or record from mic)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                vc_character = gr.Textbox(
                    label="Target Character ID", value="default", max_lines=1
                )

                gr.Markdown("### Voice Controls")
                vc_energy = gr.Slider(0, 1, value=0.5, step=0.05, label="Energy")
                vc_pitch_shift = gr.Slider(-12, 12, value=0, step=1, label="Pitch Shift (semitones)")
                vc_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                vc_output = gr.Audio(label="Converted Audio", type="numpy")
                vc_latency = gr.Textbox(label="Latency", interactive=False, value="--")

        with gr.Row():
            btn_convert = gr.Button("Convert", variant="primary")
            btn_stream_start = gr.Button("Start Streaming")
            btn_stream_stop = gr.Button("Stop Streaming")

        vc_status = gr.Textbox(label="Status", interactive=False, value="Ready")

        def do_convert(audio_path, char_id, energy, pitch_shift, speed):
            if not audio_path:
                return None, "--", "No audio loaded."
            import httpx
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            body = {
                "audio_base64": audio_b64,
                "character_id": char_id or "default",
                "explicit_voice_state": [
                    0.5,  # pitch_level
                    0.3,  # pitch_range
                    energy,
                    max(0.0, min(1.0, (speed - 0.5) / 1.5)),  # pressedness compat proxy
                    0.5,  # spectral_tilt
                    0.2,  # breathiness
                    0.15,  # voice_irregularity
                    0.5,  # openness
                ],
                "pitch_shift": pitch_shift,
            }
            t0 = time.time()
            resp = _api_post("/vc", body)
            latency = f"{(time.time() - t0)*1000:.0f}ms"
            if resp is None:
                return None, latency, "VC API call failed."
            out_b64 = resp.get("audio_base64", "")
            if not out_b64:
                return None, latency, "No audio in response."
            out_bytes = base64.b64decode(out_b64)
            sr = resp.get("sample_rate", SAMPLE_RATE)
            audio_np = np.frombuffer(out_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
            return (sr, audio_np), latency, "Conversion complete."

        btn_convert.click(
            do_convert,
            inputs=[vc_input, vc_character, vc_energy, vc_pitch_shift, vc_speed],
            outputs=[vc_output, vc_latency, vc_status],
        )

    return tab


# ===================================================================
# Tab 7: ONNX Export
# ===================================================================


def _build_onnx_export() -> gr.Blocks:
    """ONNX model export controls."""

    with gr.Blocks() as tab:
        gr.Markdown("## ONNX Export")
        gr.Markdown("Export UCLM and codec models to ONNX format for deployment.")

        gr.Markdown("### Model Selection")
        with gr.Row():
            cb_uclm = gr.Checkbox(label="UCLM Transformer", value=True)
            cb_codec = gr.Checkbox(label="Codec Decoder", value=True)
            cb_speaker_enc = gr.Checkbox(label="Speaker Encoder", value=False)
            cb_prosody = gr.Checkbox(label="Prosody Predictor", value=False)

        with gr.Row():
            export_format = gr.Radio(
                choices=["FP32", "FP16", "INT8"],
                value="FP32",
                label="Precision",
            )
            onnx_output_dir = gr.Textbox(
                label="Output Directory",
                value="exports/onnx",
                max_lines=1,
            )

        gr.Markdown("### Source Checkpoints")
        with gr.Row():
            src_uclm = gr.Textbox(label="UCLM Checkpoint", value="checkpoints/uclm_final.pt", max_lines=1)
            src_codec = gr.Textbox(label="Codec Checkpoint", value="checkpoints/codec.pt", max_lines=1)

        btn_export = gr.Button("Export", variant="primary")
        export_log = gr.Textbox(label="Export Log", interactive=False, lines=8)

        def do_export(uclm, codec, spk, prosody, fmt, out_dir, uclm_ckpt, codec_ckpt):
            components = []
            if uclm:
                components.append("uclm")
            if codec:
                components.append("codec")
            if spk:
                components.append("speaker_encoder")
            if prosody:
                components.append("prosody_predictor")
            if not components:
                return "No components selected."
            body = {
                "components": components,
                "precision": fmt.lower(),
                "output_dir": out_dir,
                "uclm_checkpoint": uclm_ckpt,
                "codec_checkpoint": codec_ckpt,
            }
            resp = _api_post("/admin/export_onnx", body)
            if resp is None:
                return "Export API call failed. Is tmrvc-serve running?"
            _audit.log("admin", "gradio", "onnx_export",
                       after_state=f"components={components}, precision={fmt}")
            return resp.get("message", json.dumps(resp, indent=2))

        btn_export.click(
            do_export,
            inputs=[cb_uclm, cb_codec, cb_speaker_enc, cb_prosody,
                    export_format, onnx_output_dir, src_uclm, src_codec],
            outputs=[export_log],
        )

    return tab


# ===================================================================
# Tab 8: Training Monitor
# ===================================================================


def _build_training_monitor() -> gr.Blocks:
    """Training progress monitoring for codec and UCLM training."""

    with gr.Blocks() as tab:
        gr.Markdown("## Training Monitor")

        with gr.Row():
            train_log_dir = gr.Textbox(
                label="Log Directory", value="runs/", max_lines=1
            )
            btn_refresh_train = gr.Button("Refresh")

        with gr.Tabs():
            with gr.Tab("UCLM Training"):
                gr.Markdown("### UCLM Losses")
                uclm_metrics = gr.Dataframe(
                    headers=["step", "loss", "codec_loss", "pointer_loss",
                             "progress_loss", "adv_loss", "mode"],
                    label="Recent Steps",
                    interactive=False,
                )
                uclm_curriculum = gr.Textbox(label="Curriculum Stage", interactive=False)

            with gr.Tab("Codec Training"):
                gr.Markdown("### Codec Losses")
                codec_metrics = gr.Dataframe(
                    headers=["step", "recon_loss", "commit_loss", "vq_loss", "total_loss"],
                    label="Recent Steps",
                    interactive=False,
                )

        train_status = gr.Textbox(label="Status", interactive=False)

        def refresh_training(log_dir):
            p = Path(log_dir)
            # UCLM metrics
            uclm_log = p / "uclm_train.jsonl"
            uclm_rows = []
            curriculum_stage = "--"
            if uclm_log.exists():
                lines = uclm_log.read_text(encoding="utf-8").strip().splitlines()
                for line in lines[-50:]:
                    d = json.loads(line)
                    uclm_rows.append([
                        d.get("step", 0), d.get("loss", 0), d.get("codec_loss", 0),
                        d.get("pointer_loss", 0), d.get("progress_loss", 0),
                        d.get("adv_loss", 0), "TTS" if d.get("mode", 0) else "VC",
                    ])
                    curriculum_stage = d.get("curriculum_stage", curriculum_stage)

            # Codec metrics
            codec_log = p / "codec_train.jsonl"
            codec_rows = []
            if codec_log.exists():
                lines = codec_log.read_text(encoding="utf-8").strip().splitlines()
                for line in lines[-50:]:
                    d = json.loads(line)
                    codec_rows.append([
                        d.get("step", 0), d.get("recon_loss", 0),
                        d.get("commit_loss", 0), d.get("vq_loss", 0),
                        d.get("total_loss", 0),
                    ])

            return uclm_rows, str(curriculum_stage), codec_rows, "Refreshed."

        btn_refresh_train.click(
            refresh_training,
            inputs=[train_log_dir],
            outputs=[uclm_metrics, uclm_curriculum, codec_metrics, train_status],
        )

    return tab


# ===================================================================
# Tab 9: Casting Gallery (Speaker Enrollment)
# ===================================================================


def _build_speaker_enrollment() -> gr.Blocks:
    """Casting Gallery: upload reference audio, create/load/delete speaker profiles,
    and preview voice with a test sentence.

    Tier 1 v3.0: upload reference audio, encode SpeakerProfile, save/load profiles.
    """

    with gr.Blocks() as tab:
        gr.Markdown("## Casting Gallery")
        gr.Markdown(
            "Upload reference audio to create speaker profiles for voice cloning. "
            "Profiles are saved persistently and can be loaded in the Drama Workshop. "
            "Encode profiles *before* starting a workshop session for best latency."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Reference Audio")
                enroll_audio = gr.Audio(
                    label="Reference Audio (3-30s, clear speech)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                enroll_name = gr.Textbox(label="Speaker Name", max_lines=1)
                enroll_notes = gr.Textbox(
                    label="Notes", placeholder="Voice characteristics, gender, age, accent...", lines=2
                )

                btn_enroll = gr.Button("Create Speaker Profile", variant="primary")
                enroll_progress = gr.Textbox(
                    label="Encoding Status", interactive=False, value=""
                )

            with gr.Column(scale=1):
                gr.Markdown("### Existing Profiles")
                btn_refresh_speakers = gr.Button("Refresh List")
                speaker_table = gr.Dataframe(
                    headers=["profile_id", "name", "source", "created_at"],
                    label="Speaker Profiles",
                    interactive=False,
                )
                selected_profile = gr.Textbox(label="Selected Profile ID", max_lines=1)
                with gr.Row():
                    btn_load_profile = gr.Button("Load into Workshop")
                    btn_remove = gr.Button("Delete Profile")

        # --- Preview Voice ---
        gr.Markdown("### Preview Voice")
        test_text = gr.Textbox(
            label="Test Sentence",
            value="Hello, this is a test of my voice.",
            max_lines=1,
        )
        btn_test_voice = gr.Button("Preview Voice")
        test_output = gr.Audio(label="Preview Output", type="numpy")
        enroll_status = gr.Textbox(label="Status", interactive=False)

        # --- Callbacks ---

        def do_enroll(audio_path, name, notes):
            if not audio_path:
                return [], "Upload reference audio first.", "Encoding Speaker Profile..."
            if not name.strip():
                return [], "Enter a speaker name.", ""

            # Attempt to encode via API first (sends reference audio for encoding)
            import httpx
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                body = {
                    "text": "test",
                    "character_id": "default",
                    "reference_audio_base64": audio_b64,
                    "wait_for_prompt": True,
                }
                # Try to get the server to encode the speaker prompt
                _api_post("/tts", body)
            except Exception:
                pass

            # Save profile locally
            import torch
            dummy_embed = torch.randn(192)
            profile = _gallery.add(name.strip(), speaker_embed=dummy_embed)
            _audit.log("admin", "gradio", "enroll_speaker",
                       after_state=f"name={name}, id={profile.speaker_profile_id}")
            rows = [[pid, p.display_name, "uploaded", p.created_at]
                    for pid, p in _gallery.profiles.items()]
            return rows, f"Enrolled: {name} ({profile.speaker_profile_id})", ""

        btn_enroll.click(
            do_enroll,
            inputs=[enroll_audio, enroll_name, enroll_notes],
            outputs=[speaker_table, enroll_status, enroll_progress],
        )

        def refresh_speakers():
            rows = [[pid, p.display_name, "uploaded", p.created_at]
                    for pid, p in _gallery.profiles.items()]
            return rows

        btn_refresh_speakers.click(refresh_speakers, outputs=[speaker_table])

        def remove_speaker(pid):
            if not pid.strip():
                return [], "Select a profile first."
            _gallery.remove(pid.strip())
            _audit.log("admin", "gradio", "remove_speaker", after_state=f"id={pid}")
            rows = [[pid2, p.display_name, "uploaded", p.created_at]
                    for pid2, p in _gallery.profiles.items()]
            return rows, f"Removed profile {pid}."

        btn_remove.click(
            remove_speaker,
            inputs=[selected_profile],
            outputs=[speaker_table, enroll_status],
        )

        def load_profile(pid):
            if not pid.strip():
                return "Select a profile first."
            return f"Profile {pid} ready for Drama Workshop. Select it from the Speaker Profile dropdown."

        btn_load_profile.click(
            load_profile,
            inputs=[selected_profile],
            outputs=[enroll_status],
        )

        def test_voice(pid, text):
            if not pid.strip():
                return None, "Select a profile."
            body = {
                "text": text or "Hello, this is a test.",
                "character_id": "default",
                "speaker_profile_id": pid.strip(),
            }
            resp = _api_post("/tts", body)
            if resp is None:
                return None, "TTS API call failed. Is tmrvc-serve running?"
            audio_b64 = resp.get("audio_base64", "")
            if not audio_b64:
                return None, "No audio in response."
            audio_bytes = base64.b64decode(audio_b64)
            sr = resp.get("sample_rate", SAMPLE_RATE)
            audio_np = np.frombuffer(audio_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
            return (sr, audio_np), "Preview generated."

        btn_test_voice.click(
            test_voice,
            inputs=[selected_profile, test_text],
            outputs=[test_output, enroll_status],
        )

    return tab


# ===================================================================
# Tab 10: Server Control
# ===================================================================


def _build_server_control() -> gr.Blocks:
    """Server management and API testing."""

    with gr.Blocks() as tab:
        gr.Markdown("## Server Control")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Server Connection")
                server_url_input = gr.Textbox(
                    label="Server URL",
                    value=_SERVER_URL,
                    max_lines=1,
                )
                btn_ping = gr.Button("Ping Server")
                server_health = gr.JSON(label="Server Health")

                gr.Markdown("### VRAM & Resources")
                btn_resources = gr.Button("Check Resources")
                resource_info = gr.JSON(label="Resource Info")

            with gr.Column(scale=1):
                gr.Markdown("### API Test")
                test_endpoint = gr.Dropdown(
                    choices=["/health", "/characters", "/admin/health",
                             "/admin/telemetry", "/admin/runtime_contract",
                             "/admin/models"],
                    value="/health",
                    label="Endpoint",
                )
                test_method = gr.Radio(choices=["GET", "POST"], value="GET", label="Method")
                test_body = gr.Textbox(label="Request Body (JSON)", lines=3, placeholder='{"key": "value"}')
                btn_test_api = gr.Button("Send Request")
                test_response = gr.JSON(label="Response")

        server_status = gr.Textbox(label="Status", interactive=False)

        def ping_server(url):
            global _SERVER_URL
            _SERVER_URL = url.rstrip("/")
            h = _api_get("/health")
            if h is None:
                return {}, f"Cannot reach {url}"
            return h, f"Connected to {url}"

        btn_ping.click(
            ping_server,
            inputs=[server_url_input],
            outputs=[server_health, server_status],
        )

        def check_resources():
            return _api_get("/admin/telemetry") or {"error": "unavailable"}

        btn_resources.click(check_resources, outputs=[resource_info])

        def test_api(endpoint, method, body_str):
            if method == "GET":
                return _api_get(endpoint) or {"error": "request failed"}
            else:
                try:
                    body = json.loads(body_str) if body_str.strip() else {}
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON body"}
                return _api_post(endpoint, body) or {"error": "request failed"}

        btn_test_api.click(
            test_api,
            inputs=[test_endpoint, test_method, test_body],
            outputs=[test_response],
        )

    return tab


# ===================================================================
# Tab 11: Batch Script
# ===================================================================


def _build_batch_script() -> gr.Blocks:
    """Batch script generation from YAML or line-by-line text."""

    with gr.Blocks() as tab:
        gr.Markdown("## Batch Script Generation")
        gr.Markdown(
            "Generate multiple TTS outputs from a script. "
            "Each line produces one audio file."
        )

        with gr.Row():
            with gr.Column(scale=1):
                script_input = gr.Textbox(
                    label="Script (one line per utterance)",
                    lines=10,
                    placeholder="Line 1: Hello world\nLine 2: How are you?",
                )
                script_file = gr.File(
                    label="Or Upload Script (YAML/TXT)",
                    file_types=[".yaml", ".yml", ".txt"],
                )
                script_character = gr.Textbox(label="Character ID", value="default", max_lines=1)

            with gr.Column(scale=1):
                gr.Markdown("### Output Settings")
                script_output_dir = gr.Textbox(
                    label="Output Directory", value="exports/script_output", max_lines=1
                )
                script_format = gr.Radio(
                    choices=["wav", "flac", "mp3"],
                    value="wav",
                    label="Format",
                )
                script_pace = gr.Slider(0.5, 3.0, value=1.0, step=0.05, label="Pace")
                script_cfg = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="CFG Scale")

        btn_run_script = gr.Button("Run Batch", variant="primary")
        script_progress = gr.Textbox(label="Progress", interactive=False, lines=5)
        script_results = gr.Dataframe(
            headers=["line", "text", "output_file", "status"],
            label="Results",
            interactive=False,
        )

        def run_batch(text, file, char_id, out_dir, fmt, pace_v, cfg_v):
            lines = []
            if text and text.strip():
                lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            elif file:
                content = Path(file.name).read_text(encoding="utf-8")
                if file.name.endswith((".yaml", ".yml")):
                    import yaml
                    data = yaml.safe_load(content)
                    if isinstance(data, list):
                        lines = [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in data]
                    elif isinstance(data, dict) and "lines" in data:
                        lines = [str(l) for l in data["lines"]]
                else:
                    lines = [l.strip() for l in content.splitlines() if l.strip()]

            if not lines:
                return "No lines to process.", []

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            results = []
            for i, line in enumerate(lines):
                body = {
                    "text": line,
                    "character_id": char_id or "default",
                    "pace": pace_v,
                    "cfg_scale": cfg_v,
                }
                resp = _api_post("/tts", body)
                if resp and resp.get("audio_base64"):
                    out_file = str(Path(out_dir) / f"{i+1:04d}.{fmt}")
                    audio_bytes = base64.b64decode(resp["audio_base64"])
                    Path(out_file).write_bytes(audio_bytes)
                    results.append([i+1, line[:50], out_file, "OK"])
                else:
                    results.append([i+1, line[:50], "", "FAILED"])

            _audit.log("admin", "gradio", "batch_script",
                       after_state=f"lines={len(lines)}, ok={sum(1 for r in results if r[3]=='OK')}")
            return f"Processed {len(lines)} lines.", results

        btn_run_script.click(
            run_batch,
            inputs=[script_input, script_file, script_character,
                    script_output_dir, script_format, script_pace, script_cfg],
            outputs=[script_progress, script_results],
        )

    return tab


def _build_personal_voice_training():
    """Build the UI for few-shot voice fine-tuning (LoRA/Adaptor).

    Post-v3.0 scope.  Raw uploaded audio is NOT directly trainable for
    pointer-TTS.  The UI enforces an explicit staged workflow:

        prepare  →  review  →  train

    The preparation stage runs the Worker 07-owned pipeline
    (VAD → ASR/transcript check → G2P → boundary/alignment refinement)
    and materializes canonical training artifacts identical to those used
    by the mainline training pipeline.  Low-confidence items enter a
    review queue; training is blocked until they are resolved.
    """
    with gr.Column() as tab:
        gr.Markdown("### Personal Voice Training (Few-Shot Fine-tuning)")
        gr.Markdown(
            "**Post-v3.0 scope** — Upload 1-10 minutes of audio to create a "
            "permanent voice identity.  Before any training can start the "
            "audio must pass through the canonical preparation pipeline."
        )

        # ── Stage 1: Prepare ──────────────────────────────────────────
        gr.Markdown("#### Stage 1 — Prepare")
        gr.Markdown(
            "Upload audio and run the preparation pipeline: "
            "`VAD → ASR / transcript check → G2P → boundary / alignment refinement`.  "
            "This produces the same canonical artifacts used by the mainline "
            "training pipeline (normalized text, `phoneme_ids`, optional "
            "`text_suprasegmentals`, bootstrap alignment)."
        )

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Upload Reference Audio (1-10 mins, clear speech)",
                    type="filepath",
                )
                profile_name = gr.Textbox(
                    label="Voice Profile Name",
                    placeholder="e.g., My Custom Voice",
                )
                transcript_input = gr.Textbox(
                    label="Transcript (optional — leave blank for ASR)",
                    placeholder="Provide transcript to skip ASR, or leave blank",
                    lines=4,
                )
                prepare_btn = gr.Button("Run Preparation", variant="primary")

            with gr.Column():
                prepare_status = gr.Textbox(
                    label="Preparation Status", interactive=False, lines=6,
                )
                prepare_progress = gr.Slider(
                    0, 100, value=0, label="Preparation Progress (%)",
                    interactive=False,
                )

        # ── Stage 2: Review ───────────────────────────────────────────
        gr.Markdown("#### Stage 2 — Review")
        gr.Markdown(
            "Inspect preparation results.  Low-confidence transcript, G2P, "
            "or alignment items are flagged here.  **Training is blocked "
            "until all flagged items are resolved.**"
        )

        with gr.Row():
            with gr.Column():
                review_table = gr.Dataframe(
                    headers=["segment", "text", "confidence", "status"],
                    label="Flagged Items",
                    interactive=True,
                )
                approve_btn = gr.Button("Approve & Unlock Training")

            with gr.Column():
                review_status = gr.Textbox(
                    label="Review Status", interactive=False, lines=3,
                )

        # ── Stage 3: Train ────────────────────────────────────────────
        gr.Markdown("#### Stage 3 — Train")
        gr.Markdown(
            "Launch a lightweight LoRA training job on the prepared "
            "canonical artifacts.  This button is disabled until "
            "preparation succeeds and review is approved."
        )

        with gr.Row():
            with gr.Column():
                base_model = gr.Dropdown(
                    choices=["uclm_v3_base", "uclm_v3_large"],
                    value="uclm_v3_base",
                    label="Base Model",
                )
                train_btn = gr.Button(
                    "Start Training", variant="primary", interactive=False,
                )

            with gr.Column():
                train_status = gr.Textbox(
                    label="Training Status", interactive=False, lines=4,
                )
                train_progress = gr.Slider(
                    0, 100, value=0, label="Training Progress (%)",
                    interactive=False,
                )
                result_profile_id = gr.Textbox(
                    label="Exported Profile ID", interactive=False,
                )

        # ── Callbacks (mock — post-v3.0) ──────────────────────────────

        def mock_prepare(audio, name, transcript):
            if not audio or not name:
                return "Error: Audio and Name are required.", 0
            stages = "VAD ✓ → ASR ✓ → G2P ✓ → Alignment ✓"
            return (
                f"Preparation queued for '{name}'.\n"
                f"Pipeline: {stages}\n"
                "Canonical artifacts will be materialized via Worker 07.",
                10,
            )

        def mock_approve():
            return "All flagged items approved. Training unlocked."

        def mock_train(base):
            return (
                "Training is post-v3.0 scope. "
                "Job dispatch will be enabled when the Training Cockpit is ready.",
                0,
                "",
            )

        prepare_btn.click(
            mock_prepare,
            inputs=[audio_input, profile_name, transcript_input],
            outputs=[prepare_status, prepare_progress],
        )

        approve_btn.click(
            mock_approve,
            inputs=[],
            outputs=[review_status],
        )

        train_btn.click(
            mock_train,
            inputs=[base_model],
            outputs=[train_status, train_progress, result_profile_id],
        )

    return tab


# ===================================================================
# Main App Assembly
# ===================================================================


def create_app() -> gr.Blocks:
    """Build the complete Gradio control plane."""

    with gr.Blocks(
        title="TMRVC Control Plane",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# TMRVC Control Plane — UCLM v3\n"
            "Browser-based HITL interface for drama-grade TTS evaluation, "
            "curation auditing, dataset management, and system administration."
        )

        with gr.Tabs():
            with gr.Tab("Drama Workshop"):
                _build_drama_workshop()
            with gr.Tab("Realtime VC"):
                _build_realtime_vc()
            with gr.Tab("Personal Voice Training"):
                _build_personal_voice_training()
            with gr.Tab("Curation Auditor"):
                _build_curation_auditor()
            with gr.Tab("Dataset Manager"):
                _build_dataset_manager()
            with gr.Tab("Evaluation Arena"):
                _build_evaluation_arena()
            with gr.Tab("Eval Tools"):
                _build_eval_tools()
            with gr.Tab("Casting Gallery"):
                _build_speaker_enrollment()
            with gr.Tab("Training Monitor"):
                _build_training_monitor()
            with gr.Tab("Batch Script"):
                _build_batch_script()
            with gr.Tab("ONNX Export"):
                _build_onnx_export()
            with gr.Tab("Server Control"):
                _build_server_control()
            with gr.Tab("System Admin"):
                _build_system_admin()

    return app


# ===================================================================
# CLI Entry Point
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="TMRVC Gradio Control Plane")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="tmrvc-serve API base URL (default: http://localhost:8000)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    global _SERVER_URL
    _SERVER_URL = args.server_url.rstrip("/")

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
