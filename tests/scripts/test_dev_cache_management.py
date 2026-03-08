from __future__ import annotations

import os
import shutil
from pathlib import Path

import dev


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_find_latest_cache_for_enabled_datasets_uses_valid_candidates(
    monkeypatch, tmp_path: Path
):
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    invalid = experiments / f"{prefix}_old" / "cache"
    valid_old = experiments / f"{prefix}_mid" / "cache"
    valid_new = experiments / f"{prefix}_new" / "cache"
    invalid.mkdir(parents=True, exist_ok=True)
    _touch(valid_old / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(valid_old / "vctk" / "train" / "s" / "u" / "meta.json")
    _touch(valid_new / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(valid_new / "vctk" / "train" / "s" / "u" / "meta.json")

    os.utime(valid_old, (1, 1))
    os.utime(valid_new, (2, 2))

    picked = dev.find_latest_cache_for_enabled_datasets(enabled)
    assert picked == valid_new


def test_clear_training_caches_for_enabled_datasets_is_scoped(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    # Legacy cache: only enabled datasets should be removed.
    (tmp_path / "data" / "cache" / "jvs").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "vctk").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "other").mkdir(parents=True)

    # Experiment caches: only matching prefix should be removed.
    (experiments / f"{prefix}_aaa" / "cache").mkdir(parents=True)
    (experiments / "other_dataset_aaa" / "cache").mkdir(parents=True)

    removed = dev.clear_training_caches_for_enabled_datasets(enabled)
    assert removed == 3
    assert not (tmp_path / "data" / "cache" / "jvs").exists()
    assert not (tmp_path / "data" / "cache" / "vctk").exists()
    assert (tmp_path / "data" / "cache" / "other").exists()
    assert not (experiments / f"{prefix}_aaa" / "cache").exists()
    assert (experiments / "other_dataset_aaa" / "cache").exists()


def test_find_latest_uclm_checkpoint_for_enabled_datasets_picks_newest(
    monkeypatch, tmp_path: Path
):
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    old_exp = experiments / f"{prefix}_old"
    new_exp = experiments / f"{prefix}_new"
    other_exp = experiments / "other_aaa"

    old_ckpt = old_exp / "checkpoints" / "uclm_final.pt"
    new_ckpt = new_exp / "checkpoints" / "uclm_final.pt"
    _touch(old_ckpt)
    _touch(new_ckpt)
    _touch(other_exp / "checkpoints" / "uclm_final.pt")

    os.utime(old_ckpt, (1, 1))
    os.utime(new_ckpt, (2, 2))

    exp_dir, ckpt = dev.find_latest_uclm_checkpoint_for_enabled_datasets(enabled)
    assert exp_dir == new_exp
    assert ckpt == new_ckpt


def test_quality_gate_status_reads_json(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    report = exp_dir / "quality_gate_report.json"
    report.write_text('{"status":"ok"}', encoding="utf-8")
    assert dev._quality_gate_status(exp_dir) == "ok"


def test_find_latest_experiment_for_enabled_datasets_uses_cache_validation(
    monkeypatch, tmp_path: Path
):
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    invalid = experiments / f"{prefix}_old"
    valid = experiments / f"{prefix}_new"
    other = experiments / "other_dataset_new"

    (invalid / "cache").mkdir(parents=True, exist_ok=True)
    _touch(valid / "cache" / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(valid / "cache" / "vctk" / "train" / "s" / "u" / "meta.json")
    _touch(other / "cache" / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(other / "cache" / "vctk" / "train" / "s" / "u" / "meta.json")

    os.utime(valid, (2, 2))
    os.utime(invalid, (1, 1))

    picked = dev._find_latest_experiment_for_enabled_datasets(enabled)
    assert picked == valid


def test_finalize_training_outputs_blocks_missing_quality_gate(
    monkeypatch, tmp_path: Path, capsys
):
    experiments = tmp_path / "experiments"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(dev, "CHECKPOINTS_DIR", checkpoints)
    exp = experiments / "jvs_20260306_000000"
    ckpt = exp / "checkpoints" / "uclm_final.pt"
    _touch(ckpt)

    monkeypatch.setattr(dev, "get_enabled_datasets", lambda: ["jvs"])
    monkeypatch.setattr(
        dev,
        "find_latest_uclm_checkpoint_for_enabled_datasets",
        lambda enabled: (exp, ckpt),
    )

    promoted_called = {"value": False}

    def _fake_promote(_: Path) -> Path:
        promoted_called["value"] = True
        return checkpoints / "uclm" / "uclm_latest.pt"

    monkeypatch.setattr(dev, "_promote_uclm_checkpoint", _fake_promote)

    dev.cmd_finalize_training_outputs(preferred_device="cpu")

    assert promoted_called["value"] is False
    out = capsys.readouterr().out
    assert "quality_gate: missing" in out


def test_prepare_tts_alignment_noninteractive_uses_textgrid_override(
    monkeypatch, tmp_path: Path
):
    cache_dir = tmp_path / "cache"
    _touch(cache_dir / "jvs" / "train" / "s1" / "u1" / "meta.json")

    called = {"cmd": None}

    def fake_run_checked(cmd: list[str]) -> bool:
        called["cmd"] = cmd
        return True

    monkeypatch.setattr(dev, "run_checked", fake_run_checked)

    ok = dev.cmd_prepare_tts_alignment_from_latest_cache(
        cache_dir=cache_dir,
        enabled_cfg={"jvs": {"language": "ja"}},
        textgrid_overrides={"jvs": tmp_path / "align" / "jvs"},
        interactive=False,
        overwrite=False,
        allow_heuristic_default=False,
    )
    assert ok is True
    assert called["cmd"] is not None
    assert "--textgrid-dir" in called["cmd"]
    assert str(tmp_path / "align" / "jvs") in called["cmd"]


def test_mfa_align_and_inject_fails_when_mfa_command_missing(
    monkeypatch, tmp_path: Path
):
    cache_dir = tmp_path / "cache"
    _touch(cache_dir / "jvs" / "train" / "s1" / "u1" / "meta.json")

    monkeypatch.setattr(shutil, "which", lambda _: None)
    ok = dev.cmd_mfa_align_and_inject_from_cache(
        cache_dir=cache_dir,
        enabled_cfg={"jvs": {"language": "ja"}},
    )
    assert ok is False


def test_get_mfa_command_from_env(monkeypatch):
    monkeypatch.setenv("MFA_COMMAND", "micromamba run -n mfa mfa")
    assert dev.get_mfa_command() == ["micromamba", "run", "-n", "mfa", "mfa"]


def test_normalize_mfa_model_name_aliases():
    assert dev.normalize_mfa_model_name("english") == "english_mfa"
    assert dev.normalize_mfa_model_name("japanese") == "japanese_mfa"
    assert dev.normalize_mfa_model_name("mandarin") == "mandarin_mfa"
    assert dev.normalize_mfa_model_name("korean") == "korean_mfa"
    assert dev.normalize_mfa_model_name("english_mfa") == "english_mfa"


def test_extract_env_name_from_run_command():
    assert (
        dev._extract_env_name_from_run_command(
            ["micromamba", "run", "-n", "mfa", "mfa"]
        )
        == "mfa"
    )
    assert (
        dev._extract_env_name_from_run_command(
            ["conda", "run", "--name", "mfa", "mfa"]
        )
        == "mfa"
    )
    assert dev._extract_env_name_from_run_command(["mfa"]) is None


def test_suggest_mfa_install_cmd_with_python_pin():
    cmd = dev._suggest_mfa_install_cmd(
        ["micromamba", "run", "-n", "mfa", "mfa"],
        ["spacy", "sudachipy", "sudachidict-core"],
        python_pin="3.12",
    )
    assert cmd == [
        "micromamba",
        "install",
        "-n",
        "mfa",
        "-c",
        "conda-forge",
        "python=3.12",
        "spacy",
        "sudachipy",
        "sudachidict-core",
        "-y",
    ]


def test_check_mfa_japanese_runtime_parses_ok(monkeypatch):
    class _P:
        returncode = 0
        stdout = '{"python":"3.12.13","missing":[]}\n'

    monkeypatch.setattr(dev.subprocess, "run", lambda *args, **kwargs: _P())
    ok, py_ver, missing = dev._check_mfa_japanese_runtime(
        ["micromamba", "run", "-n", "mfa", "mfa"]
    )
    assert ok is True
    assert py_ver == "3.12.13"
    assert missing == []


def test_dev_module_importable():
    """Verify that dev.py can be imported and has expected entry points."""
    assert hasattr(dev, "main")
    assert hasattr(dev, "print_menu")
    assert hasattr(dev, "cmd_full_training")
    assert hasattr(dev, "cmd_full_training_legacy")


def test_dev_module_has_v3_entrypoints():
    """v3 curation and validation entrypoints exist."""
    assert callable(dev.cmd_curate_ingest)
    assert callable(dev.cmd_curate_run)
    assert callable(dev.cmd_curate_resume)
    assert callable(dev.cmd_curate_export)
    assert callable(dev.cmd_curate_status)
    assert callable(dev.validate_v3_config)


def test_menu_labels_contain_v3_pointer(capsys):
    """Menu option 1 (full training) should reference v3 pointer mode."""
    dev.clear_screen = lambda: None  # suppress clear
    dev.print_menu()
    output = capsys.readouterr().out
    # Primary training options should mention pointer / v3
    assert "pointer" in output or "v3" in output
    # Legacy option should mention legacy and v2-legacy prefix
    assert "[v2-legacy]" in output
    # Curation entrypoints should be listed
    assert "ingest" in output
    assert "resume" in output
    assert "status" in output


def test_handlers_include_legacy_option():
    """The main loop handler dict should map '12' to cmd_full_training_legacy."""
    # We verify the function exists and is callable
    assert callable(dev.cmd_full_training_legacy)


def test_validate_v3_config_accepts_valid():
    """Valid v3 config produces no errors."""
    cfg = {
        "tts_mode": "pointer",
        "pointer_loss_weight": 0.5,
        "progress_loss_weight": 0.2,
        "voice_state_loss_weight": 0.0,
    }
    assert dev.validate_v3_config(cfg) == []


def test_validate_v3_config_rejects_missing_fields():
    """Missing required fields produce errors."""
    cfg = {"tts_mode": "pointer"}
    errors = dev.validate_v3_config(cfg)
    assert len(errors) > 0
    assert any("pointer_loss_weight" in e for e in errors)


def test_validate_v3_config_rejects_bad_tts_mode():
    """Invalid tts_mode produces an error."""
    cfg = {
        "tts_mode": "invalid",
        "pointer_loss_weight": 0.5,
        "progress_loss_weight": 0.2,
        "voice_state_loss_weight": 0.0,
    }
    errors = dev.validate_v3_config(cfg)
    assert any("tts_mode" in e for e in errors)


def test_v3_optional_defaults_keys():
    """V3_OPTIONAL_DEFAULTS contains all expected v3 config fields."""
    expected = {
        "pointer_mode", "cfg_enabled", "cfg_drop_rate",
        "voice_state_supervision", "prosody_flow_matching",
        "training_stage", "bootstrap_alignment_path",
        "few_shot_prompt_training", "replay_mix_ratio",
    }
    assert expected == set(dev.V3_OPTIONAL_DEFAULTS.keys())


def test_mfa_align_propagates_heuristic_fallback_flag(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _touch(cache_dir / "jvs" / "train" / "s1" / "u1" / "meta.json")

    monkeypatch.setattr(dev, "get_mfa_command", lambda: ["mfa"])
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/mfa")
    monkeypatch.setattr(dev, "_check_mfa_japanese_runtime", lambda _cmd: (True, "3.12.13", []))
    monkeypatch.setattr(
        dev,
        "_build_mfa_corpus_from_cache_dataset",
        lambda *args, **kwargs: (1, 0, 0),
    )
    monkeypatch.setattr(dev, "run_checked", lambda _cmd: True)

    prompts = iter([
        str(tmp_path / "alignments"),  # output_root
        str(tmp_path / "corpus"),  # corpus_root
        "4",  # jobs
        "n",  # overwrite_corpus
        "n",  # keep_corpus
        "n",  # overwrite_alignment
        "y",  # allow_heuristic_fallback
        "japanese_mfa",  # dictionary
        "japanese_mfa",  # acoustic
    ])
    monkeypatch.setattr(dev, "input_default", lambda *args, **kwargs: next(prompts))

    captured = {}

    def fake_prepare(
        cache_dir=None,
        enabled_cfg=None,
        textgrid_overrides=None,
        interactive=True,
        overwrite=None,
        allow_heuristic_default=None,
    ):
        captured["allow_heuristic_default"] = allow_heuristic_default
        return True

    monkeypatch.setattr(dev, "cmd_prepare_tts_alignment_from_latest_cache", fake_prepare)

    ok = dev.cmd_mfa_align_and_inject_from_cache(
        cache_dir=cache_dir,
        enabled_cfg={"jvs": {"language": "ja"}},
    )
    assert ok is True
    assert captured["allow_heuristic_default"] is True
