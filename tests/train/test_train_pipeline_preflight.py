from __future__ import annotations

from pathlib import Path

import tmrvc_train.cli.train_pipeline as train_pipeline


def _plan(name: str, raw_dir: Path, config: dict | None = None) -> train_pipeline.DatasetPlan:
    return train_pipeline.DatasetPlan(name=name, raw_dir=raw_dir, config=config or {})


def test_find_datasets_missing_raw_reports_only_missing(monkeypatch, tmp_path: Path):
    def fake_has_train_utterances(
        dataset: str, raw_dir: Path, dataset_config: dict
    ) -> bool:
        return dataset != "jvs"

    monkeypatch.setattr(
        train_pipeline, "_dataset_has_train_utterances", fake_has_train_utterances
    )

    missing = train_pipeline._find_datasets_missing_raw(
        plans=[_plan("jvs", tmp_path), _plan("vctk", tmp_path)]
    )
    assert missing == ["jvs"]


def test_find_datasets_missing_raw_uses_dataset_specific_config(
    monkeypatch, tmp_path: Path
):
    seen: list[tuple[str, dict]] = []

    def fake_has_train_utterances(
        dataset: str, raw_dir: Path, dataset_config: dict
    ) -> bool:
        seen.append((dataset, dict(dataset_config)))
        return True

    monkeypatch.setattr(
        train_pipeline, "_dataset_has_train_utterances", fake_has_train_utterances
    )

    plans = train_pipeline._build_dataset_plans(
        datasets_to_use=["jvs"],
        fallback_raw_dir=tmp_path,
        base_config={"language": "ja", "adapter_type": "base"},
        registry={"datasets": {"jvs": {"language": "en", "speaker_map": "map.json"}}},
        train_batch_size=None,
        train_device=None,
    )
    _ = train_pipeline._find_datasets_missing_raw(plans)

    assert seen[0][0] == "jvs"
    assert seen[0][1]["language"] == "en"
    assert seen[0][1]["adapter_type"] == "base"
    assert seen[0][1]["speaker_map"] == "map.json"


def test_build_dataset_plans_uses_dataset_raw_dir_over_fallback(tmp_path: Path):
    plans = train_pipeline._build_dataset_plans(
        datasets_to_use=["jvs"],
        fallback_raw_dir=tmp_path / "fallback",
        base_config={"language": "ja"},
        registry={"datasets": {"jvs": {"raw_dir": str(tmp_path / "jvs_raw"), "type": "jvs"}}},
        train_batch_size=8,
        train_device="cuda:0",
    )

    assert plans[0].raw_dir == tmp_path / "jvs_raw"
    assert plans[0].config["adapter_type"] == "jvs"
    assert plans[0].config["train_batch_size"] == 8
    assert plans[0].config["train_device"] == "cuda:0"


def test_build_dataset_plans_requires_any_raw_dir(tmp_path: Path):
    try:
        train_pipeline._build_dataset_plans(
            datasets_to_use=["jvs"],
            fallback_raw_dir=None,
            base_config={},
            registry={"datasets": {"jvs": {"type": "jvs"}}},
            train_batch_size=None,
            train_device=None,
        )
    except ValueError as e:
        assert "raw_dir not configured for dataset=jvs" in str(e)
        return
    assert False, "Expected ValueError for missing raw_dir"
