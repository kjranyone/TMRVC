"""Configuration file manager: load/save GUI and training settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG: dict[str, Any] = {
    "project_root": ".",
    "audio": {
        "input_device": None,
        "output_device": None,
        "buffer_size": 512,
    },
    "realtime": {
        "onnx_dir": "",
        "speaker_file": "",
        "dry_wet": 1.0,
        "output_gain_db": 0.0,
    },
    "training": {
        "phase": "phase0",
        "datasets": ["VCTK", "JVS"],
        "batch_size": 16,
        "lr": 1e-4,
        "total_steps": 50000,
        "mode": "local",
        "ssh_host": "",
        "ssh_user": "",
        "ssh_key": "",
        "ssh_remote_dir": "",
    },
    "data_prep": {
        "n_workers": 4,
        "steps": ["resample", "normalize", "vad_trim", "segment", "features"],
    },
}

_CONFIG_FILENAME = "tmrvc_gui_config.json"


class ConfigManager:
    """Load and save GUI configuration as JSON."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self._config_dir = config_dir or Path.cwd()
        self._config_path = self._config_dir / _CONFIG_FILENAME
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load config from disk, falling back to defaults."""
        if self._config_path.exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        # Merge with defaults (shallow per top-level key)
        for key, default_val in _DEFAULT_CONFIG.items():
            if key not in self._data:
                self._data[key] = default_val
            elif isinstance(default_val, dict) and isinstance(self._data[key], dict):
                for k, v in default_val.items():
                    self._data[key].setdefault(k, v)

    def save(self) -> None:
        """Persist config to disk."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a config value by section and key."""
        return self._data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a config value by section and key."""
        if section not in self._data:
            self._data[section] = {}
        self._data[section][key] = value

    @property
    def data(self) -> dict[str, Any]:
        """Return the full config dict (read-only view)."""
        return self._data
