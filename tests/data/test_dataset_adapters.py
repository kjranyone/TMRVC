"""Tests for tmrvc_data.dataset_adapters module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tmrvc_data.dataset_adapters import (
    ADAPTERS,
    GenericAdapter,
    JVSAdapter,
    LibriTTSRAdapter,
    TsukuyomiAdapter,
    VCTKAdapter,
    get_adapter,
)


class TestGetAdapter:
    def test_known_adapters(self):
        for name in ["vctk", "jvs", "libritts_r", "tsukuyomi"]:
            adapter = get_adapter(name)
            assert adapter.name == name

    def test_unknown_adapter_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_adapter("nonexistent")

    def test_adapter_registry(self):
        assert set(ADAPTERS.keys()) == {"vctk", "jvs", "libritts_r", "tsukuyomi", "generic"}


class TestTsukuyomiAdapter:
    def test_sanitize(self):
        assert TsukuyomiAdapter._sanitize("hello world!") == "hello_world"
        assert TsukuyomiAdapter._sanitize("") == "unknown"
        assert TsukuyomiAdapter._sanitize("abc-123_def") == "abc-123_def"

    def test_iter_single_speaker(self, tmp_path):
        """Test single-speaker layout: root/*.wav"""
        import soundfile as sf
        import numpy as np

        # Create test audio files
        for name in ["utt001.wav", "utt002.wav"]:
            sf.write(
                str(tmp_path / name),
                np.zeros(2400, dtype=np.float32),
                24000,
            )

        adapter = TsukuyomiAdapter()
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 2
        assert all(u.dataset == "tsukuyomi" for u in utts)
        assert all(u.speaker_id.startswith("tsukuyomi_") for u in utts)
        assert all(u.language == "ja" for u in utts)

    def test_iter_multi_folder(self, tmp_path):
        """Test multi-folder layout: root/subfolder/*.wav"""
        import soundfile as sf
        import numpy as np

        sub = tmp_path / "speaker_A"
        sub.mkdir()
        sf.write(str(sub / "line01.wav"), np.zeros(2400, dtype=np.float32), 24000)

        adapter = TsukuyomiAdapter()
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 1
        assert "speaker_A" in utts[0].speaker_id

    def test_empty_dir_raises(self, tmp_path):
        adapter = TsukuyomiAdapter()
        with pytest.raises(FileNotFoundError, match="No audio files"):
            list(adapter.iter_utterances(tmp_path))

    def test_nonexistent_dir_raises(self, tmp_path):
        adapter = TsukuyomiAdapter()
        with pytest.raises(FileNotFoundError, match="not found"):
            list(adapter.iter_utterances(tmp_path / "nope"))


class TestVCTKAdapter:
    def test_missing_wav_dir_raises(self, tmp_path):
        adapter = VCTKAdapter()
        with pytest.raises(FileNotFoundError, match="wav directory not found"):
            list(adapter.iter_utterances(tmp_path))

    def test_092_layout(self, tmp_path):
        """Test VCTK 0.92 layout with flac files."""
        import soundfile as sf
        import numpy as np

        wav_dir = tmp_path / "wav48_silence_trimmed" / "p225"
        wav_dir.mkdir(parents=True)
        sf.write(
            str(wav_dir / "p225_001_mic1.flac"),
            np.zeros(4800, dtype=np.float32),
            48000,
        )

        adapter = VCTKAdapter()
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 1
        assert utts[0].dataset == "vctk"
        assert utts[0].speaker_id == "vctk_p225"
        assert utts[0].language == "en"

    def test_old_layout(self, tmp_path):
        """Test older VCTK layout with wav files."""
        import soundfile as sf
        import numpy as np

        wav_dir = tmp_path / "wav48" / "p226"
        wav_dir.mkdir(parents=True)
        sf.write(
            str(wav_dir / "p226_002.wav"),
            np.zeros(4800, dtype=np.float32),
            48000,
        )

        adapter = VCTKAdapter()
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 1
        assert utts[0].utterance_id == "vctk_p226_002"


class TestJVSAdapter:
    def test_parallel100_layout(self, tmp_path):
        """Test JVS parallel100 layout."""
        import soundfile as sf
        import numpy as np

        wav_dir = tmp_path / "jvs001" / "parallel100" / "wav24kHz16bit"
        wav_dir.mkdir(parents=True)
        sf.write(
            str(wav_dir / "VOICEACTRESS100_001.wav"),
            np.zeros(2400, dtype=np.float32),
            24000,
        )

        adapter = JVSAdapter()
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 1
        assert utts[0].dataset == "jvs"
        assert "jvs001" in utts[0].speaker_id

    def test_skips_non_jvs_dirs(self, tmp_path):
        """Non-jvs directories should be skipped."""
        (tmp_path / "readme.txt").touch()
        (tmp_path / "other_dir").mkdir()

        adapter = JVSAdapter()
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 0


class TestLibriTTSRAdapter:
    def test_train_split_layout(self, tmp_path):
        """Test LibriTTS-R train-clean-100 layout."""
        import soundfile as sf
        import numpy as np

        wav_dir = tmp_path / "train-clean-100" / "19" / "198"
        wav_dir.mkdir(parents=True)
        sf.write(
            str(wav_dir / "19_198_000000_000000.wav"),
            np.zeros(2400, dtype=np.float32),
            24000,
        )

        adapter = LibriTTSRAdapter()
        utts = list(adapter.iter_utterances(tmp_path, split="train"))
        assert len(utts) == 1
        assert utts[0].dataset == "libritts_r"
        assert utts[0].speaker_id == "libritts_19"

    def test_missing_split_dirs(self, tmp_path):
        """Missing split directories should log warning, not error."""
        adapter = LibriTTSRAdapter()
        utts = list(adapter.iter_utterances(tmp_path, split="train"))
        assert len(utts) == 0


class TestGenericAdapter:
    def test_flat_layout(self, tmp_path):
        """Test flat layout: root/*.wav with no speaker folders."""
        import json
        import numpy as np
        import soundfile as sf

        for name in ["utt001.wav", "utt002.wav", "utt003.wav"]:
            sf.write(str(tmp_path / name), np.zeros(2400, dtype=np.float32), 24000)

        adapter = GenericAdapter(dataset_name="mydata", language="ja")
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 3
        assert all(u.dataset == "mydata" for u in utts)
        assert all(u.language == "ja" for u in utts)

    def test_speaker_map(self, tmp_path):
        """Test GenericAdapter with speaker_map.json."""
        import json
        import numpy as np
        import soundfile as sf

        # Create flat audio files
        for name in ["a01.wav", "a02.wav", "b01.wav", "noise01.wav"]:
            sf.write(str(tmp_path / name), np.zeros(2400, dtype=np.float32), 24000)

        # Create speaker map
        speaker_map = {
            "version": 1,
            "method": "hdbscan",
            "n_speakers": 2,
            "n_noise": 1,
            "mapping": {
                "a01.wav": "spk_0001",
                "a02.wav": "spk_0001",
                "b01.wav": "spk_0002",
                "noise01.wav": "spk_noise",
            },
        }
        map_path = tmp_path / "_speaker_map.json"
        map_path.write_text(json.dumps(speaker_map), encoding="utf-8")

        adapter = GenericAdapter(
            dataset_name="test", language="ja", speaker_map_path=map_path,
        )
        utts = list(adapter.iter_utterances(tmp_path))

        # noise01.wav should be excluded
        assert len(utts) == 3
        names = {u.utterance_id for u in utts}
        assert any("a01" in n for n in names)
        assert any("b01" in n for n in names)
        assert not any("noise01" in n for n in names)

        # Check speaker ids are properly mapped
        spk_ids = {u.utterance_id: u.speaker_id for u in utts}
        a01_utt = [u for u in utts if "a01" in u.utterance_id][0]
        a02_utt = [u for u in utts if "a02" in u.utterance_id][0]
        b01_utt = [u for u in utts if "b01" in u.utterance_id][0]
        assert a01_utt.speaker_id == a02_utt.speaker_id  # same speaker
        assert a01_utt.speaker_id != b01_utt.speaker_id  # different speaker

    def test_speaker_map_via_get_adapter(self, tmp_path):
        """Test that get_adapter passes speaker_map_path to GenericAdapter."""
        import json
        import numpy as np
        import soundfile as sf

        sf.write(str(tmp_path / "x.wav"), np.zeros(2400, dtype=np.float32), 24000)

        speaker_map = {
            "version": 1, "method": "hdbscan", "n_speakers": 1, "n_noise": 0,
            "mapping": {"x.wav": "spk_0001"},
        }
        map_path = tmp_path / "_speaker_map.json"
        map_path.write_text(json.dumps(speaker_map), encoding="utf-8")

        adapter = get_adapter(
            "myds", adapter_type="generic", language="ja",
            speaker_map_path=map_path,
        )
        assert isinstance(adapter, GenericAdapter)
        assert adapter._speaker_map is not None
        utts = list(adapter.iter_utterances(tmp_path))
        assert len(utts) == 1
        assert "spk_0001" in utts[0].speaker_id
