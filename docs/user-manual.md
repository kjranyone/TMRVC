# TMRVC ユーザーマニュアル

TMRVC は、`UCLM v3` を用いて TTS / VC / curation / evaluation を扱うシステムである。mainline の人間運用は `tmrvc-gui` + `tmrvc-serve` を前提とし、`dev.py` と CLI は開発者向け補助経路とする。

## 1. 必要ファイル

- `checkpoints/uclm/uclm_latest.pt`
- `checkpoints/codec/codec_latest.pt`

エクスポート済み運用では、対応する ONNX 一式を使う。

## 2. 標準運用

### 2.1 WebUI 起動

1. `tmrvc-serve` を起動する
2. `tmrvc-gui` を起動する
3. ブラウザから Control Plane に接続する

```bash
uv run tmrvc-serve \
  --uclm-checkpoint checkpoints/uclm/uclm_latest.pt \
  --codec-checkpoint checkpoints/codec/codec_latest.pt

uv run tmrvc-gui
```

### 2.2 人間運用フロー

1. `Dataset Manager` で dataset を登録し、合法性 / provenance を設定する
2. `Curation Auditor` で transcript / speaker / language を修正し、promotion を監査する
3. `Dataset Manager` または export UI で promoted subset を materialize する
4. `Drama Workshop` で TTS を試聴し、`Casting Gallery` / `SpeakerProfile` を使って few-shot 話者を選ぶ
5. `Evaluation Arena` で blind A/B 評価を行う
6. `System Admin` / `Server Control` で runtime と export 状態を確認する

## 3. 運用原則

- dataset は単一言語を基本とし、code-switch は `language_spans` 付きで扱う
- mainline TTS は `MFA` と `durations.npy` を必要としない
- pacing は `pace`, `hold_bias`, `boundary_bias` で扱う
- few-shot 話者条件は `SpeakerProfile` / `speaker_profile_id` を正本とする
- multi-user 更新は canonical `metadata_version` による optimistic locking を前提とする

## 4. 開発者向け補助経路

CLI や `dev.py` は次の場合に使う。

- ローカル開発
- 学習パイプラインの反復
- legacy 比較実験
- CI / 自動化

学習フローの詳細は `TRAIN_GUIDE.md` を参照する。

## 5. トラブル時の確認

1. checkpoint が両方揃っているか
2. `phonemizer` / `pyopenjtalk` / `espeak-ng` が揃っているか
3. dataset language と `language_spans` が正しいか
4. quality gate と curation status が通っているか
5. 更新失敗時は `metadata_version` の競合が起きていないか

## 6. 参照文書

- `README.md`
- `TRAIN_GUIDE.md`
- `docs/design/architecture.md`
- `docs/design/gui-design.md`
- `docs/design/curation-contract.md`
