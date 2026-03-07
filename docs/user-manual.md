# TMRVC ユーザーマニュアル

TMRVC は、学習済み `UCLM + codec` を用いて TTS と VC を提供するリアルタイム音声生成システムである。現行の主運用面は `tmrvc-serve` と `dev.py` に集約する。

## 1. 必要ファイル

- `checkpoints/uclm/uclm_latest.pt`
- `checkpoints/codec/codec_latest.pt`

エクスポート済み運用では、対応する ONNX 一式を使う。

## 2. 基本操作

### 学習

```bash
uv run python dev.py
```

推奨順:

1. `6` 設定初期化
2. `4` データセット追加
3. `1` フル学習
4. `8` Codec 学習
5. `7` 成果物確定

### 推論サーバー

```bash
uv run tmrvc-serve \
  --uclm-checkpoint checkpoints/uclm/uclm_latest.pt \
  --codec-checkpoint checkpoints/codec/codec_latest.pt
```

## 3. 運用原則

- dataset は単一言語で登録する
- mainline TTS は `MFA` を必要としない
- pacing は pointer / prosody 制御で扱う

## 4. トラブル時の確認

1. checkpoint が両方揃っているか
2. `phonemizer` / `pyopenjtalk` / `espeak-ng` が揃っているか
3. dataset language が正しいか
4. quality gate が通っているか

## 5. 参照文書

- `README.md`
- `TRAIN_GUIDE.md`
- `docs/design/architecture.md`
- `docs/design/streaming-design.md`
