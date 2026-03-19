# Decision: Model Scale — A1 (248M)

## Date: 2026-03-19

## Decision

d_model=768, n_layers=16, ~248M params (CosyVoice 2級)

## Rationale

- 99M (d_model=512, n_layers=12) では v4 全機能を高品質に達成できない
  - 8×1024 multi-codebook 予測に対して backbone 53M は不足
  - 14,976サンプル × 10,000ステップで loss_a = 2.41 止まり、音声として不成立
  - CB0 予測精度が低い（確信的に間違ったトークンを出す）
- RTX 2080 Ti 22GB に収まる最大サイズ
  - B (598M) は VRAM オーバー
- CosyVoice 2 (300M) と同スケールで音声品質の実績がある帯域
- single-codebook 検討はスキップ（codec 変更より先にモデル規模を上げて現行アーキテクチャの限界を見る）

## Constants Changes

```yaml
# Before (99M)
d_model: 512
uclm_n_heads: 8
uclm_n_layers: 12

# After (248M)
d_model: 768
uclm_n_heads: 12
uclm_n_layers: 16
n_kv_heads: 3
```

head_dim = 768 / 12 = 64 (維持)
n_kv_heads = 12 / 4 = 3 (GQA 4:1 ratio 維持)

## Impact

- チェックポイント: ~360MB → ~1GB
- VRAM (学習, batch=4): ~4GB → ~12GB
- VRAM (推論): ~1GB → ~3GB
- 学習速度: ~1.5s/step → ~4-5s/step (推定)

## Not Changed

- n_codebooks: 8 (EnCodec 維持)
- rvq_vocab_size: 1024
- codec: EnCodec 24kHz
- 12-D physical / 24-D latent / inline tags / pointer — 全維持
