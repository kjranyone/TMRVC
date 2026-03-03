#!/bin/bash

echo "=== TMRVC Development Menu ==="
echo "1) キャッシュ削除 + フル学習"
echo "2) 既存キャッシュで学習のみ"
echo "3) データセット一覧"
echo "4) データセット追加 (対話式)"
echo "5) 話者分離実行"
echo "6) 設定ファイル初期化"
echo ""
read -p "選択 [1-6]: " choice

case $choice in
    1)
        echo "キャッシュを削除中..."
        rm -rf data/cache/
        echo "フル学習を開始..."
        tmrvc-train-pipeline \
            --dataset vctk \
            --raw-dir data/raw \
            --output-dir experiments \
            --workers 5 \
            --seed 42
        ;;
    2)
        echo "既存キャッシュで学習を開始..."
        tmrvc-train-pipeline \
            --dataset vctk \
            --raw-dir data/raw \
            --output-dir experiments \
            --skip-preprocess \
            --seed 42
        ;;
    3)
        echo "データセット一覧:"
        echo "参照: configs/datasets.yaml"
        echo ""
        python3 -c "
import yaml
with open('configs/datasets.yaml') as f:
    cfg = yaml.safe_load(f)
for name, ds in cfg.get('datasets', {}).items():
    status = '有効' if ds.get('enabled', False) else '無効'
    print(f'  - {name} [{status}] ({ds.get(\"language\", \"?\")}) {ds.get(\"raw_dir\", \"\")}')
"
        ;;
    4)
        python3 scripts/config_generator.py --add-dataset
        ;;
    5)
        echo "話者分離を実行"
        read -p "入力ディレクトリ (例: data/moe_multispeaker_voices): " input_dir
        read -p "デバイス [cuda/cpu]: " device
        device=${device:-cuda}
        python3 scripts/eval/cluster_speakers.py --input "$input_dir" --device "$device"
        ;;
    6)
        python3 scripts/config_generator.py --init
        ;;
    *)
        echo "無効な選択: $choice"
        exit 1
        ;;
esac
