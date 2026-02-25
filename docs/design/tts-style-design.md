# TMRVC-TTS スタイル・文脈統合設計

Kojiro Tanaka — style, context, and VTuber integration design
Created: 2026-02-24 (Asia/Tokyo)

> **Goal:** 感情・演技スタイルの制御、会話コンテキストに応じた自動演技判断、
> VTuber ライブ配信での音声応答を実現する。

---

## 1. StyleEncoder 詳細設計

### 1.1 emotion_style[32d] ベクトル仕様

| Index | 次元 | 名称 | 値域 | 説明 |
|---|---|---|---|---|
| 0 | 1d | Valence | [-1, 1] | 快—不快 |
| 1 | 1d | Arousal | [-1, 1] | 覚醒—沈静 |
| 2 | 1d | Dominance | [-1, 1] | 支配—服従 |
| 3-5 | 3d | VAD uncertainty | [0, 1] | 各 VAD 次元の不確実性 |
| 6 | 1d | Speech rate | [-1, 1] | 遅い—速い |
| 7 | 1d | Energy | [-1, 1] | 静か—大声 |
| 8 | 1d | Pitch range | [-1, 1] | 単調—抑揚豊か |
| 9-20 | 12d | Emotion category | [0, 1] softmax | 12 感情カテゴリ |
| 21-28 | 8d | Learned latent | free | ラベル外のニュアンス |
| 29-31 | 3d | Reserved | 0 | 将来拡張用 |

### 1.2 12 感情カテゴリ

| ID | EN | JA | 典型的 VAD |
|---|---|---|---|
| 0 | happy | 喜び | V+, A+, D+ |
| 1 | sad | 悲しみ | V-, A-, D- |
| 2 | angry | 怒り | V-, A+, D+ |
| 3 | fearful | 恐怖 | V-, A+, D- |
| 4 | surprised | 驚き | V±, A+, D± |
| 5 | disgusted | 嫌悪 | V-, A±, D+ |
| 6 | neutral | 中立 | V0, A0, D0 |
| 7 | bored | 退屈 | V-, A-, D- |
| 8 | excited | 興奮 | V+, A+, D+ |
| 9 | tender | 優しさ | V+, A-, D± |
| 10 | sarcastic | 皮肉 | V-, A±, D+ |
| 11 | whisper | 囁き | V±, A-, D- |

### 1.3 AudioStyleEncoder 学習

**Phase 3a: 音声→スタイル**

```
入力: mel_ref[B, 80, T_ref]  (参照音声)
    → Conv2d x 4 (stride=2, BN, SiLU)
    → GlobalAvgPool(time) → Flatten
    → MLP → style[B, 32]

補助ヘッド:
    style → emotion_head → emotion_logits[12]  (cross-entropy)
    style → vad_head → vad[3]                  (MSE)
    style → prosody_head → prosody[3]           (MSE)
```

**損失:**
```
L_style = λ_cls * CE(emotion_logits, gt_emotion)
        + λ_vad * MSE(vad, gt_vad)
        + λ_prosody * MSE(prosody, gt_prosody)
```

**Phase 3b: テキスト→スタイル (将来)**

テキスト記述 ("怒りを抑えて冷たく") からスタイルベクトルを予測。
対照学習: テキスト埋め込みと音声スタイル埋め込みのコサイン類似度を最大化。

### 1.4 スタイルの結合

```python
# VC モード (legacy checkpoint support removed)
style_params = cat([acoustic_params[32], zeros[32]])  # [64]

# TTS モード
emotion_style = style_encoder(mel_ref)  # [32]
style_params = cat([acoustic_params[32], emotion_style[32]])  # [64]

# Converter への入力
cond = cat([spk_embed[192], style_params[64]])  # [256]
```

## 2. Cross-lingual Transfer 戦略

### 2.1 感情の言語横断性

感情の音響特徴は言語によらず共通する部分が大きい:

| 感情 | 言語非依存の音響特徴 |
|---|---|
| 怒り | 高ピッチ、高エネルギー、速い話速、硬い声質 |
| 悲しみ | 低ピッチ、低エネルギー、遅い話速、息混じり |
| 喜び | 高ピッチ、高エネルギー、ピッチ変動大 |
| 恐怖 | 高ピッチ、不安定、震え |
| 中立 | 中程度の全パラメータ |

### 2.2 Transfer 手順

```
Step 1: Expresso (40h EN, 26 スタイル)
    → StyleEncoder 事前学習
    → emotion_head + vad_head 学習

Step 2: JVNV (4h JA, 6 感情)
    → StyleEncoder fine-tune (lr=1e-5, 低学習率)
    → 日本語音響特徴への適応

Step 3: J-MAC / JTubeSpeech (擬似ラベル)
    → Step 1 の分類器で擬似ラベル付け
    → confidence > 0.8 のデータで追加学習

Step 4: ターゲット VTuber (5-10 分)
    → Few-shot fine-tune (LoRA)
```

## 3. Converter FiLM 拡張の詳細

### 3.1 重み移行の数学的保証

FiLM の変換:
```
gamma, beta = Linear(cond).chunk(2)
output = gamma * x + beta
```

`Linear(d_cond=256, d_out=768)` の重み行列 `W[768, 256]`:

```
W = [W_old[768, 224] | W_new[768, 32]]
```

旧VC checkpoint (d_cond=224) からの重み移行は行わない。ランタイムは style-conditioned converter (d_cond=256) 前提。

### 3.2 Fine-tune 時の学習対象

Phase 3 の Joint Fine-tune では FiLM 層のみを学習対象とする:

```python
# FiLM のみ学習可能に
for block in converter.blocks:
    block.conv_block.requires_grad_(False)  # ConvNeXt 凍結
    block.film.requires_grad_(True)         # FiLM のみ学習
converter.input_proj.requires_grad_(False)
converter.output_proj.requires_grad_(False)
```

## 4. LLM 文脈統合 (Phase 4)

### 4.1 ContextStylePredictor

```python
class ContextStylePredictor:
    """Claude API で会話コンテキストからスタイルを予測."""

    def predict(
        self,
        character: CharacterProfile,
        history: list[DialogueTurn],
        next_text: str,
        situation: str | None = None,
    ) -> StyleParams:
        """会話履歴と次の発話テキストからスタイルを推定.

        Args:
            character: キャラクタープロファイル
            history: 直近の会話履歴 (5-10 ターン)
            next_text: 次に発話するテキスト
            situation: 状況説明 (オプション)

        Returns:
            StyleParams (emotion_style[32] にマッピング)
        """
```

### 4.2 プロンプト設計

```
あなたは音声合成システムの感情コントローラーです。
以下のキャラクターと会話コンテキストを踏まえ、
次の発話に最適な感情パラメータを JSON で出力してください。

## キャラクター
名前: {character.name}
性格: {character.personality}
声の特徴: {character.voice_description}

## 状況
{situation}

## 会話履歴
{history の最新 5-10 ターン}

## 次の発話
「{next_text}」

## 出力形式 (JSON)
{
  "emotion": "happy" | "sad" | ... (12カテゴリ),
  "valence": -1.0 ~ 1.0,
  "arousal": -1.0 ~ 1.0,
  "dominance": -1.0 ~ 1.0,
  "speech_rate": -1.0 ~ 1.0,
  "energy": -1.0 ~ 1.0,
  "pitch_range": -1.0 ~ 1.0,
  "reasoning": "推論の根拠を簡潔に"
}
```

### 4.3 API 設計判断

| 判断 | **API ベース (Claude)** を選択 |
|---|---|
| **根拠** | ローカル LLM は B570 の 9.6 GB VRAM を圧迫。TTS は発話単位なので API レイテンシ (~1s) は許容範囲。日本語文脈理解の品質が高い。コスト ~$0.001/発話。 |

### 4.4 フォールバック

API 不可時のルールベース fallback:

```python
RULE_BASED_MAPPING = {
    "！": {"arousal": +0.3, "energy": +0.2},
    "？": {"pitch_range": +0.3},
    "…": {"arousal": -0.2, "speech_rate": -0.3},
    "笑": {"emotion": "happy", "valence": +0.5},
    "泣": {"emotion": "sad", "valence": -0.5},
}
```

### 4.5 型定義

```python
@dataclass
class CharacterProfile:
    name: str
    personality: str        # "明るく元気、たまにツンデレ"
    voice_description: str  # "高めの声、やや息混じり"
    default_style: StyleParams
    speaker_file: Path      # .tmrvc_speaker

@dataclass
class DialogueTurn:
    speaker: str
    text: str
    emotion: str | None     # 手動指定 or LLM 推定
    timestamp: float | None = None
```

配置: `tmrvc-core/src/tmrvc_core/dialogue_types.py`

### 4.6 台本フォーマット

```yaml
# 台本フォーマット例
title: "Scene 1 - 再会"
situation: "10年ぶりに駅で再会した幼馴染"
characters:
  sakura:
    profile: "明るく感情的な25歳女性"
    speaker_file: "models/sakura.tmrvc_speaker"
  yuki:
    profile: "落ち着いた控えめな26歳女性"
    speaker_file: "models/yuki.tmrvc_speaker"
dialogue:
  - speaker: sakura
    text: "ゆきちゃん！本当に久しぶり！"
    hint: "歓喜、涙ぐみ"
  - speaker: yuki
    text: "さくら...まさか来てくれるなんて。"
    hint: "驚き、感動"
```

- `hint` がない場合: ContextStylePredictor が会話履歴から自動推定
- `hint` がある場合: hint をスタイルベクトルに優先的に反映

## 5. VTuber 統合 (Phase 5)

### 5.1 `.tmrvc_character` フォーマット

`.tmrvc_speaker` の拡張:

```
Offset  Size     Content
0       4        Magic: "TMCH" (0x544D4348)
4       4        Version: 1
8       192*4    spk_embed[192] (float32)
776     N*4      lora_delta[lora_delta_size] (float32)
776+N*4 8*4      voice_source_preset[8] (float32)
+32     32*4     default_style_params[32] (float32)
+128    4        character_json_len (uint32)
+132    M        character_profile (JSON, UTF-8)
+M      32       SHA256 checksum
```

`character_profile` JSON:
```json
{
  "name": "桜",
  "personality": "明るく元気、たまにツンデレ",
  "voice_description": "高めの声、やや息混じり",
  "language": "ja",
  "default_emotion": "neutral",
  "greeting": "やっほー！今日も元気？"
}
```

### 5.2 ライブ配信チャットレスポンス

```
YouTube/Twitch Chat API
    ↓
[Chat Stream Adapter] → コメント受信
    ↓
[Comment Selector] → 応答すべきコメントを選択
    ↓                  (スパチャ優先、連投除外)
[Response Generator] → レスポンステキスト生成
    ↓                    ↓ (非同期並行)
[Context Builder]    [Style Predictor (Claude API)]
  会話履歴蓄積          スタイル推定 (~0.5-1s)
    ↓                    ↓
[TTS Pipeline] → ストリーミング音声生成
    ↓
[Virtual Audio Device] → 配信ソフト (OBS 等)
```

### 5.3 レイテンシ分析

| ステップ | 処理時間 | 備考 |
|---|---|---|
| コメント受信 | ~0ms | YouTube/Twitch API |
| レスポンステキスト | 0ms or ~1-2s | 手動入力 or AI 生成 |
| Style 予測 (Claude) | ~0.5-1s | 非同期、テキスト入力と並行可能 |
| TTS フロントエンド | ~30ms | G2P + TextEncoder + Duration + F0 + ContentSynth |
| Converter + Vocoder 初期フレーム | ~100ms | ストリーミング再生開始 |
| **体感レイテンシ** | **~0.5-1.5s** | テキスト確定から音声再生開始まで |

### 5.4 ストリーミング TTS 出力

VC の Converter + Vocoder はフレーム単位 (10ms) ストリーミング対応済み。
TTS フロントエンドが最初の数十フレームを生成した時点で再生を開始:

```
[ContentSynth T=0~50] → [Converter T=0] → [Vocoder T=0] → 再生開始
                        → [Converter T=1] → [Vocoder T=1]
[ContentSynth T=50~]  → ...                               → 再生継続
```

### 5.5 優先度キュー

```python
class CommentQueue:
    """優先度付きコメントキュー."""

    PRIORITY_SUPERCHAT = 0      # 最高優先
    PRIORITY_MEMBER = 1
    PRIORITY_GENERAL = 2

    max_queue_size: int = 5     # 超過分は古いものから破棄

    def push(self, comment: Comment, priority: int) -> None: ...
    def pop(self) -> Comment | None: ...
    def interrupt(self, comment: Comment) -> None:
        """高優先コメント到着時に現在の再生を中断 (crossfade)."""
```

### 5.6 応答テキスト生成の 2 モード

| モード | 入力 | レイテンシ | 用途 |
|---|---|---|---|
| **手動** | VTuber がテキスト入力 | 0ms (入力次第) | 精密な応答が必要な場面 |
| **AI** | Claude API が自動生成 | ~1-2s | カジュアルなチャット応答 |

## 6. FastAPI サーバー設計 (Phase 5)

### 6.1 エンドポイント

```python
# REST API
POST /tts          → 音声生成 (一括)
POST /tts/stream   → ストリーミング音声生成

# WebSocket
WS   /ws/chat      → チャットストリーム双方向通信

# 管理
GET  /characters   → キャラクター一覧
POST /characters   → キャラクター登録
GET  /health       → ヘルスチェック
```

### 6.2 リクエスト/レスポンス

```python
# POST /tts
class TTSRequest(BaseModel):
    text: str
    character_id: str
    emotion: str | None = None
    context: list[DialogueTurn] | None = None
    speed: float = 1.0

class TTSResponse(BaseModel):
    audio: bytes          # WAV (24kHz, float32)
    duration_sec: float
    style_used: dict      # 実際に使用されたスタイルパラメータ
```

### 6.3 WebSocket プロトコル

```json
// Client → Server: コメント送信
{
  "type": "comment",
  "text": "こんにちは！",
  "user": "viewer123",
  "priority": 2
}

// Client → Server: 応答テキスト指定 (手動モード)
{
  "type": "response",
  "text": "こんにちは！元気？",
  "character_id": "sakura"
}

// Server → Client: 音声チャンク (ストリーミング)
{
  "type": "audio_chunk",
  "data": "<base64 encoded float32 PCM>",
  "frame_index": 0,
  "is_last": false
}

// Server → Client: スタイル情報
{
  "type": "style_info",
  "emotion": "happy",
  "vad": [0.7, 0.5, 0.3],
  "reasoning": "挨拶に対する明るい応答"
}
```

### 6.4 GUI 拡張

既存の `tmrvc-gui` に追加するページ:

| ページ | 機能 |
|---|---|
| **TTSPage** | テキスト入力、キャラクター選択、スタイルスライダー、プレビュー |
| **ScriptPage** | 台本エディタ、タイムライン、バッチ生成 |
| **ChatMonitorPage** | YouTube/Twitch チャット表示、応答キュー管理、ライブ TTS |

## 7. 技術判断の根拠

| 判断項目 | 選択 | 理由 |
|---|---|---|
| テキスト表現 | **音素ベース** (IPA) | 漢字の多義性、日英共通化、TTS で標準的 |
| デュレーション | **明示的** (FastSpeech2 方式) | VTuber・演技でタイミング制御が必須 |
| スタイル表現 | **連続値** (VAD + カテゴリ + 潜在) | "皮肉混じりの優しさ" など粒度の細かい制御 |
| LLM 統合 | **API ベース** (Claude) | VRAM 節約、日本語品質、発話単位でレイテンシ許容 |
| リアルタイム性 | **TTS はオフライン** (発話単位) | テキスト全体が必要。VC リアルタイムパスは維持 |
| 音素体系 | **統一 IPA** (~200) | 日英共通、将来の多言語拡張が容易 |
| サーバー | **FastAPI + WebSocket** | 非同期処理、ストリーミング対応、Python エコシステム |

## 8. 整合性チェックリスト

- [x] `emotion_style[32d]` の合計 (3+3+3+12+8+3=32) が `d_style=32` と一致 (`constants.yaml`)
- [x] 12 感情カテゴリ数が `n_emotion_categories=12` と一致 (`constants.yaml`)
- [x] StyleEncoder 出力 `[B, 32]` が FiLM 入力の追加次元と一致 (`tts-architecture.md §7`)
- [x] `style_params[64]` = `acoustic_params[32]` + `emotion_style[32]` (`tts-architecture.md §8`)
- [x] ContextStylePredictor の出力が emotion_style ベクトルにマッピング可能
- [x] `.tmrvc_character` フォーマットが `.tmrvc_speaker` (`onnx-contract.md §6`) とは別仕様
- [x] ストリーミング TTS は既存 Converter/Vocoder のフレーム単位処理を利用 (`streaming-design.md §4`)
- [x] WebSocket プロトコルが VC リアルタイムエンジンと独立 (別ポート/プロセス)
- [x] Claude API コスト ~$0.001/発話は VTuber 配信で実用的 (1000 発話/配信 → ~$1)

---

**関連資料:**

- `docs/design/tts-architecture.md` — TTS モジュール構成・テンソル仕様
- `docs/design/tts-training-plan.md` — TTS 学習ロードマップ
- `docs/design/acoustic-condition-pathway.md` — IR→Acoustic Pathway (style_params の前半 32d)
- `docs/design/onnx-contract.md` — `.tmrvc_speaker` フォーマット
- `docs/design/streaming-design.md` — ストリーミングパイプライン
- `docs/design/gui-design.md` — GUI 設計
