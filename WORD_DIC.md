# TMRVC 用語集 (WORD_DIC.md)

TMRVC プロジェクトで使用される主要な用語、概念、およびアーキテクチャ・コンポーネントの解説です。

## 1. コア・コンセプト (Core Concepts)

| 用語 | 解説 |
|---|---|
| **TMRVC** | Unified Codec Language Model (UCLM) v3 を核とした、リアルタイム TTS (音声合成) および高精度 VC (音声変換) を実現する統合音声生成システム。 |
| **UCLM (Unified Codec Language Model)** | TTS と VC を単一のトランスフォーマー・アーキテクチャで統合した言語モデル (v3)。ポインタベースのテキスト進行機構を持ち、MFA 非依存での学習が可能。 |
| **TTS Mode (Text-to-Speech)** | テキスト（音素）を入力として、ターゲット話者の音声を生成するタスク。 |
| **VC Mode (Voice Conversion)** | ソース音声を入力として、その内容や抑揚を維持しつつターゲット話者の声質に変換するタスク。 |
| **Real-time Streaming** | 10ms フレーム単位の因果的推論により、極低遅延（理論上 10ms 〜 実効 25ms 程度）で音声を生成する仕組み。 |

## 2. アーキテクチャ & モデル (Architecture & Models)

| 用語 | 解説 |
|---|---|
| **UCLM (v3)** | TMRVC のメインエンジンとなるトランスフォーマー・モデル。音響トークン (`A_t`) と制御トークン (`B_t`) を同時に予測するデュアルヘッド構成に加え、ポインタベースのアライメント機構を持つ。 |
| **Emotion-Aware Codec** | 感情表現や物理的な声の状態を保持したまま、音声をトークン化および復元するための高品質なニューラル・オーディオ・コーデック。 |
| **Dual-Stream Token Spec** | 音響情報（音色・フォルマント）を担う `A_t` と、非言語イベント（呼吸、強弱、タイミング）を担う `B_t` の2系統のトークンを並行して扱う仕様。 |
| **TextEncoder** | TTS モードにおいて、音素 ID 列を UCLM が解釈可能な連続的な特徴量（潜在表現）に変換するモジュール。 |
| **VCEncoder** | VC モードにおいて、ソース音声を VQ Bottleneck を通じて抽象化し、話者性に依存しない内容情報を抽出するモジュール。 |
| **VoiceStateEncoder** | 8次元の物理パラメータと SSL (WavLM) 潜在空間を統合し、発話の表情を制御するための特徴量を生成するモジュール。 |
| **Codec Decoder** | UCLM が生成したトークン列から、高品質な 24kHz 波形を復元するデコーダ。 |

## 3. トークン & パラメータ (Tokens & Parameters)

| 用語 | 解説 |
|---|---|
| **A_t (Acoustic Token)** | 音響ストリーム。RVQ (Residual Vector Quantization) によって量子化されたトークン。音色や質感の情報を保持する。 |
| **B_t (Control Token)** | 制御ストリーム。呼吸、音の強弱、持続時間、インテンシティなどの非言語的なイベント情報を表現するトークン。 |
| **8-dim Physical Voice State** | 音声を物理的な側面から制御するための 8 つの次元。以下のパラメータで構成される：<br>1. **Breathiness** (息漏れ)<br>2. **Tension** (緊張度)<br>3. **Arousal** (覚醒度・エネルギー)<br>4. **Valence** (感情の正負)<br>5. **Roughness** (掠れ・ざらつき)<br>6. **Voicing** (有声性の強さ)<br>7. **Energy** (全体的な音量感)<br>8. **Speech Rate** (発話速度) |
| **Speaker Embedding** | 話者の特徴を数値化したベクトル。ターゲット話者の声を再現するために使用される。 |

## 4. 機能 & 技術 (Features & Techniques)

| 用語 | 解説 |
|---|---|
| **LoRA (Low-Rank Adaptation)** | 少量の参照音声から、特定のターゲット話者の声質へ高速かつ効率的にモデルを適応させる技術。 |
| **RVQ (Residual Vector Quantization)** | 情報を段階的に量子化する手法。オーディオ・コーデックにおいて、高圧縮かつ高品質なトークン表現を可能にする。 |
| **GRL (Gradient Reversal Layer)** | 敵対的学習において、特定の情報（例：話者情報）を意図的に排除するために使用される層。 |
| **CFG (Classifier-Free Guidance)** | 生成時に条件付けの強度を調整する手法。生成される音声の品質や忠実度を向上させるために使用される。 |
| **KV Cache (Key-Value Cache)** | トランスフォーマーの推論を高速化するためのキャッシュ機構。過去の計算結果を保持することで、毎ステップの計算量を削減する。 |
| **G2P (Grapheme-to-Phoneme)** | 書き言葉（書記素）を読み（音素）に変換する処理。TTS の前処理として行われる。 |
| **10ms Hop / Frame** | 処理の最小単位。24kHz サンプリングにおいて 240 サンプルに相当する。 |
| **RT-safe (Real-time safe)** | オーディオスレッドにおいて、メモリ確保 (malloc) やロック (mutex) などのブロッキング操作を行わず、安定したリアルタイム処理を保証すること。 |

## 5. 予測器 & 補助モジュール (Predictors & Aux Modules)

| 用語 | 解説 |
|---|---|
| **ContextStylePredictor** | LLM などから得られる文脈情報に基づいて、最適な 8次元物理パラメータを予測するモジュール。 |
| **PointerHead** | TTS モードにおいて、テキストトークン列上の読み上げ位置を自律的に追跡・進行させるモジュール。DurationPredictor に代わる v3 のコア。 |
| **WavLM** | 音声の汎用的な特徴を抽出するために使用される自己教師あり学習 (SSL) モデル。 |
