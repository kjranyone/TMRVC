# Programmable Expressive Speech Contract

この文書は、TMRVC が Fish Audio S2 等の「プロンプト依存 TTS」に対して戦略的優位性を確立するための、**決定論的な音声制御および編集**に関する技術契約を定義する。

## 1. 核心概念

TMRVC は「プロンプトからの生成」を単なる探索フェーズとし、**「軌道（Trajectory）の固定と編集」**を真の製品価値とする。

### A. Intent Compiler (意図の固定)
自然言語のプロンプトやタグは、直接音響モデルに流し込まれるのではなく、一度 `IntentCompilerOutput` に変換（コンパイル）される。
- **不変性**: 同一のプロンプトは、同一の物理制御量（Pace, Hold Bias, Voice State 等）に変換されなければならない。
- **可視性**: ユーザーはコンパイルされた制御量を生成前に確認・編集できる。

### B. Trajectory Record (実績の固定)
生成された音声の「演技」は `TrajectoryRecord` として永続化される。
- **Deterministic Replay**: 記録されたトークン列と `voice_state` を用いることで、モデルのサンプリングによらずビット精度の再現を保証する。
- **Edit Locality**: 軌道の特定のフレーム区間だけを、周囲の文脈（KV Cache）を保ったまま部分的に書き換えることができる。

## 2. 物理スキーマ (tmrvc-core)

### IntentCompilerOutput
```python
@dataclass
class IntentCompilerOutput:
    compile_id: str
    pacing: PacingControls  # pace, hold_bias, boundary_bias
    explicit_voice_state: torch.Tensor # [1, 12] 全体的な感情指定
    local_prosody_plan: dict[int, torch.Tensor] # 音素ごとの個別制御
```

### TrajectoryRecord
```python
@dataclass
class TrajectoryRecord:
    trajectory_id: str
    phoneme_ids: torch.Tensor # 生成時の音素列 (G2P変動を遮断)
    pointer_trace: List[Tuple[int, int]] # [text_index, frames_spent] の全履歴
    acoustic_trace: torch.Tensor # RVQ Stream A [8, T]
    control_trace: torch.Tensor  # Stream B [4, T]
    voice_state_trajectory: torch.Tensor # 実現された 12-D パラメータ [T, 12]
```

## 3. 検証ゲート (Worker 06)

本契約の遵守は、以下の指標によって物理的に強制される。

1. **Replay Fidelity Score**: 
   - `replay(TrajectoryRecord)` の出力が、初回生成時の音声と 100% 同一（ビット一致）であることを保証。
2. **Edit Locality Score**:
   - 軌道の $t_{start}$ から $t_{end}$ をパッチした際、区間外の音声波形および制御信号に 1e-5 以上の変化がないことを保証。

## 4. 戦略的マイルストーン

- **v3.0**: `IntentCompiler` によるタグベースの制御と `TrajectoryRecord` の永続化・リプレイをサポート。
- **v3.1**: UI 上でのタイムライン編集（軌道のドラッグ＆ドロップ）をサポート。
