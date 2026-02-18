# Stream 5: VST3 Plugin (tmrvc-plugin)

## Goal

JUCE ベースの VST3 プラグインを実装し、DAW からリアルタイム Voice Conversion を実行可能にする。

**Blocked by:** Stream #4 (C++ Engine)

## Architecture

```
tmrvc-plugin (JUCE)
    │
    │ uses (header-only include)
    ▼
tmrvc-engine (JUCE-free C++ library)
    │
    │ links
    ▼
ONNX Runtime (static, CPU EP)
```

## Scope

### In Scope
- TMRVCProcessor (AudioProcessor): processBlock, prepareToPlay, state persistence
- TMRVCEditor (AudioProcessorEditor): minimal GUI
- VST3 + Standalone formats
- Parameters: Dry/Wet, Output Gain, IR Mode, IR params (manual)
- Speaker file loading (async)
- Latency compensation reporting

### Out of Scope (v1)
- AAX / AU formats
- Advanced GUI (waveform display, spectrum analyzer)
- Preset management
- Multi-channel (mono only, copy to stereo)

## Implementation Plan

### 5.1 Project Setup

```
tmrvc-plugin/
├── CMakeLists.txt
└── src/
    ├── TMRVCProcessor.h
    ├── TMRVCProcessor.cpp
    ├── TMRVCEditor.h
    └── TMRVCEditor.cpp
```

JUCE は CPM or git submodule で取得。

### 5.2 TMRVCProcessor

```cpp
class TMRVCProcessor : public juce::AudioProcessor {
    StreamingEngine engine_;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void processBlock(AudioBuffer<float>&, MidiBuffer&) override;
    void releaseResources() override;

    void getStateInformation(MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
};
```

Reference: `docs/design/cpp-engine-design.md` §8-9

### 5.3 TMRVCEditor (Minimal)

v1 は最小限の GUI:
- Speaker file selector (file browser)
- Model directory selector
- Dry/Wet slider
- Output Gain slider
- IR Mode toggle (Auto/Manual)
- Status text (loaded model, latency)

### 5.4 Parameters

| Parameter | ID | Range | Default |
|---|---|---|---|
| Dry/Wet | 0 | 0.0 - 1.0 | 1.0 |
| Output Gain | 1 | -60 - +12 dB | 0 dB |
| IR Mode | 2 | Auto / Manual | Auto |

### 5.5 Testing

- Standalone build でスモークテスト
- JUCE AudioPluginHost でテスト
- DAW (REAPER / Ableton) で動作確認

## Dependencies

- tmrvc-engine (Stream #4)
- JUCE 7+
- CMake >= 3.20

## Acceptance Criteria

- [ ] VST3 としてビルド成功
- [ ] Standalone で音声入出力が動作
- [ ] Speaker file をロードして声質変換が動作
- [ ] Dry/Wet, Gain パラメータが反映される
- [ ] DAW (REAPER) で latency compensation が正しく動作
- [ ] State persistence (DAW プロジェクト保存・復元)
