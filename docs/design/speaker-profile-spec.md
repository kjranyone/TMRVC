# Speaker Profile Specification (The Casting Gallery Contract)

## Purpose

Define the `SpeakerProfile` as a first-class, serializable contract owned by `tmrvc-core`. This enables a single "Actor" or "Character" profile to be shared across the WebUI, Python Serve, Rust Engine, and VST plugins.

## 1. Ownership & Storage

- **Schema Owner:** `tmrvc-core` (Python/Rust types).
- **Persistence Owner:** backend-managed storage consumed via `tmrvc-serve`; the UI is not a schema owner.
- **Primary Key:** `speaker_profile_id` (A unique identifier for the actor).

## 2. Profile Schema (tmrvc-core)

A `SpeakerProfile` must contain:
- `schema_version`: Integer.
- `speaker_profile_id`: String (UUID or human-readable unique slug).
- `reference_audio_hash`: String (SHA-256 of the source audio file).
- `speaker_embed`: `[d_speaker]` float tensor (Timbre anchor).
- `prompt_codec_tokens`: `[T_prompt, n_codebooks]` int tensor (Acoustic context for KV cache).
- optional `prompt_kv_cache`: precomputed cache blob or external blob reference.
- `prompt_text_tokens`: `[L_prompt]` int tensor (Optional; text of the reference clip).
- `prompt_encoder_fingerprint`: String (tokenizer / encoder / checkpoint hash used to derive prompt features).
- `metadata`:
    - `display_name`: Human-readable name (e.g., "Yuri - Drama v3").
    - `language`: Primary language (e.g., "ja").
    - `gender`: "male", "female", "other".
    - `license`: Rights/legality status.
    - `created_at`: ISO 8601 timestamp.
    - `tags`: List of descriptive tags (e.g., ["energetic", "villain", "mature"]).

## 3. Serialization Contract

The `SpeakerProfile` must be serializable to:
- **JSON:** For WebUI and Python Serve interchange.
- **Binary (Protobuf/FlatBuffers/Raw):** For low-latency loading in `tmrvc-engine-rs` and VST plugins.

## 4. Usage Lifecycle

1. **Extraction:** The WebUI / control plane uploads reference audio.
2. **Encoding:** `uclm_model.encode_speaker_prompt` produces `speaker_embed` and `prompt_kv_cache`.
3. **Gallery Save:** The `SpeakerProfile` is saved in backend-managed storage under `models/characters/`.
4. **Casting:** The user selects a `speaker_profile_id` in the UI.
5. **Runtime Loading:**
    - The Python Serve (Worker 04) loads the profile from the `speaker_profile_id`.
    - It reuses stored `prompt_kv_cache` when valid, or reconstructs it from `prompt_codec_tokens` when stale or absent.
    - It passes the embedding and cache to `forward_tts_pointer`.
6. **DAW Integration:** The VST plugin (Worker 04) loads the binary `SpeakerProfile` to provide the same voice locally.

## 5. Security & Provenance

- **Uniqueness:** No two profiles should share the same `speaker_profile_id`.
- **Integrity:** The `reference_audio_hash` allows the system to verify that the extracted features match the intended source.
- **Invalidation:** If `prompt_encoder_fingerprint` no longer matches the active tokenizer / encoder / checkpoint contract, the cached prompt representation must be treated as stale and re-encoded.
- **Audit:** Any creation or deletion of a `SpeakerProfile` must generate an entry in the Audit Log (see `docs/design/auth-spec.md`).
