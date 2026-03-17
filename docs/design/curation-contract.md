# TMRVC Curation Contract

この文書は、mainline curation の manifest / promotion / export 契約の正本である。active backlog が必要な場合のみ `plan/README.md` と `plan/repo_remaining_inventory_2026_03.md` を参照する。

## 1. Scope

- curation record の canonical schema
- optimistic locking に使う `metadata_version`
- promotion buckets
- cache / evaluation bundle への export contract
- WebUI と backend の責務境界
- provider artifact / provider-revision pinning は `docs/design/provider-registry.md` を参照する

## 2. Authoritative Boundary

- multi-user 書き込みは `tmrvc-serve` の `/ui/*` と `/admin/*` を通す
- `tmrvc-data` は内部 curation service として実装されてよいが、filesystem 直接更新は dev-only
- UI は manifest を直接 mutate しない
- provider artifact identity, language support, and fallback policy は `docs/design/provider-registry.md` の pinned entry に従う

## 3. Manifest Contract

各 record は最低限以下を持つ。

- `record_id`
- `source_path`
- `audio_hash`
- `segment_start_sec`
- `segment_end_sec`
- `duration_sec`
- `language`
- `transcript`
- `transcript_confidence`
- `speaker_cluster`
- `diarization_confidence`
- `conversation_id`
- `turn_index`
- `prev_record_id`
- `next_record_id`
- `context_window_ids`
- `quality_score`
- `score_components`
- `status`
- `promotion_bucket`
- `rejection_reasons`
- `review_reasons`
- `providers`
- canonical provider decision provenance:
  - `provider_id`
  - `provider_revision`
  - `calibration_version`
  - optional `fallback_class`
- `pass_index`
- `source_legality`
- `voice_state_target_source`
- `voice_state_observed_ratio`
- `voice_state_confidence_summary`
- `metadata_version`

## 4. Optimistic Locking

- canonical field name は `metadata_version`
- UI は read 時点の `metadata_version` を write request に含める
- backend は一致時のみ更新し、成功時に increment する
- 不一致時は typed `409 Conflict` を返す

最低限の conflict type:

- `stale_version`
- `locked_by_other`
- `already_submitted`
- `policy_forbidden`

## 5. Status and Decision Policy

record の主状態は次を前提とする。

- `review`
- `reject`
- `promote`

判定の基本方針:

- hard reject: transcript empty, severe overlap, wrong language, corruption, separation damage
- review: provider disagreement, marginal transcript confidence, speaker uncertainty, partial event failure
- promote: hard constraints を満たし、quality score と provenance が sufficient
- fallback/review required:
  - required provider が対象言語を未サポート
  - provider confidence に `calibration_version` がない
  - provider downgrade が入っているのに mainline threshold をそのまま適用しようとしている

## 6. Promotion Buckets

mainline bucket は以下を使う。

### `tts_mainline`

要求:

- reliable transcript
- reliable language
- usable text units
- acceptable speaker trust
- dialogue-derived sample では context graph preserved
- `voice_state` supervision status が recorded

### `vc_prior`

要求:

- usable speech quality
- transcript qualityは `tts_mainline` より緩和可能

### `expressive_prior`

要求:

- useful prosody / event signal
- accepted recipe で使える `voice_state` pseudo-label density

### `holdout_eval`

要求:

- high confidence
- diversity
- strict no-leak split

## 7. Export Targets

### Target A: Curation manifest

再監査・再実行用の完全 manifest snapshot。

### Target B: TMRVC cache-ready subset

直接学習に入る cache bundle。

### Target C: Evaluation subset package

再現可能な holdout / blind A/B 用 bundle。

## 8. Required Export Fields

export 先にかかわらず最低限保持する。

- transcript
- language
- text units
- speaker metadata
- conversation metadata
- source legality
- quality score
- provider provenance
- provider decision provenance (`provider_id`, `provider_revision`, `calibration_version`, optional `fallback_class`)
- promotion bucket
- `voice_state` supervision status

## 9. Cache Export Contract

cache-ready export は以下を materialize する。

- `phoneme_ids.npy`
- `meta.json`
- `speaker` metadata
- `conversation_id`, `turn_index`, `prev_record_id`, `next_record_id`, `context_window_ids`
- optional `bootstrap_alignment.json`
- optional `voice_state.npy`
- optional `voice_state_observed_mask.npy`
- optional `voice_state_confidence.npy`
- optional `voice_state_meta.json`

規則:

- `bootstrap_alignment.json` は canonical phoneme space に投影済みであること
- frame convention は `24 kHz`, `hop_length = 240`, `T = ceil(num_samples / 240)`、`start_frame` inclusive / `end_frame` exclusive
- `voice_state` supervision を export する場合、mask と provenance を必須とする
- any exported score or bucket decision that depends on provider output must retain `provider_id`, `provider_revision`, and `calibration_version`

## 10. Separation Policy

- separated waveform は初期 mainline では annotation aid として扱う
- `tts_mainline` は separation-derived waveform teachers を自動採用しない
- research bucket での teacher 利用は `separation_confidence`、artifact、timbre preservation、人手承認を必要とする

## 11. Human Workflow Fields

audit-critical action は最低限以下を保存する。

- `actor_role`
- `actor_id`
- `timestamp`
- `action_type`
- `before_state`
- `after_state`
- `rationale`
- `metadata_version`

## 12. Forbidden

- UI から manifest を直接 mutate すること
- `metadata_version` を使わずに multi-user update を通すこと
- review item を train bucket に混ぜること
- provenance を落として export すること
- shell access を export / download の前提にすること
- `provider_id` / `provider_revision` / `calibration_version` を欠いた score を mainline promote policy に流すこと
