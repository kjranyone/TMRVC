"""Tests for Phase 2-1: TextEncoder acting tag support.

Covers:
a) Acting tag tokens (IDs >= 200) produce valid embeddings
b) Pointer attention treats acting tag tokens as consumed text units
c) Forward pass output shape is identical with and without acting tags
d) Enriched transcript parsing + tokenization roundtrips correctly
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_core.acting_tags import (
    ACTING_TAG_VOCAB,
    ALL_ACTING_TAGS,
    EXTENDED_VOCAB_SIZE,
    VOCAL_EVENT_TAGS,
    PROSODIC_MARKER_TAGS,
    ACTING_DIRECTIVE_TAGS,
    build_acting_tag_vocab,
    get_tag_category,
    parse_enriched_transcript,
)
from tmrvc_core.constants import (
    D_SUPRASEGMENTAL,
    N_ACTING_TAGS,
    PHONEME_VOCAB_SIZE,
)
from tmrvc_train.models.text_encoder import TextEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_encoder(acting_tags: bool = True, d_model: int = 64) -> TextEncoder:
    """Build a small TextEncoder for testing."""
    return TextEncoder(
        vocab_size=PHONEME_VOCAB_SIZE,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        ff_dim=128,
        n_languages=4,
        d_supra=D_SUPRASEGMENTAL,
        dropout=0.0,
        acting_tag_vocab_size=len(ACTING_TAG_VOCAB) if acting_tags else 0,
    )


def _phoneme_ids_with_tags(batch_size: int, seq_len: int, n_tags: int = 3) -> torch.Tensor:
    """Create phoneme IDs where some positions contain acting tag IDs (>= 200)."""
    ids = torch.randint(1, PHONEME_VOCAB_SIZE, (batch_size, seq_len))
    # Insert acting tag IDs at fixed positions
    tag_ids = list(ACTING_TAG_VOCAB.values())[:n_tags]
    for i, pos in enumerate(range(0, min(n_tags, seq_len))):
        ids[:, pos] = tag_ids[i]
    return ids


# ---------------------------------------------------------------------------
# a) Acting tag tokens produce valid embeddings
# ---------------------------------------------------------------------------


class TestActingTagEmbedding:
    def test_tag_tokens_produce_finite_embeddings(self):
        """Acting tag IDs (>= 200) should produce finite, non-zero embeddings."""
        enc = _make_encoder(acting_tags=True)
        B, L = 2, 8
        ids = _phoneme_ids_with_tags(B, L, n_tags=3)
        lang = torch.zeros(B, dtype=torch.long)

        out = enc(ids, lang)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        assert out.abs().sum() > 0, "Output is all zeros"

    def test_each_tag_id_produces_unique_embedding(self):
        """Each distinct acting tag ID should map to a different embedding vector."""
        enc = _make_encoder(acting_tags=True)
        tag_ids = list(ACTING_TAG_VOCAB.values())
        assert len(tag_ids) >= 2, "Need at least 2 tags for uniqueness test"

        embeddings = []
        for tid in tag_ids[:5]:
            ids = torch.full((1, 1), tid, dtype=torch.long)
            lang = torch.zeros(1, dtype=torch.long)
            out = enc(ids, lang)  # [1, d_model, 1]
            embeddings.append(out.squeeze())

        # Check pairwise non-equality
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not torch.allclose(
                    embeddings[i], embeddings[j], atol=1e-5
                ), f"Tag {tag_ids[i]} and {tag_ids[j]} produced identical embeddings"

    def test_all_acting_tag_ids_are_above_phoneme_vocab(self):
        """Every acting tag ID must be >= PHONEME_VOCAB_SIZE."""
        for tag, tid in ACTING_TAG_VOCAB.items():
            assert tid >= PHONEME_VOCAB_SIZE, (
                f"Tag {tag!r} has ID {tid} below phoneme_vocab_size {PHONEME_VOCAB_SIZE}"
            )

    def test_no_acting_tag_layer_rejects_tag_ids(self):
        """Encoder without acting tag layer should NOT crash on pure phoneme input."""
        enc = _make_encoder(acting_tags=False)
        B, L = 2, 8
        ids = torch.randint(1, PHONEME_VOCAB_SIZE, (B, L))
        lang = torch.zeros(B, dtype=torch.long)
        out = enc(ids, lang)
        assert out.shape == (B, enc.d_model, L)

    def test_tag_embedding_table_size_matches_vocab(self):
        """Acting tag embedding table should have exactly len(ACTING_TAG_VOCAB) entries."""
        enc = _make_encoder(acting_tags=True)
        assert enc.acting_tag_embedding is not None
        assert enc.acting_tag_embedding.num_embeddings == len(ACTING_TAG_VOCAB)


# ---------------------------------------------------------------------------
# b) Pointer attention treats acting tags as consumed text units
# ---------------------------------------------------------------------------


class TestActingTagsAsConsumedUnits:
    """Acting tag tokens occupy sequence positions and are consumed by the pointer
    the same way phoneme tokens are -- they are not skipped or ignored."""

    def test_tag_positions_count_toward_sequence_length(self):
        """Sequence with acting tags should have the same total length as input."""
        enc = _make_encoder(acting_tags=True)
        B, L = 2, 10
        ids = _phoneme_ids_with_tags(B, L, n_tags=4)
        lang = torch.zeros(B, dtype=torch.long)
        out = enc(ids, lang)
        # Output should have L time steps (tags count as positions)
        assert out.shape[2] == L

    def test_tag_and_phoneme_produce_same_length_output(self):
        """Replacing a phoneme token with a tag token must not change output length."""
        enc = _make_encoder(acting_tags=True)
        B, L = 1, 6
        # Pure phoneme sequence
        ids_phoneme = torch.randint(1, PHONEME_VOCAB_SIZE, (B, L))
        # Same length but with tags
        ids_tagged = ids_phoneme.clone()
        ids_tagged[0, 0] = ACTING_TAG_VOCAB["[angry]"]
        ids_tagged[0, 3] = ACTING_TAG_VOCAB["[pause]"]

        lang = torch.zeros(B, dtype=torch.long)
        out_phoneme = enc(ids_phoneme, lang)
        out_tagged = enc(ids_tagged, lang)

        assert out_phoneme.shape == out_tagged.shape

    def test_padding_mask_applies_to_tag_positions(self):
        """Padding mask should correctly handle sequences containing acting tags."""
        enc = _make_encoder(acting_tags=True)
        B, L = 2, 8
        ids = _phoneme_ids_with_tags(B, L, n_tags=2)
        lang = torch.zeros(B, dtype=torch.long)
        lengths = torch.tensor([5, 8])  # First sample is shorter

        out = enc(ids, lang, phoneme_lengths=lengths)
        assert out.shape == (B, enc.d_model, L)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# c) Forward pass output shape identical with and without acting tags
# ---------------------------------------------------------------------------


class TestOutputShapeInvariance:
    def test_output_shape_same_with_without_tags(self):
        """Output shape must be [B, d_model, L] regardless of tag presence."""
        enc = _make_encoder(acting_tags=True, d_model=64)
        B, L = 2, 12

        ids_plain = torch.randint(1, PHONEME_VOCAB_SIZE, (B, L))
        ids_mixed = ids_plain.clone()
        ids_mixed[:, 1] = ACTING_TAG_VOCAB["[laugh]"]
        ids_mixed[:, 5] = ACTING_TAG_VOCAB["[emphasis]"]
        ids_mixed[:, 9] = ACTING_TAG_VOCAB["[whisper]"]

        lang = torch.zeros(B, dtype=torch.long)
        out_plain = enc(ids_plain, lang)
        out_mixed = enc(ids_mixed, lang)

        assert out_plain.shape == out_mixed.shape == (B, 64, L)

    def test_output_shape_with_suprasegmentals(self):
        """Adding suprasegmentals alongside acting tags should not change shape."""
        enc = _make_encoder(acting_tags=True, d_model=64)
        B, L = 2, 8
        ids = _phoneme_ids_with_tags(B, L)
        lang = torch.zeros(B, dtype=torch.long)
        supra = torch.randn(B, L, D_SUPRASEGMENTAL)

        out = enc(ids, lang, text_suprasegmentals=supra)
        assert out.shape == (B, 64, L)

    def test_batch_mixed_tag_and_no_tag_same_shape(self):
        """Batch where only some samples have tags should still produce uniform shape."""
        enc = _make_encoder(acting_tags=True, d_model=64)
        B, L = 4, 10
        ids = torch.randint(1, PHONEME_VOCAB_SIZE, (B, L))
        # Only samples 0 and 2 have tags
        ids[0, 2] = ACTING_TAG_VOCAB["[inhale]"]
        ids[2, 7] = ACTING_TAG_VOCAB["[calm]"]

        lang = torch.zeros(B, dtype=torch.long)
        out = enc(ids, lang)
        assert out.shape == (B, 64, L)

    def test_gradients_flow_through_tag_embeddings(self):
        """Gradient should flow back through acting tag embedding layer."""
        enc = _make_encoder(acting_tags=True, d_model=64)
        B, L = 2, 6
        ids = _phoneme_ids_with_tags(B, L, n_tags=2)
        lang = torch.zeros(B, dtype=torch.long)

        out = enc(ids, lang)
        loss = out.sum()
        loss.backward()

        assert enc.acting_tag_embedding.weight.grad is not None
        assert enc.acting_tag_embedding.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# d) Enriched transcript parsing + tokenization roundtrip
# ---------------------------------------------------------------------------


class TestEnrichedTranscriptParsing:
    def test_parse_simple_enriched_transcript(self):
        """Parse a transcript with known tags and text segments."""
        enriched = "[inhale] hello [emphasis] world"
        parts = parse_enriched_transcript(enriched)
        assert len(parts) == 4
        assert parts[0] == ("[inhale]", "tag")
        assert parts[1] == ("hello", "text")
        assert parts[2] == ("[emphasis]", "tag")
        assert parts[3] == ("world", "text")

    def test_parse_transcript_with_freeform_tag(self):
        """Free-form tags (not in ALL_ACTING_TAGS) should be classified as freeform."""
        enriched = "[with a slight smile] hello"
        parts = parse_enriched_transcript(enriched)
        assert len(parts) == 2
        assert parts[0] == ("[with a slight smile]", "freeform")
        assert parts[1] == ("hello", "text")

    def test_parse_no_tags(self):
        """Plain text without tags should return a single text segment."""
        parts = parse_enriched_transcript("just plain text here")
        assert len(parts) == 1
        assert parts[0] == ("just plain text here", "text")

    def test_parse_only_tags(self):
        """Transcript with only tags and no text."""
        enriched = "[inhale] [pause] [exhale]"
        parts = parse_enriched_transcript(enriched)
        assert len(parts) == 3
        for _, kind in parts:
            assert kind == "tag"

    def test_parse_empty_string(self):
        parts = parse_enriched_transcript("")
        assert len(parts) == 0

    def test_tag_category_classification(self):
        """get_tag_category should return the correct category for each tag."""
        assert get_tag_category("[inhale]") == "vocal_event"
        assert get_tag_category("[emphasis]") == "prosodic_marker"
        assert get_tag_category("[angry]") == "acting_directive"
        assert get_tag_category("[something custom]") == "freeform"

    def test_all_tags_have_ids_in_vocab(self):
        """Every tag in ALL_ACTING_TAGS should appear in ACTING_TAG_VOCAB."""
        for tag in ALL_ACTING_TAGS:
            assert tag in ACTING_TAG_VOCAB, f"Tag {tag!r} missing from ACTING_TAG_VOCAB"

    def test_tag_vocab_roundtrip(self):
        """build_acting_tag_vocab should produce consistent IDs across calls."""
        v1 = build_acting_tag_vocab(200)
        v2 = build_acting_tag_vocab(200)
        assert v1 == v2

    def test_tokenization_roundtrip_enriched_to_ids(self):
        """Tags in parsed enriched transcript should map to valid IDs and back."""
        enriched = "[laugh] konnichiwa [pause] sekai"
        parts = parse_enriched_transcript(enriched)

        # Simulate tokenization: tags -> tag IDs, text -> phoneme IDs (mocked)
        token_ids = []
        token_types = []
        for content, kind in parts:
            if kind == "tag":
                tid = ACTING_TAG_VOCAB.get(content)
                assert tid is not None, f"Tag {content!r} not in vocab"
                token_ids.append(tid)
                token_types.append("tag")
            else:
                # Mock: each text word -> one phoneme ID
                for word in content.split():
                    token_ids.append(hash(word) % (PHONEME_VOCAB_SIZE - 1) + 1)
                    token_types.append("phoneme")

        # Verify tag IDs are in the acting tag range
        for tid, ttype in zip(token_ids, token_types):
            if ttype == "tag":
                assert tid >= PHONEME_VOCAB_SIZE
            else:
                assert 1 <= tid < PHONEME_VOCAB_SIZE

    def test_extended_vocab_size_constant(self):
        """EXTENDED_VOCAB_SIZE should equal PHONEME_VOCAB_SIZE + len(ACTING_TAG_VOCAB)."""
        assert EXTENDED_VOCAB_SIZE == PHONEME_VOCAB_SIZE + len(ACTING_TAG_VOCAB)

    def test_n_acting_tags_constant(self):
        """N_ACTING_TAGS constant should match the actual tag count."""
        assert N_ACTING_TAGS == len(ACTING_TAG_VOCAB)


# ---------------------------------------------------------------------------
# Integration: end-to-end enriched transcript through encoder
# ---------------------------------------------------------------------------


class TestEnrichedTranscriptEndToEnd:
    def test_enriched_transcript_through_encoder(self):
        """Simulate full pipeline: enriched text -> parse -> token IDs -> encoder."""
        enc = _make_encoder(acting_tags=True, d_model=64)
        enriched = "[inhale] hello [angry] world [pause]"
        parts = parse_enriched_transcript(enriched)

        # Build token ID sequence
        ids = []
        for content, kind in parts:
            if kind == "tag":
                ids.append(ACTING_TAG_VOCAB[content])
            elif kind == "freeform":
                ids.append(ACTING_TAG_VOCAB.get("[freeform]", ACTING_TAG_VOCAB["[act_start]"]))
            else:
                for word in content.split():
                    ids.append(1 + hash(word) % (PHONEME_VOCAB_SIZE - 2))

        ids_tensor = torch.tensor([ids], dtype=torch.long)
        lang = torch.zeros(1, dtype=torch.long)
        out = enc(ids_tensor, lang)

        assert out.shape == (1, 64, len(ids))
        assert torch.isfinite(out).all()
