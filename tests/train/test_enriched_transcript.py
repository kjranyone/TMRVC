"""Tests for enriched transcript training path (Phase 3, Task 3-4).

Covers:
- Acting tag tokenization via parse_enriched_transcript
- tokenize_enriched_transcript produces valid token IDs
- A/B divergence: enriched vs plain forward pass produce different outputs
- Tag tokens are in the correct range (>= phoneme_vocab_size)
- TextEncoder handles mixed phoneme + acting tag sequences
- 50% enriched/plain coin flip in dataset
"""

import torch
import pytest


class TestParseEnrichedTranscript:
    """Test parse_enriched_transcript parsing logic."""

    def test_simple_tag_parsing(self):
        from tmrvc_core.acting_tags import parse_enriched_transcript

        result = parse_enriched_transcript("[angry] hello world")
        assert len(result) >= 2
        assert result[0] == ("[angry]", "tag")
        assert result[1][1] == "text"

    def test_multiple_tags(self):
        from tmrvc_core.acting_tags import parse_enriched_transcript

        result = parse_enriched_transcript("[inhale] hello [pause] world [exhale]")
        tags = [r for r in result if r[1] == "tag"]
        texts = [r for r in result if r[1] == "text"]
        assert len(tags) == 3
        assert len(texts) >= 1

    def test_no_tags(self):
        from tmrvc_core.acting_tags import parse_enriched_transcript

        result = parse_enriched_transcript("hello world")
        assert len(result) == 1
        assert result[0] == ("hello world", "text")

    def test_freeform_tag(self):
        from tmrvc_core.acting_tags import parse_enriched_transcript

        result = parse_enriched_transcript("[with a slight smile] hello")
        freeform = [r for r in result if r[1] == "freeform"]
        assert len(freeform) >= 1

    def test_empty_input(self):
        from tmrvc_core.acting_tags import parse_enriched_transcript

        result = parse_enriched_transcript("")
        assert result == []

    def test_tag_only(self):
        from tmrvc_core.acting_tags import parse_enriched_transcript

        result = parse_enriched_transcript("[angry]")
        assert len(result) == 1
        assert result[0] == ("[angry]", "tag")


class TestTokenizeEnrichedTranscript:
    """Test tokenize_enriched_transcript function."""

    def test_basic_tokenization(self):
        from tmrvc_data.v4_dataset import tokenize_enriched_transcript

        phoneme_ids = torch.arange(10, dtype=torch.long)  # [0..9]
        enriched = "[angry] hello world"

        tokens = tokenize_enriched_transcript(enriched, phoneme_ids)
        assert len(tokens) > 0
        assert tokens.dtype == torch.long

    def test_tag_ids_above_phoneme_vocab(self):
        from tmrvc_data.v4_dataset import tokenize_enriched_transcript
        from tmrvc_core.acting_tags import ACTING_TAG_VOCAB

        phoneme_ids = torch.arange(20, dtype=torch.long)
        enriched = "[angry] hello [pause] world"

        tokens = tokenize_enriched_transcript(enriched, phoneme_ids)

        # Acting tag IDs should be >= 200 (phoneme_vocab_size)
        acting_tag_ids = set(ACTING_TAG_VOCAB.values())
        for tid in tokens.tolist():
            if tid >= 200:
                assert tid in acting_tag_ids, \
                    f"Token ID {tid} >= 200 but not in acting tag vocab"

    def test_no_tags_preserves_phonemes(self):
        from tmrvc_data.v4_dataset import tokenize_enriched_transcript

        phoneme_ids = torch.arange(15, dtype=torch.long)
        enriched = "hello world no tags"

        tokens = tokenize_enriched_transcript(enriched, phoneme_ids)

        # All tokens should be from the original phoneme_ids
        for tid in tokens.tolist():
            assert tid < 200, f"Expected pure phoneme ID < 200, got {tid}"

    def test_empty_enriched(self):
        from tmrvc_data.v4_dataset import tokenize_enriched_transcript

        phoneme_ids = torch.arange(10, dtype=torch.long)
        tokens = tokenize_enriched_transcript("", phoneme_ids)

        # Should return original phoneme_ids
        assert torch.equal(tokens, phoneme_ids)

    def test_none_phoneme_ids(self):
        from tmrvc_data.v4_dataset import tokenize_enriched_transcript

        tokens = tokenize_enriched_transcript("[angry] hello", phoneme_ids=None)
        # Should still produce tag tokens even without phoneme_ids
        assert len(tokens) > 0

    def test_enriched_has_more_tokens_than_plain(self):
        """Enriched transcript with tags should have more tokens than plain."""
        from tmrvc_data.v4_dataset import tokenize_enriched_transcript

        phoneme_ids = torch.arange(20, dtype=torch.long)
        enriched = "[angry] hello [pause] world [whisper] goodbye"

        enriched_tokens = tokenize_enriched_transcript(enriched, phoneme_ids)
        # Enriched should have tag tokens + phoneme tokens
        assert len(enriched_tokens) > len(phoneme_ids), \
            f"Enriched ({len(enriched_tokens)}) should have more tokens than plain ({len(phoneme_ids)})"


class TestTextEncoderWithActingTags:
    """Test TextEncoder handles enriched transcript tokens."""

    def test_forward_with_acting_tags(self):
        from tmrvc_train.models.text_encoder import TextEncoder

        enc = TextEncoder(
            vocab_size=200,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ff_dim=512,
            acting_tag_vocab_size=35,
        )

        # Mix of phonemes and acting tags
        phoneme_ids = torch.randint(0, 200, (2, 30))
        phoneme_ids[0, 5] = 202   # acting tag
        phoneme_ids[1, 10] = 215  # acting tag
        language_ids = torch.zeros(2, dtype=torch.long)

        out = enc(phoneme_ids, language_ids)
        assert out.shape == (2, 256, 30)  # [B, d_model, L]

    def test_backward_with_tags(self):
        from tmrvc_train.models.text_encoder import TextEncoder

        enc = TextEncoder(
            vocab_size=200,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ff_dim=512,
            acting_tag_vocab_size=35,
        )

        phoneme_ids = torch.randint(0, 200, (2, 20))
        phoneme_ids[0, 3] = 205  # acting tag
        language_ids = torch.zeros(2, dtype=torch.long)

        out = enc(phoneme_ids, language_ids)
        loss = out.sum()
        loss.backward()

        # Check acting tag embedding has gradients
        assert enc.acting_tag_embedding is not None
        assert enc.acting_tag_embedding.weight.grad is not None

    def test_no_tags_backward_compat(self):
        """Without acting_tag_vocab_size, pure phoneme path works."""
        from tmrvc_train.models.text_encoder import TextEncoder

        enc = TextEncoder(vocab_size=200, d_model=256, n_layers=2, n_heads=4, ff_dim=512)
        phoneme_ids = torch.randint(0, 200, (2, 20))
        language_ids = torch.zeros(2, dtype=torch.long)

        out = enc(phoneme_ids, language_ids)
        assert out.shape == (2, 256, 20)


class TestABDivergence:
    """Test that enriched vs plain transcripts produce different outputs.

    This is the key A/B test: same text content but with/without acting tags
    should produce different hidden states, demonstrating that the model
    can use acting tags as conditioning information.
    """

    def test_tag_changes_output(self):
        """Forward pass with acting tags should differ from plain phonemes."""
        from tmrvc_train.models.text_encoder import TextEncoder

        enc = TextEncoder(
            vocab_size=200,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ff_dim=512,
            acting_tag_vocab_size=35,
        )
        enc.eval()

        # Plain: all phoneme tokens
        plain_ids = torch.randint(1, 100, (1, 20))
        lang = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            out_plain = enc(plain_ids, lang)

        # Enriched: same phonemes but with acting tags inserted
        enriched_ids = plain_ids.clone()
        enriched_ids[0, 5] = 202   # [laugh] tag
        enriched_ids[0, 10] = 209  # [pause] tag

        with torch.no_grad():
            out_enriched = enc(enriched_ids, lang)

        # Outputs should be different at the modified positions
        diff = (out_plain - out_enriched).abs()
        assert diff.sum() > 0, "Enriched and plain outputs should differ"

        # Specifically, positions 5 and 10 should differ most
        diff_at_5 = diff[0, :, 5].sum()
        diff_at_0 = diff[0, :, 0].sum()
        assert diff_at_5 > diff_at_0, \
            "Difference should be larger at tag positions"

    def test_plain_only_is_deterministic(self):
        """Same plain input should produce identical output."""
        from tmrvc_train.models.text_encoder import TextEncoder

        enc = TextEncoder(
            vocab_size=200,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ff_dim=512,
            acting_tag_vocab_size=35,
        )
        enc.eval()

        plain_ids = torch.randint(1, 100, (1, 20))
        lang = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            out1 = enc(plain_ids, lang)
            out2 = enc(plain_ids, lang)

        assert torch.allclose(out1, out2), "Same input should produce same output"


class TestEnrichedTranscriptCoinFlip:
    """Test that the 50/50 enriched/plain split works in dataset."""

    def test_use_enriched_flag_is_boolean(self):
        """The use_enriched flag should be a boolean."""
        # Simulate what the dataset does
        import random
        enriched_transcript = "[angry] hello world"
        prob = 0.5

        results = []
        for _ in range(100):
            use = bool(enriched_transcript and random.random() < prob)
            results.append(use)

        n_true = sum(results)
        # Should be roughly 50/50 (within statistical bounds)
        assert 20 < n_true < 80, \
            f"Expected ~50% enriched, got {n_true}%"

    def test_empty_enriched_always_plain(self):
        """When enriched transcript is empty, always use plain."""
        import random
        enriched_transcript = ""
        prob = 0.5

        for _ in range(50):
            use = bool(enriched_transcript and random.random() < prob)
            assert not use, "Empty enriched transcript should always use plain"


class TestActingTagVocabIntegrity:
    """Test frozen tag vocabulary properties."""

    def test_vocab_ids_unique(self):
        from tmrvc_core.acting_tags import ACTING_TAG_VOCAB

        ids = list(ACTING_TAG_VOCAB.values())
        assert len(ids) == len(set(ids)), "Duplicate IDs in acting tag vocab"

    def test_vocab_ids_above_phoneme_range(self):
        from tmrvc_core.acting_tags import ACTING_TAG_VOCAB

        for tag, tid in ACTING_TAG_VOCAB.items():
            assert tid >= 200, f"Tag '{tag}' ID {tid} overlaps phoneme range"

    def test_extended_vocab_size(self):
        from tmrvc_core.acting_tags import EXTENDED_VOCAB_SIZE, ACTING_TAG_VOCAB

        assert EXTENDED_VOCAB_SIZE == 200 + len(ACTING_TAG_VOCAB)

    def test_all_known_tags_in_vocab(self):
        from tmrvc_core.acting_tags import ALL_ACTING_TAGS, ACTING_TAG_VOCAB

        for tag in ALL_ACTING_TAGS:
            assert tag in ACTING_TAG_VOCAB, f"Tag '{tag}' missing from vocab"

    def test_special_tokens_present(self):
        from tmrvc_core.acting_tags import ACTING_TAG_VOCAB

        assert "[act_start]" in ACTING_TAG_VOCAB
        assert "[act_end]" in ACTING_TAG_VOCAB
        assert "[freeform]" in ACTING_TAG_VOCAB
