import torch

from tmrvc_train.cli.train_uclm import collate_fn
from tmrvc_train.models.uclm_model import DisentangledUCLM


def _make_item(T: int) -> dict[str, torch.Tensor]:
    return {
        "target_a": torch.randint(0, 1024, (8, T), dtype=torch.long),
        "target_b": torch.randint(0, 64, (4, T), dtype=torch.long),
        "source_a_t": torch.randint(0, 1024, (8, T), dtype=torch.long),
        "explicit_state": torch.randn(T, 8),
        "ssl_state": torch.randn(T, 128),
        "speaker_embed": torch.randn(192),
        "speaker_id": torch.tensor(0, dtype=torch.long),
    }


def test_collate_source_padding_stays_embedding_safe():
    batch = [_make_item(12), _make_item(9)]
    out = collate_fn(batch)

    # source_a_t is embedding input, so padded region must be non-negative.
    assert (out["source_a_t"][1, :, 9:] == 0).all()
    # Targets keep -1 padding for CE ignore_index.
    assert (out["target_a"][1, :, 9:] == -1).all()
    assert (out["target_b"][1, :, 9:] == -1).all()


def test_forward_vc_accepts_padded_target_b_context():
    B, T = 2, 16
    model = DisentangledUCLM(d_model=256, n_heads=4, n_layers=2, vq_bins=64)

    source_a_t = torch.randint(0, 1024, (B, 8, T), dtype=torch.long)
    target_b = torch.randint(0, 64, (B, 4, T), dtype=torch.long)
    target_b[:, :, -4:] = -1

    out = model.forward_vc(
        source_a_t=source_a_t,
        target_b=target_b,
        explicit_state=torch.randn(B, T, 8),
        ssl_state=torch.randn(B, T, 128),
        speaker_embed=torch.randn(B, 192),
    )
    assert out["logits_a"].shape == (B, 8, T, 1024)
    assert out["logits_b"].shape == (B, 4, T, 64)


def test_forward_tts_pointer_accepts_padded_target_b_context():
    B, L, T = 2, 6, 16
    model = DisentangledUCLM(d_model=256, n_heads=4, n_layers=2, vq_bins=64)

    target_b = torch.randint(0, 64, (B, 4, T), dtype=torch.long)
    target_b[:, :, -3:] = -1

    out = model.forward_tts_pointer(
        phoneme_ids=torch.randint(0, 256, (B, L), dtype=torch.long),
        language_ids=torch.zeros((B,), dtype=torch.long),
        pointer_state=None,
        speaker_embed=torch.randn(B, 192),
        explicit_state=torch.randn(B, T, 8),
        ssl_state=torch.randn(B, T, 128),
        target_a=torch.zeros(B, 8, T, dtype=torch.long),
        target_b=target_b,
        target_length=T,
    )
    assert out["logits_a"].shape == (B, 8, T, 1024)
    assert out["logits_b"].shape == (B, 4, T, 64)
