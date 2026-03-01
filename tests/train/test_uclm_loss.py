import torch
from tmrvc_train.models.uclm_loss import uclm_loss


def test_uclm_loss():
    B, T = 2, 50
    n_cb, vocab_a = 8, 1024
    n_slots, vocab_b = 4, 64

    logits_a = torch.randn(B, n_cb, T, vocab_a)
    logits_b = torch.randn(B, n_slots, T, vocab_b)

    target_a = torch.randint(0, vocab_a, (B, n_cb, T))
    target_b = torch.randint(0, vocab_b, (B, n_slots, T))

    vq_loss = torch.tensor(0.5)

    losses = uclm_loss(logits_a, logits_b, target_a, target_b, vq_loss)

    assert "loss" in losses
    assert "loss_a" in losses
    assert "loss_b" in losses
    assert "loss_vq" in losses  # vq_loss provided, should be in output

    assert losses["loss"].dim() == 0
    assert losses["loss_a"].dim() == 0
    assert losses["loss_b"].dim() == 0
    assert losses["loss_vq"].dim() == 0

    # check without vq_loss - loss_vq should NOT be in output
    losses_no_vq = uclm_loss(logits_a, logits_b, target_a, target_b)
    assert "loss_vq" not in losses_no_vq  # vq_loss not provided, key should be absent


if __name__ == "__main__":
    test_uclm_loss()
    print("Loss function tests passed!")
