import torch

from tmrvc_train.cli.train_codec import _prepare_decoder_inputs
from tmrvc_train.models.emotion_codec import CodecLoss, EmotionAwareCodec


def test_emotion_codec_single_step_backward_runs():
    """Regression test: codec train step must not fail with in-place autograd errors."""
    torch.manual_seed(0)

    batch_size = 2
    n_frames = 12
    model = EmotionAwareCodec(d_model=64)
    criterion = CodecLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    waveform = torch.randn(batch_size, 1, n_frames * 240)
    voice_state = torch.randn(batch_size, n_frames, 8)
    target_a = torch.randint(0, 1024, (batch_size, 8, n_frames), dtype=torch.long)
    target_b = torch.randint(0, 64, (batch_size, 4, n_frames), dtype=torch.long)

    # Simulate padding labels used by collate_fn.
    target_a[:, :, -1] = -1
    target_b[:, :, -1] = -1

    optimizer.zero_grad(set_to_none=True)
    _, pred_b_logits, _, pred_a_logits = model.encode(waveform)
    target_a_in, target_b_in = _prepare_decoder_inputs(target_a, target_b)
    recon, _ = model.decode(target_a_in, target_b_in, voice_state)
    losses = criterion(recon, waveform, pred_b_logits, target_b, pred_a_logits, target_a)
    losses["loss"].backward()
    optimizer.step()

    assert model.encoder.rvq.codebooks[0].weight.grad is not None
