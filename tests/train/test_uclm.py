import torch
from tmrvc_train.models.uclm import VoiceStateEncoder, VectorQuantizer, VCEncoder
from tmrvc_train.models.uclm_transformer import CodecTransformer


def test_voice_state_encoder():
    B, T = 2, 50
    explicit = torch.randn(B, T, 8)
    ssl = torch.randn(B, T, 128)
    model = VoiceStateEncoder(d_model=512)

    out = model(explicit, ssl)
    # VoiceStateEncoder may return tuple (state_cond, adv_logits) or just state_cond
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (B, T, 512), f"VoiceStateEncoder shape mismatch: {out.shape}"


def test_vector_quantizer():
    B, T, d = 2, 50, 512
    x = torch.randn(B, T, d)
    vq = VectorQuantizer(n_bins=128, d_model=d)

    x_q, loss, indices = vq(x)
    assert x_q.shape == (B, T, d), f"VQ quantized shape mismatch: {x_q.shape}"
    assert indices.shape == (B, T), f"VQ indices shape mismatch: {indices.shape}"
    assert loss.dim() == 0, "VQ loss should be scalar"


def test_vc_encoder():
    B, n_cb, T = 2, 8, 50
    source_tokens = torch.randint(0, 1024, (B, n_cb, T))
    model = VCEncoder(d_model=512, vq_bins=128)

    content_features, loss = model(source_tokens)
    assert content_features.shape == (B, T, 512), (
        f"VCEncoder shape mismatch: {content_features.shape}"
    )
    assert loss.dim() == 0, "VCEncoder loss should be scalar"


def test_uclm_transformer():
    B, T, d = 1, 50, 512
    content = torch.randn(B, T, d) # Corrected shape [B, T, d]
    b_ctx = torch.randint(0, 64, (B, 4, T))
    spk_embed = torch.randn(B, 192)
    state_cond = torch.randn(B, d)
    
    model = CodecTransformer(d_model=d)
    logits_a, logits_b, kv_out = model(content, b_ctx, spk_embed, state_cond)

    assert logits_a.shape == (B, 8, T, 1024), (
        f"Transformer A_t shape mismatch: {logits_a.shape}"
    )
    assert logits_b.shape == (B, 4, T, 64), (
        f"Transformer B_t shape mismatch: {logits_b.shape}"
    )


if __name__ == "__main__":
    test_voice_state_encoder()
    test_vector_quantizer()
    test_vc_encoder()
    test_uclm_transformer()
    print("All UCLM model tests passed!")
