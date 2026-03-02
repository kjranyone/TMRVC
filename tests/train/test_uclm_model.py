import torch
from tmrvc_train.models.uclm_model import DisentangledUCLM

def test_uclm_vc_forward():
    B, T = 2, 50
    model = DisentangledUCLM(
        d_model=256, 
        n_heads=4, 
        n_layers=2, 
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_explicit=8,
        d_ssl=128,
        d_speaker=192,
        vq_bins=64
    )
    
    source_a_t = torch.randint(0, 1024, (B, 8, T))
    explicit_state = torch.randn(B, T, 8)
    ssl_state = torch.randn(B, T, 128)
    speaker_embed = torch.randn(B, 192)
    
    target_b = torch.randint(0, 64, (B, 4, T))
    out = model.forward_vc(source_a_t, target_b, explicit_state, ssl_state, speaker_embed)
    
    assert "logits_a" in out
    assert "logits_b" in out
    assert "vq_loss" in out
    
    assert out["logits_a"].shape == (B, 8, T, 1024)
    assert out["logits_b"].shape == (B, 4, T, 64)
    assert out["vq_loss"].dim() == 0

def test_uclm_tts_forward():
    B, L, T = 2, 20, 50
    model = DisentangledUCLM(
        d_model=256, 
        n_heads=4, 
        n_layers=2, 
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_explicit=8,
        d_ssl=128,
        d_speaker=192,
        vq_bins=64,
        vocab_size=256
    )
    
    phonemes = torch.randint(0, 256, (B, L))
    # mock lengths to expand phonemes to T frames
    phoneme_lens = torch.full((B,), L) 
    language_ids = torch.zeros((B,), dtype=torch.long)
    
    explicit_state = torch.randn(B, T, 8)
    ssl_state = torch.randn(B, T, 128)
    speaker_embed = torch.randn(B, 192)
    
    target_b = torch.randint(0, 64, (B, 4, T))
    out = model.forward_tts(phonemes, phoneme_lens, language_ids, target_b, explicit_state, ssl_state, speaker_embed)
    
    assert "logits_a" in out
    assert "logits_b" in out
    assert "vq_loss" not in out
    
    assert out["logits_a"].shape == (B, 8, T, 1024)
    assert out["logits_b"].shape == (B, 4, T, 64)

if __name__ == "__main__":
    test_uclm_vc_forward()
    test_uclm_tts_forward()
    print("DisentangledUCLM integration tests passed!")
