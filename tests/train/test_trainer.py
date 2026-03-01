import torch
from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.trainer import UCLMTrainer

def test_trainer_step():
    B, T = 2, 50
    model = DisentangledUCLM(
        d_model=128, 
        n_heads=2, 
        n_layers=1, 
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_explicit=8,
        d_ssl=128,
        d_speaker=192,
        vq_bins=32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = UCLMTrainer(model, optimizer, device="cpu")
    
    batch = {
        "source_a_t": torch.randint(0, 1024, (B, 8, T)),
        "target_a": torch.randint(0, 1024, (B, 8, T)),
        "target_b": torch.randint(0, 64, (B, 4, T)),
        "explicit_state": torch.randn(B, T, 8),
        "ssl_state": torch.randn(B, T, 128),
        "speaker_embed": torch.randn(B, 192),
    }
    
    losses = trainer.train_step(batch)
    
    assert "loss" in losses
    assert "loss_a" in losses
    assert "loss_b" in losses
    assert "loss_vq" in losses
    assert losses["loss"] > 0

if __name__ == "__main__":
    test_trainer_step()
    print("Trainer test passed!")
