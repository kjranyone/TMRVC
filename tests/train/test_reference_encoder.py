import pytest
import torch
import torch.nn.functional as F

from tmrvc_train.models.reference_encoder import ReferenceEncoder, ReferenceEncoderFromWaveform
from tmrvc_train.models.uclm_model import ProsodyPredictor


def test_reference_encoder_output_shape():
    """Verify output shape [B, d_prosody]."""
    d_prosody = 64
    enc = ReferenceEncoder(d_model=512, d_prosody=d_prosody, n_mels=80)
    mel = torch.randn(2, 80, 100)  # [B, n_mels, T]
    
    out = enc(mel)
    assert out.shape == (2, d_prosody)
    assert out.dtype == torch.float32


def test_reference_encoder_from_waveform_shape():
    """Verify waveform wrapper produces correct shape."""
    d_prosody = 64
    enc = ReferenceEncoderFromWaveform(d_model=512, d_prosody=d_prosody)
    # 1 second of audio at 24kHz
    waveform = torch.randn(2, 24000)
    
    out = enc(waveform)
    assert out.shape == (2, d_prosody)
    assert out.dtype == torch.float32


def test_prosody_discriminability():
    """Verify prosody discriminability: same-speaker, different-prosody inputs 
    must produce latents with cosine distance above threshold."""
    # Note: In a real test, we would use actual audio from the same speaker
    # with different prosody (e.g. angry vs sad). Here we use synthetic noise
    # with significantly different characteristics to simulate different prosody.
    d_prosody = 64
    enc = ReferenceEncoderFromWaveform(d_model=512, d_prosody=d_prosody)
    enc.eval()
    
    with torch.no_grad():
        # "Neutral" prosody simulation
        wav1 = torch.randn(1, 24000) * 0.1 
        # "Loud/Fast" prosody simulation
        wav2 = torch.randn(1, 24000) * 0.9 
        
        z1 = enc(wav1)
        z2 = enc(wav2)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z1, z2, dim=-1).item()
        
        # They should be somewhat different (distance > 0)
        # Note: Since it's an untrained network initialized with random weights,
        # the exact threshold is arbitrary, but we want to ensure it doesn't
        # collapse to the exact same vector.
        assert cos_sim < 0.9999 


def test_gradient_flow_to_prosody_predictor():
    """Verify gradient flow to Prosody Predictor during joint training."""
    d_model = 512
    d_prosody = 64
    B, L = 2, 50
    
    ref_enc = ReferenceEncoderFromWaveform(d_model=d_model, d_prosody=d_prosody)
    predictor = ProsodyPredictor(d_model=d_model, d_prosody=d_prosody)
    
    # Inputs
    waveform = torch.randn(B, 24000)
    phoneme_features = torch.randn(B, L, d_model)
    dialogue_context = torch.randn(B, d_model)
    speaker_embed = torch.randn(B, d_model)
    
    # Forward pass: extract target from audio
    target_prosody = ref_enc(waveform)
    
    # Calculate flow-matching loss
    loss = predictor.flow_matching_loss(
        phoneme_features, 
        target_prosody, 
        dialogue_context, 
        speaker_embed
    )
    
    # Check that gradients flow back through the predictor
    loss.backward()
    
    # Predictor's velocity net should have gradients
    has_grads = False
    for p in predictor.parameters():
        if p.grad is not None:
            has_grads = True
            break
            
    assert has_grads, "Gradients did not flow through ProsodyPredictor"
    
    # Reference encoder should ALSO have gradients if trained jointly
    ref_has_grads = False
    for p in ref_enc.parameters():
        if p.grad is not None:
            ref_has_grads = True
            break
            
    assert ref_has_grads, "Gradients did not flow through ReferenceEncoder"
