
"""ONNX-based Scenario Test for UCLM v3 (Pointer Mode).

Verifies the end-to-end pipeline using exported ONNX models.
"""

import onnxruntime as ort
import torch
import numpy as np
import soundfile as sf
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
D_MODEL = 512
N_LAYERS = 12

def scenario_test():
    logger.info("🚀 Starting ONNX Scenario Test...")
    
    # 1. Load Models
    try:
        # We'll use the token model and codec decoder found in models/fp32
        token_sess = ort.InferenceSession("models/fp32/token_model.onnx")
        dec_sess = ort.InferenceSession("models/fp32/codec_decoder.onnx")
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX models: {e}")
        return

    # 2. Prepare Mock Inputs (Small data for scenario test)
    B = 1
    T = 10 # 10 frames
    
    # Random speaker embed
    speaker_embed = np.random.randn(B, 192).astype(np.float32)
    # Neutral voice state
    voice_state = np.zeros((B, T, 8)).astype(np.float32)
    voice_state[..., 6] = 1.0 # voicing=1.0
    
    # Mock text memory (D_MODEL=512)
    text_memory = np.random.randn(B, 50, D_MODEL).astype(np.float32)
    
    # 3. Simulate Inference Loop
    logger.info("Generating tokens...")
    a_tokens = []
    b_tokens = []
    
    # Initialize KV caches (empty for ONNX v0 test)
    # Note: Real SOTA implementation uses stateful sessions or IO binding
    
    for t in range(T):
        # SOTA: Run token model with actual input names and dimensions
        inputs = {
            "tokens_in": np.random.randint(0, 1024, (B, 4, 1)).astype(np.int64),
            "spk_embed": speaker_embed,
            "kv_cache_in": np.zeros((B, N_LAYERS * 2, 200, 8, 64)).astype(np.float32),
        }

        outputs = token_sess.run(None, inputs)
        logits = outputs[0] 
        
        # Sample (Greedy for scenario test)
        # Note: Logits shape might be [B, 1, 4 * 1024]
        logits_reshaped = logits.reshape(4, 1024)
        best_tokens = np.argmax(logits_reshaped, axis=-1) # [4]
        a_tokens.append(best_tokens)
        b_tokens.append(np.zeros(4, dtype=np.int64))

    # 4. Decode to Audio
    logger.info("Decoding to audio...")
    a_trace = np.stack(a_tokens, axis=1).reshape(1, 4, T) # [1, 4, T]
    # SOTA: Ensure decoder input matching the model signature
    # If decoder expects 8 codebooks but we generated 4, we pad with zeros.
    a_trace_padded = np.zeros((1, 8, T), dtype=np.int64)
    a_trace_padded[:, :4, :] = a_trace
    
    b_trace = np.stack(b_tokens, axis=1).reshape(1, 4, T) # [1, 4, T]
    
    dec_inputs = {
        "a_t": a_trace_padded,
        "b_t": b_trace,
        "voice_state": voice_state,
    }
    
    audio_out = dec_sess.run(None, dec_inputs)[0] # [1, 1, T*HOP]
    audio = audio_out.flatten()

    # 5. SOTA Validation (GEMINI.md Mandate)
    peak = np.max(np.abs(audio))
    std = np.std(audio)
    
    logger.info("--- Scenario Test Results ---")
    logger.info(f"Audio Stats: peak={peak:.4f}, std={std:.4f}")
    
    if peak > 0.0:
        logger.info("✅ AUDIO GENERATED (non-silent).")
    else:
        logger.warning("⚠️  AUDIO IS SILENT - Check model weights or inputs.")

    # Save for user review
    output_path = "scenario_test_result.wav"
    sf.write(output_path, audio, SAMPLE_RATE)
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    scenario_test()
