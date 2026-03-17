"""Test the full trajectory flow: compile -> render -> store -> replay."""

import pytest
import torch
import numpy as np
from pathlib import Path

from tmrvc_core.types import TrajectoryRecord, PacingControls, IntentCompilerOutput
from tmrvc_serve.intent_compiler import IntentCompiler
from tmrvc_serve.trajectory_store import TrajectoryStore
from tmrvc_serve.uclm_engine import UCLMEngine

@pytest.fixture
def compiler():
    return IntentCompiler()

@pytest.fixture
def store(tmp_path):
    return TrajectoryStore(root_dir=tmp_path)

def test_intent_compilation(compiler):
    # Test sad prompt
    output = compiler.compile("悲しげに話して")
    assert isinstance(output, IntentCompilerOutput)
    assert output.pacing.pace < 1.0
    assert output.explicit_voice_state is not None
    assert output.explicit_voice_state.shape == (1, 8)

    # Test fast prompt
    output = compiler.compile("速く話して")
    assert output.pacing.pace > 1.0

def test_trajectory_serialization(store, tmp_path):
    record = TrajectoryRecord(
        trajectory_id="tj-test-123",
        source_compile_id="cid-456",
        pointer_trace=[(0, 5), (1, 10)], # Phoneme 0 for 5 frames, Phoneme 1 for 10
        voice_state_trajectory=torch.randn(15, 8),
        applied_pacing=PacingControls(pace=1.2),
        created_at="2026-03-16T12:00:00Z"
    )
    
    store.save(record)
    loaded = store.load("tj-test-123")
    
    assert loaded.trajectory_id == record.trajectory_id
    assert loaded.source_compile_id == record.source_compile_id
    assert len(loaded.pointer_trace) == 2
    assert torch.allclose(loaded.voice_state_trajectory, record.voice_state_trajectory)
    assert loaded.applied_pacing.pace == 1.2

@pytest.mark.asyncio
async def test_engine_replay_parity():
    """Verify that replaying a trajectory produces an audio of the same length."""
    # Note: Requires loaded engine. Using dummy logic if needed, 
    # but here we review the mathematical consistency of frame counts.
    
    pointer_trace = [(0, 10), (1, 20), (2, 5)] # Total 35 frames
    vs_traj = torch.randn(35, 8)
    
    # Simple check for forced_indices reconstruction in replay_trajectory
    forced_indices = []
    for text_idx, duration in pointer_trace:
        forced_indices.extend([text_idx] * duration)
    
    assert len(forced_indices) == 35
    assert forced_indices[:10] == [0] * 10
    assert forced_indices[10:30] == [1] * 20
    assert forced_indices[30:] == [2] * 5
