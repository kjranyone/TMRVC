#!/bin/bash
# Parallel VCTK preprocessing - 3 workers

SPEAKERS=$(ls data/raw/wav48_silence_trimmed/ | grep "^p[0-9]" | sort)
TOTAL=$(echo "$SPEAKERS" | wc -l)
CHUNK_SIZE=$((($TOTAL + 2) / 3))

echo "Total speakers: $TOTAL"
echo "Chunk size: $CHUNK_SIZE"

# Split into 3 groups
echo "$SPEAKERS" | split -l $CHUNK_SIZE -d - /tmp/speaker_list_

# Worker function
run_worker() {
    local id=$1
    local file=/tmp/speaker_list_0$id
    
    if [ -f "$file" ]; then
        echo "Worker $id: processing $(wc -l < $file) speakers"
        
        PYTHONPATH=tmrvc-core/src:tmrvc-data/src .venv/bin/python -c "
import sys
sys.path.insert(0, 'tmrvc-core/src')
sys.path.insert(0, 'tmrvc-data/src')

from pathlib import Path
from tmrvc_data.dataset_adapters import VCTKAdapter
from tmrvc_data.preprocessing import preprocess_audio
from tmrvc_data.codec import UCLMCodecWrapper
from tmrvc_data.voice_state import SSLVoiceStateEstimator
from tmrvc_data.speaker import SpeakerEncoder
from tmrvc_core.audio import compute_mel
from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.types import UCLMFeatureSet
from tmrvc_data.cache import FeatureCache
import torch
import numpy as np
import tqdm
import soundfile as sf
import torchaudio.transforms as T
from faster_whisper import WhisperModel

device = 'cuda'
speakers = Path('$file').read_text().strip().split('\n')

print(f'Worker $id: {len(speakers)} speakers')

# Load models
codec = UCLMCodecWrapper(None, device=device)
vs_estimator = SSLVoiceStateEstimator(device=device)
spk_encoder = SpeakerEncoder(device=device)
whisper = WhisperModel('large-v3-turbo', device=device, compute_type='float16')
cache = FeatureCache(Path('data/cache'))

adapter = VCTKAdapter()
all_utterances = list(adapter.iter_utterances(Path('data/raw'), 'train'))

# Filter by speakers
utterances = [u for u in all_utterances if any(s in u.speaker_id for s in speakers)]
print(f'Worker $id: {len(utterances)} utterances')

for utt in tqdm.tqdm(utterances, desc=f'Worker $id'):
    try:
        info = sf.info(str(utt.audio_path))
        if info.duration < 0.1 or info.duration > 30.0:
            continue
            
        waveform, sr = preprocess_audio(str(utt.audio_path), target_sr=SAMPLE_RATE)
        waveform_t = waveform.unsqueeze(0).to(device)
        
        a_tokens, b_logits = codec.encode(waveform_t)
        b_tokens = b_logits.argmax(dim=-1)
        
        mel = compute_mel(waveform_t.squeeze(1)).to(device)
        f0 = torch.zeros(1, 1, mel.shape[-1], device=device)
        waveform_16k = T.Resample(SAMPLE_RATE, 16000).to(device)(waveform_t.squeeze(1))
        vs_dict = vs_estimator(waveform_16k, waveform_t.squeeze(1), mel, f0)
        
        segments, _ = whisper.transcribe(str(utt.audio_path), language='en')
        text = ''.join(seg.text for seg in segments).strip()
        
        spk_embed = spk_encoder.extract(waveform_t.squeeze(1))
        
        # Frame alignment (with assertions)
        T_target = a_tokens.shape[-1]
        T_mel = mel.shape[-1]
        assert T_mel == T_target, f'Frame mismatch: mel={T_mel}, codec={T_target}'
        
        explicit_state = vs_dict['explicit_state'].detach().cpu()
        if explicit_state.dim() == 3:
            explicit_state = explicit_state.squeeze(0)
        T_explicit = explicit_state.shape[0]
        assert T_explicit == T_target, f'Frame mismatch: explicit={T_explicit}, codec={T_target}'
        explicit_state = explicit_state.transpose(0, 1)
        
        ssl_state = vs_dict['ssl_state'].detach().cpu()
        if ssl_state.dim() == 3:
            ssl_state = ssl_state.squeeze(0)
        ssl_state = torch.nn.functional.interpolate(
            ssl_state.unsqueeze(0).transpose(1, 2),
            size=T_target, mode='linear', align_corners=False
        ).transpose(1, 2).squeeze(0).transpose(0, 1)
        
        b_tokens_aligned = b_tokens.detach().cpu().squeeze(0)
        T_b = b_tokens_aligned.shape[-1]
        assert T_b == T_target, f'Frame mismatch: b={T_b}, codec={T_target}'
        
        features = UCLMFeatureSet(
            codec_tokens_a=a_tokens.detach().cpu().squeeze(0),
            codec_tokens_b=b_tokens_aligned,
            voice_state_explicit=explicit_state,
            voice_state_ssl=ssl_state,
            spk_embed=spk_embed.detach().cpu().squeeze(0),
            text=text,
            utterance_id=utt.utterance_id,
            speaker_id=utt.speaker_id,
            n_frames=T_target,
            waveform=waveform.detach()
        )
        cache.save(features, 'vctk', 'train')
        
    except Exception as e:
        print(f'Error {utt.utterance_id}: {e}')
        continue

print(f'Worker $id done (skipped {skipped} cached)')
" >> logs/preprocess_worker_$id.log 2>&1 &
    fi
}

# Start 3 workers
for i in 0 1 2; do
    run_worker $i
done

wait
echo "All workers done"
