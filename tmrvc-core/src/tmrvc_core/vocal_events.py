"""Vocal event taxonomy for v4 enriched transcripts.

Defines the full set of non-linguistic and para-linguistic vocal events
that the bootstrap pipeline must detect and the text encoder must learn.

These are distinct from acting_tags.py (which covers directive tags like
[angry], [whisper]). This module covers *observable audio events* that
a DSP detector can identify from the waveform.

Categories:
1. Respiratory events (breath-related)
2. Phonation events (voice quality changes)
3. Emotional vocalizations (non-speech expressions)
4. Articulatory events (mouth/tongue sounds)
5. Prosodic events (rhythm/timing markers)
6. Paralinguistic events (communication sounds)
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class VocalEventDef:
    """Definition of a vocal event type."""
    tag: str                    # inline tag string
    category: str               # category name
    description: str            # human-readable description
    dsp_detectable: bool        # can be detected by DSP alone
    min_duration_ms: float      # minimum typical duration
    detection_method: str       # how to detect: dsp, classifier, llm_infer


# ---------------------------------------------------------------------------
# Full Vocal Event Taxonomy
# ---------------------------------------------------------------------------

VOCAL_EVENTS: Dict[str, VocalEventDef] = {
    # --- 1. Respiratory ---
    "inhale": VocalEventDef(
        tag="[inhale]", category="respiratory",
        description="Audible breath intake before speech",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_spectral_energy",
    ),
    "exhale": VocalEventDef(
        tag="[exhale]", category="respiratory",
        description="Audible breath release",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_spectral_energy",
    ),
    "gasp": VocalEventDef(
        tag="[gasp]", category="respiratory",
        description="Sudden sharp intake of breath (surprise/shock)",
        dsp_detectable=True, min_duration_ms=50,
        detection_method="dsp_energy_spike",
    ),
    "held_breath": VocalEventDef(
        tag="[held_breath]", category="respiratory",
        description="Audible breath hold before speaking",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_silence_after_inhale",
    ),

    # --- 2. Phonation (voice quality) ---
    "voice_break": VocalEventDef(
        tag="[voice_break]", category="phonation",
        description="Involuntary break in phonation (emotion, strain)",
        dsp_detectable=True, min_duration_ms=30,
        detection_method="dsp_energy_drop",
    ),
    "creak": VocalEventDef(
        tag="[creak]", category="phonation",
        description="Vocal fry / creaky voice onset",
        dsp_detectable=True, min_duration_ms=50,
        detection_method="dsp_subharmonic",
    ),
    "falsetto": VocalEventDef(
        tag="[falsetto]", category="phonation",
        description="Sudden shift to head voice register",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_pitch_jump",
    ),
    "tremor": VocalEventDef(
        tag="[tremor]", category="phonation",
        description="Voice trembling/shaking (nervousness, crying)",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_pitch_modulation",
    ),
    "strained": VocalEventDef(
        tag="[strained]", category="phonation",
        description="Effortful/tense phonation",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_spectral_tilt_shift",
    ),

    # --- 3. Emotional vocalizations ---
    "laugh": VocalEventDef(
        tag="[laugh]", category="emotional",
        description="Laughter (full laugh, not just smile in voice)",
        dsp_detectable=True, min_duration_ms=300,
        detection_method="dsp_periodic_bursts",
    ),
    "chuckle": VocalEventDef(
        tag="[chuckle]", category="emotional",
        description="Brief, quiet laugh (1-2 bursts)",
        dsp_detectable=True, min_duration_ms=150,
        detection_method="dsp_periodic_bursts_short",
    ),
    "sob": VocalEventDef(
        tag="[sob]", category="emotional",
        description="Sobbing / convulsive crying",
        dsp_detectable=True, min_duration_ms=300,
        detection_method="dsp_irregular_energy_pitch",
    ),
    "whimper": VocalEventDef(
        tag="[whimper]", category="emotional",
        description="Soft crying / pained vocalization",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_low_energy_pitch_instability",
    ),
    "sigh": VocalEventDef(
        tag="[sigh]", category="emotional",
        description="Audible sigh (resignation, relief, frustration)",
        dsp_detectable=True, min_duration_ms=300,
        detection_method="dsp_long_exhale_with_phonation",
    ),
    "groan": VocalEventDef(
        tag="[groan]", category="emotional",
        description="Low-pitched vocalization of pain or displeasure",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_low_pitch_sustained",
    ),
    "scream": VocalEventDef(
        tag="[scream]", category="emotional",
        description="High-energy, high-pitch vocalization",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_high_energy_high_pitch",
    ),
    "exclaim": VocalEventDef(
        tag="[exclaim]", category="emotional",
        description="Short vocal burst of surprise/emphasis",
        dsp_detectable=True, min_duration_ms=50,
        detection_method="dsp_energy_spike_voiced",
    ),

    # --- 4. Articulatory ---
    "click": VocalEventDef(
        tag="[click]", category="articulatory",
        description="Tongue click (tsk, tch)",
        dsp_detectable=True, min_duration_ms=20,
        detection_method="dsp_transient",
    ),
    "lip_smack": VocalEventDef(
        tag="[lip_smack]", category="articulatory",
        description="Lip smacking sound",
        dsp_detectable=True, min_duration_ms=30,
        detection_method="dsp_transient",
    ),
    "swallow": VocalEventDef(
        tag="[swallow]", category="articulatory",
        description="Audible swallowing",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_low_energy_transient",
    ),
    "cough": VocalEventDef(
        tag="[cough]", category="articulatory",
        description="Cough",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_broadband_burst",
    ),
    "throat_clear": VocalEventDef(
        tag="[throat_clear]", category="articulatory",
        description="Throat clearing",
        dsp_detectable=True, min_duration_ms=150,
        detection_method="dsp_broadband_burst",
    ),
    "sniff": VocalEventDef(
        tag="[sniff]", category="articulatory",
        description="Nasal sniffing (crying, cold)",
        dsp_detectable=True, min_duration_ms=100,
        detection_method="dsp_nasal_noise",
    ),

    # --- 5. Prosodic ---
    "pause": VocalEventDef(
        tag="[pause]", category="prosodic",
        description="Deliberate silence between phrases",
        dsp_detectable=True, min_duration_ms=300,
        detection_method="dsp_silence",
    ),
    "long_pause": VocalEventDef(
        tag="[long_pause]", category="prosodic",
        description="Extended silence (dramatic effect, hesitation)",
        dsp_detectable=True, min_duration_ms=800,
        detection_method="dsp_silence",
    ),
    "hesitation": VocalEventDef(
        tag="[hesitation]", category="prosodic",
        description="Filled pause (えー, あの, um)",
        dsp_detectable=False, min_duration_ms=200,
        detection_method="llm_infer",  # needs transcript context
    ),
    "emphasis": VocalEventDef(
        tag="[emphasis]", category="prosodic",
        description="Stressed/emphasized word or syllable",
        dsp_detectable=True, min_duration_ms=50,
        detection_method="dsp_energy_pitch_peak",
    ),
    "prolonged": VocalEventDef(
        tag="[prolonged]", category="prosodic",
        description="Deliberately lengthened vowel/consonant",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_sustained_voicing",
    ),
    "rush": VocalEventDef(
        tag="[rush]", category="prosodic",
        description="Rapid speech segment",
        dsp_detectable=True, min_duration_ms=300,
        detection_method="dsp_high_syllable_rate",
    ),

    # --- 6. Paralinguistic ---
    "hmm": VocalEventDef(
        tag="[hmm]", category="paralinguistic",
        description="Thinking/considering vocalization",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_nasal_sustained_low",
    ),
    "uh_huh": VocalEventDef(
        tag="[uh_huh]", category="paralinguistic",
        description="Agreement/acknowledgment vocalization",
        dsp_detectable=False, min_duration_ms=150,
        detection_method="llm_infer",
    ),
    "tsk": VocalEventDef(
        tag="[tsk]", category="paralinguistic",
        description="Disapproval click",
        dsp_detectable=True, min_duration_ms=30,
        detection_method="dsp_transient",
    ),
    "shh": VocalEventDef(
        tag="[shh]", category="paralinguistic",
        description="Shushing / hushing sound",
        dsp_detectable=True, min_duration_ms=200,
        detection_method="dsp_fricative_sustained",
    ),
}


# Convenience groupings
RESPIRATORY_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.category == "respiratory"}
PHONATION_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.category == "phonation"}
EMOTIONAL_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.category == "emotional"}
ARTICULATORY_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.category == "articulatory"}
PROSODIC_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.category == "prosodic"}
PARALINGUISTIC_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.category == "paralinguistic"}

DSP_DETECTABLE_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if v.dsp_detectable}
LLM_INFERRED_EVENTS = {k: v for k, v in VOCAL_EVENTS.items() if not v.dsp_detectable}

# All event tags for text encoder vocabulary
ALL_VOCAL_EVENT_TAGS: Tuple[str, ...] = tuple(v.tag for v in VOCAL_EVENTS.values())

# Summary
N_VOCAL_EVENTS = len(VOCAL_EVENTS)
N_DSP_DETECTABLE = len(DSP_DETECTABLE_EVENTS)
N_LLM_INFERRED = len(LLM_INFERRED_EVENTS)
N_CATEGORIES = len(set(v.category for v in VOCAL_EVENTS.values()))
