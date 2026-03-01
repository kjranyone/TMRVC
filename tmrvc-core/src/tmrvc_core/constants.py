"""Universal TMRVC UCLM v2 Constants.

This module re-exports constants from _generated_constants.py (auto-generated from YAML)
and adds backward compatibility aliases.

Single source of truth: configs/constants.yaml
Run: python scripts/codegen/generate_constants.py
"""

# Auto-generated from YAML - DO NOT EDIT THESE VALUES DIRECTLY
from tmrvc_core._generated_constants import *  # noqa: F401, F403

# Backward compatibility aliases (deprecated - will be removed)
# TODO: Update all imports to use new names, then remove these
D_CONTENT = D_MODEL  # Was: 512 -> use D_MODEL
D_CONTENT_VEC = 768  # ContentVec dim - not used in UCLM v2
D_WAVLM_LARGE = 1024  # WavLM large dim - not used in UCLM v2
VOCAB_SIZE = RVQ_VOCAB_SIZE  # Use RVQ_VOCAB_SIZE instead
TOKENIZER_VOCAB_SIZE = 256  # Not used in UCLM v2
