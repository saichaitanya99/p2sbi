"""P2SBI — Pulsar Profile SBI."""
__version__ = "0.1.0"

import sys as _sys
from . import pulsarDMvary as _pulsarDMvary
_sys.modules.setdefault("pulsarDMvary", _pulsarDMvary)   # posteriors saved before 0.1.0

from .pulsar_config import load_config, get_config, resolve_config_path, PulsarConfig
from .features import build_features
from .pulsarDMvary import PulsarProfileSimulator, build_encoder, sim_generator
