import os
import sys
from pathlib import Path
from loguru import logger

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
DATA = ROOT / 'data'
SOLO = ROOT / "solovision"
TRACKING = ROOT / "tracking"
TRACKER_CONFIGS = ROOT / "solovision" / "configs"
WEIGHTS = ROOT / "tracking" / "weights"
REQUIREMENTS = ROOT / "requirements.txt"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  




logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")
