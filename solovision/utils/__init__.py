import os
import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
DATA = ROOT / 'data'
TRACKER_CONFIGS = ROOT / "configs"
WEIGHTS = ROOT  / "weights"
REQUIREMENTS = ROOT / "requirements.txt"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads


# global logger
from loguru import logger

logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")
