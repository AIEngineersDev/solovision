__version__ = '2.0.0'

from solovision.postprocessing.gsi import gsi
from solovision.tracker_zoo import create_tracker, get_tracker_config
from solovision.trackers.bytetrack.bytetracker import ByteTracker


__all__ = ("__version__",
            "ByteTracker", "create_tracker", "get_tracker_config", "gsi")
