__version__ = '0.0.1'

from solovision.utils.checks import RequirementsChecker
# Check requirements
checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/AIEngineersDev/solo-ultralytics.git', )) 

from solovision.tracker_zoo import create_tracker, get_tracker_config
from solovision.trackers.bytetrack.bytetracker import ByteTracker

__all__ = ("__version__",
            "ByteTracker", "create_tracker", "get_tracker_config")
