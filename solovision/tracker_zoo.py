from types import SimpleNamespace

import yaml

from solovision.utils import SOLO
from solovision.trackers.bytetrack.byte_tracker import BYTETracker

def get_tracker_config(tracker_type):
    tracking_config = \
        SOLO /\
        'configs' /\
        (tracker_type + '.yaml')
    return tracking_config


def create_tracker(tracker_config=None, per_class = None):
    # If config_dict is not provided, read from the file
    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg = SimpleNamespace(**cfg)  # easier dict access by dot, instead of ['']

    bytetracker = BYTETracker(
    per_class=per_class,
    track_thresh=cfg.track_thresh,
    match_thresh=cfg.match_thresh,
    track_buffer=cfg.track_buffer,
    frame_rate=cfg.frame_rate,
    sec_match_thresh = cfg.sec_match_thresh,
    min_hits = cfg.min_hits
    )
    return bytetracker

    
