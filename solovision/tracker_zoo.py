import yaml
from solovision.utils import TRACKER_CONFIGS
from solovision.trackers.bytetrack.bytetracker import ByteTracker


def get_tracker_config(tracker_type):
    """Returns the path to the tracker configuration file."""
    return TRACKER_CONFIGS / f'{tracker_type}.yaml'

def create_tracker(tracker_config=None, reid_weights=None, device=None, half=None, per_class=None, evolve_param_dict=None):
    """
    Creates and returns an instance of the specified tracker type.
    
    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification).
    - device: Device to run the tracker on (e.g., 'cpu', 'cuda').
    - half: Boolean indicating whether to use half-precision.
    - per_class: Boolean for class-specific tracking (optional).
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.
    
    Returns:
    - An instance of the selected tracker.
    """
    
    # Load configuration from file or use provided dictionary
    if evolve_param_dict is None:
        with open(tracker_config, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {param: details['default'] for param, details in yaml_config.items()}
    else:
        tracker_args = evolve_param_dict

    # Arguments specific to ReID models
    reid_args = {
        'reid_weights': reid_weights,
        'device': device,
        'half': half,
    }

    tracker_class = ByteTracker
    
    tracker_args['per_class'] = per_class
    tracker_args.update(reid_args)
    
    # Return the instantiated tracker class with arguments
    return tracker_class(**tracker_args)