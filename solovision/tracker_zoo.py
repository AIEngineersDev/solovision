import yaml
from solovision.utils import TRACKER_CONFIGS

def get_tracker_config(tracker):
    """Returns the path to the tracker configuration file."""
    return TRACKER_CONFIGS / f'{tracker}.yaml'

def create_tracker(tracker=None, tracker_config=None, with_reid=True, reid_weights=None, device=None, half=None, per_class=None, evolve_param_dict=None):
    """
    Creates and returns an instance of the specified tracker type.

    Parameters:
    - tracker_config: Path to the tracker configuration file.
    - with_reid: Boolean indicating whether to use ReID features (default: True)
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
        print('Got here')
        print(tracker_config)
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
        'with_reid': with_reid
    }

    # Map tracker types to their corresponding classes
    tracker_mapping = {
        'bytetrack': 'solovision.trackers.bytetrack.bytetrack.ByteTrack',
        'hybridsort': 'solovision.trackers.hybridsort.hybridsort.HybridSort',
    }

    module_path, class_name = tracker_mapping[tracker].rsplit('.', 1)
    tracker_class = getattr(__import__(module_path, fromlist=[class_name]), class_name)
    tracker_args['per_class'] = per_class
    tracker_args.update(reid_args)
    
    # Return the instantiated tracker class with arguments
    return tracker_class(**tracker_args)