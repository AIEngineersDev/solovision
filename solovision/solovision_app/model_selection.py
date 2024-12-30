# common model size mappings
SIZES = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "XLarge"
}

YOLO_MODEL_SIZES = {
    "YOLOv3": {
        "-tiny": "Tiny",
        "": "Small",
        "-spp": "SPP"
    },
    "YOLOv5": SIZES,
    "YOLOv8": SIZES,
    "YOLOv9": {
        "t": "Tiny",
        "s": "Small",
        "m": "Medium",
        "c": "Large",
        "e": "XLarge"
    },
    "YOLOv10": SIZES | {"b": "Big"},
    "YOLO11": SIZES,
}

# common variant mappings
VARIANTS = {
    "": "Detection",
    "-cls": "Classification",
    "-seg": "Segmentation",
    "-pose": "Pose Estimation",
    "-obb": "Oriented Bounding Box"
}

YOLO_MODEL_VARIANTS = {
    "YOLOv3": {},
    "YOLOv5": {},
    "YOLOv8": VARIANTS,
    "YOLOv9": {
        "": "Detection",
        "-seg": "Segmentation"},
    "YOLOv10": {},
    "YOLO11": VARIANTS,
}

def get_model_sizes(yolo_version):
    """Returns available model sizes and their display names for a given YOLO version."""
    model_sizes = YOLO_MODEL_SIZES.get(yolo_version, {})
    return list(model_sizes.keys()), list(model_sizes.values())

def get_model_variants(yolo_version, size=None):
    """Returns available variants and their display names for a given YOLO version and size."""
    variants = YOLO_MODEL_VARIANTS.get(yolo_version, {})
    
    if yolo_version == "YOLOv9" and size not in ['c', 'e']:
        # Only 'c' and 'e' models support segmentation
        variants = {}
    
    return list(variants.keys()), list(variants.values())

def construct_model_path(yolo_version, size, variant=""):
    """Constructs the model path based on version, size, and variant."""
    if yolo_version == "YOLOv3":
        return f"yolov3{size}.pt"
    elif yolo_version == "YOLOv5":
        return f"yolov5{size}{variant}.pt"
    else:  # YOLOv8, YOLOv9, YOLOv10, YOLO11
        return f"{yolo_version.lower()}{size}{variant}.pt"

def get_available_yolo_versions():
    """Returns list of available YOLO versions."""
    return ["YOLOv3", "YOLOv5", "YOLOv8", "YOLOv9", "YOLOv10", "YOLO11", "Custom"]

def get_reid_models():
    """Returns list of available ReID models."""
    return [
        "osnet_x0_25_msmt17.pt",
        "osnet_x0_5_msmt17.pt",
        "osnet_x1_0_msmt17.pt",
        "osnet_ain_x1_0_msmt17.pt"
        "osnet_ibn_x1_0_msmt17.pt",
        "clip_market1501.pt"
    ] 