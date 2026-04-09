from .baseline_model import build_baseline_model
from .resnet import resnet50 as build_npr_model


def build_detector(model_type="npr", num_classes=1):
    if model_type == "npr":
        return build_npr_model(num_classes=num_classes)
    if model_type == "baseline":
        return build_baseline_model(num_classes=num_classes)
    raise ValueError(f"Unsupported model_type: {model_type}")
