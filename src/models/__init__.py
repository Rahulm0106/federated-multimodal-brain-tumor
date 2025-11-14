# src/models/__init__.py
from .multimodal import MultimodalFLModel


def get_model(config):
    return MultimodalFLModel(config)