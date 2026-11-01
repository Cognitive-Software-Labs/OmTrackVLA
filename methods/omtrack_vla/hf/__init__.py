#!/usr/bin/env python3
from typing import Optional

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

from .configuration import OmTrackVLAConfig
from .modeling import OmTrackVLAForWaypoint

__all__ = ["OmTrackVLAConfig", "OmTrackVLAForWaypoint"]


def _register_with_auto():
    model_type = OmTrackVLAConfig.model_type
    if model_type not in CONFIG_MAPPING:
        CONFIG_MAPPING.register(model_type, OmTrackVLAConfig)
    try:
        AutoConfig.register(model_type, OmTrackVLAConfig)
    except Exception:
        pass
    if OmTrackVLAConfig not in MODEL_MAPPING:
        MODEL_MAPPING.register(OmTrackVLAConfig, OmTrackVLAForWaypoint)
    try:
        AutoModel.register(OmTrackVLAConfig, OmTrackVLAForWaypoint)
    except Exception:
        pass


_register_with_auto()

