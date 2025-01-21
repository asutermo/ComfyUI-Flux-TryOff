from .try_off_nodes import *

# Node Registrations
NODE_CLASS_MAPPINGS = {
    "TryOffModelNode": TryOffModelNode,
    "FluxFillModelNode": FluxFillModelNode,
    "TryOffRunNode": TryOffRunNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelNode": "TryOff Model Loader",
    "FluxFillModelNode": "FluxFill Model Loader",
    "TryOffRunNode": "Run TryOff Inference",
}

