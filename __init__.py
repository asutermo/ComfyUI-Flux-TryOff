from .try_off_nodes import *

# Node Registrations
NODE_CLASS_MAPPINGS = {
    "TryOffModelNode": TryOffModelNode,
    "TryOffFluxFillModelNode": HFTryOffFluxFillModelNode,
    "TryOffFluxFillModelNode2": FluxFillModelNode2,
    "TryOffRunNode": TryOffRunNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelNode": "TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillModelNode": "HF FluxFill Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillModelNode2": "FluxFill Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffRunNode": "Run TryOff Inference [ComfyUI-Flux-TryOff]",
}

