from .try_off_nodes import *

# Node Registrations
NODE_CLASS_MAPPINGS = {
    "TryOffModelNode": TryOffModelNode,
    "TryOffFluxFillModelNode": TryOffFluxFillModelNode,
    "TryOffRunNode": TryOffRunNode,
    "TryOnModelNode": TryOnModelNode,
    "TryOnRunNode": TryOnRunNode,
    "TryOnOffRunNode": TryOnOffRunNode,
    "TryOffHuggingFaceTokenNode": TryOffHuggingFaceTokenNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelNode": "TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillModelNode": "FluxFill Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffRunNode": "Run TryOff Inference [ComfyUI-Flux-TryOff]",
    "TryOnModelNode": "TryOn Model Loader [ComfyUI-Flux-TryOff]",
    "TryOnRunNode": "Run TryOn Inference [ComfyUI-Flux-TryOff]",
    "TryOnOffRunNode": "Run TryOn or TryOff Inference [ComfyUI-Flux-TryOff]",
    "TryOffHuggingFaceTokenNode": "HuggingFace Token Setter [ComfyUI-Flux-TryOff]"
}

