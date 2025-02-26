from .try_off_nodes import (  # noqa
    FluxFillPipelineNode,
    TryOffFluxFillModelNode,
    TryOffQuantizerNode,
    TryOffRunNode,
    TryOnOffModelNode,
    TryOnOffRunNode,
    TryOnRunNode,
)

# Node Registrations
NODE_CLASS_MAPPINGS = {
    "TryOffModelNode": TryOnOffModelNode,
    "TryOnOffModelNode": TryOnOffModelNode,
    "TryOffFluxFillModelNode": TryOffFluxFillModelNode,
    "TryOffRunNode": TryOffRunNode,
    "TryOffQuantizerNode": TryOffQuantizerNode,
    "TryOffFluxFillPipelineNode": FluxFillPipelineNode,
    "TryOnRunNode": TryOnRunNode,
    "TryOnOffRunNode": TryOnOffRunNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelNode": "TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOnOffModelNode": "TryOnOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillModelNode": "FluxFill Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffRunNode": "Run TryOff Inference [ComfyUI-Flux-TryOff]",
    "TryOffQuantizerNode": "TryOff Quantizer [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillPipelineNode": "FluxFill Pipeline Loader [ComfyUI-Flux-TryOff]",
    "TryOnRunNode": "Run TryOn Inference [ComfyUI-Flux-TryOff]",
    "TryOnOffRunNode": "Run TryOn or TryOff Inference [ComfyUI-Flux-TryOff]",
}
