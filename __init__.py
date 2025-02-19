from .try_off_nodes import (  # noqa
    FluxFillPipelineNode,
    TryOffFluxFillModelNode,
    TryOffModelNode,
    TryOffQuantizerNode,
    TryOffRunNode,
    TryOnOffModelNode,
    FluxFillPipelineNode2
)

# Node Registrations
NODE_CLASS_MAPPINGS = {
    "TryOffModelNode": TryOffModelNode,
    "TryOffFluxFillModelNode": TryOffFluxFillModelNode,
    "TryOffRunNode": TryOffRunNode,
    "TryOffQuantizerNode": TryOffQuantizerNode,
    "TryOffFluxFillPipelineNode": FluxFillPipelineNode,
    "TryOnOffModelNode": TryOnOffModelNode,
    "TryOffFluxFillPipelineNode2": FluxFillPipelineNode2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelNode": "TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillModelNode": "FluxFill Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffRunNode": "Run TryOff Inference [ComfyUI-Flux-TryOff]",
    "TryOffQuantizerNode": "TryOff Quantizer [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillPipelineNode": "FluxFill Pipeline Loader [ComfyUI-Flux-TryOff]",
    "TryOnOffModelNode": "TryOn or TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillPipelineNode2": "FluxFill Pipeline Loader [ComfyUI-Flux-TryOff]",
}
