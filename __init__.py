from .try_off_nodes import (  # noqa
    TryOnOffLoader,
    TryOnOffSampler,
    FluxFillPipelineNode,
    TryOffFluxFillModelNode,
    TryOffModelNode,
    TryOffQuantizerNode,
    TryOnOffModelNode,
    TryOnOffRunNode,
    TryOnOffRunNodeAdvanced,
)

# Node Registrations
NODE_CLASS_MAPPINGS = {
    "TryOffModelNode": TryOffModelNode,
    "TryOffFluxFillModelNode": TryOffFluxFillModelNode,
    "TryOffQuantizerNode": TryOffQuantizerNode,
    "TryOffFluxFillPipelineNode": FluxFillPipelineNode,
    "TryOnOffModelNode": TryOnOffModelNode,
    "TryOnOffRunNode": TryOnOffRunNode,
    "TryOnOffRunNodeAdvanced": TryOnOffRunNodeAdvanced,
    "TryOnOffLoader": TryOnOffLoader,
    "TryOnOffSampler": TryOnOffSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelNode": "TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillModelNode": "FluxFill Model Loader [ComfyUI-Flux-TryOff]",
    "TryOnOffRunNode": "Run TryOff Inference [ComfyUI-Flux-TryOff]",
    "TryOffQuantizerNode": "TryOff Quantizer [ComfyUI-Flux-TryOff]",
    "TryOffFluxFillPipelineNode": "FluxFill Pipeline Loader [ComfyUI-Flux-TryOff]",
    "TryOnOffModelNode": "TryOn or TryOff Model Loader [ComfyUI-Flux-TryOff]",
    "TryOnOffRunNode (Advanced)": "Run TryOff Inference (Advanced) [ComfyUI-Flux-TryOff]",
    "TryOnOffLoader": "Load CatVTON Model",
    "TryOnOffSampler": "CatVTON Sampler",
}
