from diffusers import FluxPriorReduxPipeline, FluxFillPipeline
from diffusers import FluxTransformer2DModel
from torchvision import transforms
import torch 

try_off_model_list = [
    "xiaozaa/cat-tryoff-flux"
]

class TryOffModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (try_off_model_list,),
            }
        }

    CATEGORY = "TryOff"
    FUNCTION = "main"
    RETURN_TYPES = ("TryOffModel",)

    def main(self, model_name):
        transformer = FluxTransformer2DModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16
        )
        return (transformer,)
     
class TryOffNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # checkpoint
            "required": { 
                "image_in" : ("IMAGE", {}),
                "try_off_pipe": ("TryOffModel",),
                "model": ("MODEL",),
                "mask_in": ("MASK", {})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    CATEGORY = "TryOff"
    FUNCTION = "tryoff"

NODE_CLASS_MAPPINGS = {
    "TryOffModelLoader": TryOffModelLoader,
    "TryOffNode": TryOffNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffModelLoader": "Load TryOff Model",
    "TryOffNode": "Apply TryOff on Image",
}