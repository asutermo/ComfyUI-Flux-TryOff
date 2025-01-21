import torch
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
import numpy as np

device = comfy.model_management.get_torch_device()

# TryOffModel Node
class TryOffModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["xiaozaa/cat-tryoff-flux"],),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_name):
        model = FluxTransformer2DModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        return (model,)


# FluxFillModel Node
class FluxFillModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("MODEL",),
                "pipeline_name": (["black-forest-labs/FLUX.1-dev"],),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_pipeline"

    def load_pipeline(self, transformer, pipeline_name):
        
        pipeline = FluxFillPipeline.from_pretrained(
            pipeline_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(device)
        return (pipeline,)


# TryOffRun Node
class TryOffRunNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "pipe": ("MODEL",),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": ("INT", {"default": 768, "min": 128, "max": 1024, "step": 16}),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "seed": ("INT", {"default": 42}),
                "prompt": ("STRING", {"default": 
                        "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryon_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(self, image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt):
        # Ensure input images are PIL images
        if isinstance(image_in, np.ndarray):
            image_in = Image.fromarray(image_in)
        if isinstance(mask_in, np.ndarray):
            mask_in = Image.fromarray(mask_in)

        pipe.transformer.to(torch.bfloat16)

        # Preprocessing transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Resize and preprocess
        image = image_in.convert("RGB").resize((width, height))
        mask = mask_in.convert("RGB").resize((width, height))

        image_tensor = transform(image)
        mask_tensor = mask_transform(mask)[:1]  # Use only the first channel

        garment_tensor = torch.zeros_like(image_tensor)
        image_tensor = image_tensor * mask_tensor

        # Concatenate inputs for FluxFillPipeline
        inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
        garment_mask = torch.zeros_like(mask_tensor)
        extended_mask = torch.cat([1 - garment_mask, garment_mask], dim=2)

        # # Define prompt
        # prompt = (
        #     "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
        #     "[IMAGE1] Detailed product shot of clothing "
        #     "[IMAGE2] The same clothing is worn by a model in a lifestyle setting."
        # )

        # Set random seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)

        # Run pipeline
        result = pipe(
            height=height,
            width=width * 2,
            image=inpaint_image,
            mask_image=extended_mask,
            num_inference_steps=num_steps,
            generator=generator,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            prompt=prompt,
        ).images[0]

        # Split result into garment and try-on images
        garment_result = result.crop((0, 0, width, height))
        tryoff_result = result.crop((width, 0, width * 2, height))

        return garment_result, tryoff_result


