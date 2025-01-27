import os
import torch
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
import numpy as np


device_list = ['cuda', 'cpu']
node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_dir = os.path.abspath(os.path.join(node_dir, '..', '..'))
models_dir = os.path.abspath(os.path.join(comfy_dir, 'models'))
checkpoints_dir = os.path.abspath(os.path.join(models_dir, 'checkpoints'))
diffusers_dir = os.path.abspath(os.path.join(models_dir, 'diffusion_models'))
encoders_dir = os.path.abspath(os.path.join(models_dir, 'text_encoders'))
clip_dir = os.path.abspath(os.path.join(models_dir, 'clip_vision'))
vae_dir = os.path.abspath(os.path.join(models_dir, 'vae'))

class TryOffHuggingFaceTokenNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING",),
            }
        }
    
    CATEGORY = "Secrets"
    FUNCTION = "get_hf_token"

    def get_hf_token(self, hf_token: str) -> None:
        os.environ.set("HF_TOKEN", hf_token)
        os.environ.set("HUGGING_FACE_HUB_TOKEN", hf_token)

# Try On Model Node
class TryOnModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["xiaozaa/catvton-flux-beta"],),
                "device": (device_list,),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_name, device):
        model = FluxTransformer2DModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        return (model,)
    
# TryOffModel Node
class TryOffModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["xiaozaa/cat-tryoff-flux"],),
                "device": (device_list,),
                "cpu_offload": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_name, device, cpu_offload):
        model = FluxTransformer2DModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
        if cpu_offload:
            model.enable_model_cpu_offload()
        return (model,)

# FluxFillModel Node
class TryOffFluxFillModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("MODEL",),
                "model_name": (["FLUX.1-dev"],),
                "device": (device_list,),
                "cpu_offload": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_pipeline"

    def load_pipeline(self, transformer, model_name, device, cpu_offload):
        model_path = os.path.join(checkpoints_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")
        pipeline = FluxFillPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(device)

        if cpu_offload:
            pipeline.enable_model_cpu_offload()
        return (pipeline,)


def inference(image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device, try_on: bool = False):
    print(type(pipe.transformer))
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
    def convert_image(tnsr):
        return Image.fromarray(np.clip(255.0 * tnsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGB")

    image = convert_image(image_in).resize((width, height))
    mask = convert_image(mask_in).resize((width, height))

    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]  # Use only the first channel

    garment_tensor = torch.zeros_like(image_tensor)
    image_tensor = image_tensor * mask_tensor

    # Concatenate inputs for FluxFillPipeline
    inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
    garment_mask = torch.zeros_like(mask_tensor)

    if try_on:
        extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
    else:
        extended_mask = torch.cat([1 - garment_mask, garment_mask], dim=2)

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

    tryoff_result = torch.tensor(
                np.array(tryoff_result) / 255.0, dtype=torch.float32
            ).unsqueeze(0)
    garment_result = torch.tensor(
                np.array(garment_result) / 255.0, dtype=torch.float32
            ).unsqueeze(0)

    return  (tryoff_result, garment_result,)

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
                "prompt": ("STRING", {"multiline": True, "default": 
                        "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting."}),
                "device": (device_list,),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryoff_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(self, image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device):
        return inference(image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device, try_on=False)


# TryOffRun Node
class TryOnRunNode:
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
                "prompt": ("STRING", {"multiline": True, "default": 
                        "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting."}),
                "device": (device_list,),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryon_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(self, image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device):
        return inference(image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device, try_on=True)

class TryOnOffRunNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "pipe": ("MODEL",),
                "try_on": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": ("INT", {"default": 768, "min": 128, "max": 1024, "step": 16}),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "seed": ("INT", {"default": 42}),
                "prompt": ("STRING", {"multiline": True, "default": 
                        "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting."}),
                "device": (device_list,),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryonoff_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(self, image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device, try_on):
        return inference(image_in, mask_in, pipe, width, height, num_steps, guidance_scale, seed, prompt, device, try_on)
