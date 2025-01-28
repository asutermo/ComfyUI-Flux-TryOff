import os
import torch
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
import numpy as np

from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel, CLIPTextModel
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from folder_paths import models_dir, get_filename_list, get_full_path_or_raise, supported_pt_extensions

device_list = ['cuda', 'cpu']
node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_dir = os.path.abspath(os.path.join(node_dir, '..', '..'))
models_dir = os.path.abspath(os.path.join(comfy_dir, 'models'))
checkpoints_dir = os.path.abspath(os.path.join(models_dir, 'checkpoints'))
vae_dir = os.path.abspath(os.path.join(models_dir, 'vae'))    
text_encoders = os.path.abspath(os.path.join(models_dir, 'text_encoders'))    
dtype = torch.bfloat16

t_quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,)
d_quant_config = BitsAndBytesConfig(load_in_8bit=True)

# TryOffModel Node
class TryOffModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["xiaozaa/cat-tryoff-flux"],),
                "eight_bit_quantize": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_name, eight_bit_quantize):
        if eight_bit_quantize:
            model = FluxTransformer2DModel.from_pretrained(model_name, torch_dtype=dtype, quantization_config=t_quant_config)
        else:
            model = FluxTransformer2DModel.from_pretrained(model_name, torch_dtype=dtype)
        return (model,)

# add switches


class FluxFillModelNode2:
    @staticmethod
    def vae_list():
        return get_filename_list("vae")

    @staticmethod
    def te_list():
        return get_filename_list("text_encoders")
    
    MODEL_PATHS = ["checkpoints", "diffusion_models", "unet"]

    @staticmethod
    def model_list():
        paths = []
        [paths.extend(get_filename_list(path)) for path in FluxFillModelNode2.MODEL_PATHS]
        return paths
    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (FluxFillModelNode2.model_list(),),
                "transformer": ("MODEL",),
                "vae": (FluxFillModelNode2.vae_list(),),
                "eight_bit_quantize": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_pipeline"
    
    def load_pipeline(self, model, transformer, vae, eight_bit_quantize):
        if eight_bit_quantize:
            d_args = {"quantization_config": d_quant_config}
            t_args = {"quantization_config": t_quant_config}
        else:
            d_args = {}
            t_args = {}

        text_encoder_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype, **t_args)
        text_encoder_2_model = T5EncoderModel.from_pretrained("XLabs-AI/xflux_text_encoders", torch_dtype=dtype, **t_args)

        vae_path = os.path.join(vae_dir, vae)
        vae_model = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype, **d_args)

        for path in FluxFillModelNode2.MODEL_PATHS:
            model_path = os.path.join(models_dir, path, model)
            if model_path:
                break

        if not model_path:
            raise ValueError(f"Model {model} not found in {FluxFillModelNode2.MODEL_PATHS}")

        print(model_path, transformer,  vae_path)
        pipeline = FluxFillPipeline.from_single_file(
            model_path,
            transformer=transformer,
            text_encoder=text_encoder_model,
            text_encoder_2=text_encoder_2_model,
            vae=vae_model,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            **d_args
        )
        return (pipeline,)


# FluxFillModel Node
class HFTryOffFluxFillModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("MODEL",),
                "model_name": (["FLUX.1-dev"],),
            }
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_pipeline"

    def load_pipeline(self, transformer, model_name):
        model_path = os.path.join(checkpoints_dir, model_name)
        
        pipeline = FluxFillPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        pipeline.enable_model_cpu_offload()
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
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
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


