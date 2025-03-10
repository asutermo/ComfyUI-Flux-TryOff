import json
import os

import comfy.model_patcher
import comfy.sample
import comfy.sd
import comfy.utils
import numpy as np  # type: ignore
import torch  # type: ignore
from diffusers import (  # type: ignore
    AutoencoderTiny,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    FluxTransformer2DModel,
)
from diffusers.utils import load_image
from PIL import Image
from torchvision import transforms  # type: ignore
from transformers import (  # type: ignore
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

__all__ = [
    "TryOffFluxFillModelNode",
    "TryOffRunNode",
    "TryOffQuantizerNode",
    "FluxFillPipelineNode",
    "TryOnOffModelNode",
    "TryOnRunNode",
    "TryOnOffRunNode",
]

device_list = ["cuda", "cpu"]
node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_dir = os.path.abspath(os.path.join(node_dir, "..", ".."))
models_dir = os.path.abspath(os.path.join(comfy_dir, "models"))
checkpoints_dir = os.path.abspath(os.path.join(models_dir, "checkpoints"))
encoders_dir = os.path.abspath(os.path.join(models_dir, "text_encoders"))
vae_dir = os.path.abspath(os.path.join(models_dir, "vae"))

DTYPE = torch.bfloat16


class TryOffQuantizerNode:
    """Enable quantization to load heavier models"""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "quantizer": (["None", "8Bit", "4Bit"],),
            }
        }

    CATEGORY = "Quantize"
    RETURN_TYPES = (
        "transformers_config",
        "diffusers_config",
    )
    FUNCTION = "make_config"

    def make_config(self, quantizer):
        if quantizer == "8Bit":
            return (
                TransformersBitsAndBytesConfig(load_in_8bit=True),
                DiffusersBitsAndBytesConfig(load_in_8bit=True),
            )
        elif quantizer == "4Bit":
            return (
                TransformersBitsAndBytesConfig(load_in_4bit=True),
                DiffusersBitsAndBytesConfig(load_in_4bit=True),
            )
        else:
            return (None, None)


class TryOnOffImagePrepNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "try_on": ("BOOLEAN",),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 1024, "step": 16},
                ),
            },
            "optional": {
                "garment_in": ("IMAGE",),
            },
        }

    CATEGORY = "Utility"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("pixels", "mask")
    FUNCTION = "initialize_tensors"

    def initialize_tensors(self, image_in, mask_in, try_on, width, height, garment_in=None):
        # ComfyUI format: image_in is [batch, height, width, channels]
        # mask_in is [batch, height, width]
        batch_size = image_in.shape[0]
        import torch.nn.functional as F

        
        # Debug info
        print(f"Input shapes - Image: {image_in.shape}, Mask: {mask_in.shape}")
        
        # Check for garment when try_on is True
        if try_on and garment_in is None:
            raise ValueError("Garment input is required when try_on is True")
        
        # Create output tensors
        output_images = []
        output_masks = []
        
        for b in range(batch_size):
            # Extract single images from batch
            img = image_in[b].float()  # [height, width, channels]
            mask = mask_in[b].float()  # [height, width]
            
            # Resize using F.interpolate (needs [batch, channels, height, width] format temporarily)
            img_resized = F.interpolate(
                img.permute(2, 0, 1).unsqueeze(0),  # [1, channels, height, width]
                size=(height, width), 
                mode='bilinear'
            ).squeeze(0).permute(1, 2, 0)  # Back to [height, width, channels]
            
            mask_resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),  # [1, 1, height, width]
                size=(height, width), 
                mode='bilinear'
            ).squeeze(0).squeeze(0)  # Back to [height, width]
            
            # Process garment if provided
            if try_on and garment_in is not None:
                garment = garment_in[b].float()  # [height, width, channels]
                garment_resized = F.interpolate(
                    garment.permute(2, 0, 1).unsqueeze(0),  # [1, channels, height, width]
                    size=(height, width), 
                    mode='bilinear'
                ).squeeze(0).permute(1, 2, 0)  # Back to [height, width, channels]
            else:
                # Create empty garment if none provided
                garment_resized = torch.zeros((height, width, 3), dtype=torch.float32, device=img.device)
            
            # Apply mask to image in try_off mode
            if not try_on:
                # Expand mask to have same dimensions as img
                mask_expanded = mask_resized.unsqueeze(-1).expand(-1, -1, 3)
                img_resized = img_resized * mask_expanded
            
            # Create side-by-side image (garment | image)
            combined_width = width * 2
            combined_img = torch.zeros((height, combined_width, 3), dtype=torch.float32, device=img.device)
            combined_img[:, :width, :] = garment_resized
            combined_img[:, width:, :] = img_resized
            
            # Create appropriate mask (in ComfyUI format: [height, width])
            combined_mask = torch.zeros((height, combined_width), dtype=torch.float32, device=mask.device)
            
            if try_on:
                # For try-on, mask applies to the right side (person image)
                combined_mask[:, width:] = mask_resized
            else:
                # For try-off, invert mask logic
                combined_mask[:, :width] = 1.0 - mask_resized
                combined_mask[:, width:] = mask_resized
                
            # Add to batch results
            output_images.append(combined_img.unsqueeze(0))
            output_masks.append(combined_mask.unsqueeze(0))
        
        # Combine batch results
        final_image = torch.cat(output_images, dim=0)
        final_mask = torch.cat(output_masks, dim=0)
        
        print(f"Output shapes - Image: {final_image.shape}, Mask: {final_mask.shape}")
        
        return (final_image, final_mask)


class TryOnOffModelNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "model_name": (
                    [
                        "xiaozaa/cat-tryoff-flux",
                        "xiaozaa/catvton-flux-beta",
                        "xiaozaa/catvton-flux-alpha",
                    ],
                ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
            },
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_name, weight_dtype):
        from huggingface_hub import hf_hub_download
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
       
        # Set the appropriate torch dtype
        if weight_dtype == "fp8_e4m3fn":
            dtype = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            dtype = torch.float8_e5m2
        elif weight_dtype == "default":
            dtype = torch.float16  # Use float16 as default for CUDA

        # First download the model from HuggingFace if it doesn't exist locally
        local_model_path = os.path.join(checkpoints_dir, model_name.replace("/", "_"))
        os.makedirs(local_model_path, exist_ok=True)
        
        # Download model files
        print(f"Checking/downloading model: {model_name} to {local_model_path}")
        
        try:
            # Try to download HF config and model files
            config_file = hf_hub_download(repo_id=model_name, filename=CONFIG_NAME)
            print(f"Downloaded config from {model_name}")
            
            # Get list of available model files 
            from huggingface_hub import list_repo_files
            model_files = [f for f in list_repo_files(model_name) if f.endswith('.safetensors') or f.endswith('.bin')]
            
            for model_file in model_files:
                local_file = os.path.join(local_model_path, os.path.basename(model_file))
                if not os.path.exists(local_file):
                    print(f"Downloading {model_file}...")
                    hf_hub_download(repo_id=model_name, filename=model_file, local_dir=local_model_path)
            
            print(f"Model files downloaded to {local_model_path}")
        except Exception as e:
            print(f"Error downloading model from HF Hub: {e}")
            # Even if download fails, try to proceed with local files if they exist
      
            
        # Try to load model
        try:
            # Check if the model has safetensors files
            safetensors_files = [f for f in os.listdir(local_model_path) if f.endswith('.safetensors')]
            
            if safetensors_files:
                # If multiple safetensors files, load and merge them
                if len(safetensors_files) > 1:
                    print(f"Found multiple safetensors files: {safetensors_files}, merging...")
                    
                    # Sort the files to ensure consistent ordering
                    safetensors_files.sort()
                    
                    # Combine all state dicts
                    state_dict = {}
                    for sf in safetensors_files:
                        sf_path = os.path.join(local_model_path, sf)
                        print(f"Loading part: {sf}")
                        part_dict = comfy.utils.load_torch_file(sf_path, safe_load=True)
                        state_dict.update(part_dict)
                        del part_dict
                        
                    # Load the combined state dict
                    model = comfy.sd.load_diffusion_model_state_dict(
                        state_dict, 
                        model_options={"dtype": dtype}
                    )
                else:
                    # Single file, load directly
                    model_path = os.path.join(local_model_path, safetensors_files[0])
                    print(f"Loading single safetensors file: {model_path}")
                    model = comfy.sd.load_diffusion_model(
                        model_path,
                        model_options={"dtype": dtype}
                    )
            else:
                # Check for .bin files (PyTorch format)
                bin_files = [f for f in os.listdir(local_model_path) if f.endswith('.bin')]
                
                if bin_files:
                    if len(bin_files) > 1:
                        print(f"Found multiple .bin files: {bin_files}, merging...")
                        
                        # Sort the files
                        bin_files.sort()
                        
                        # Combine all state dicts
                        state_dict = {}
                        for bf in bin_files:
                            bf_path = os.path.join(local_model_path, bf)
                            print(f"Loading part: {bf}")
                            part_dict = torch.load(bf_path, map_location="cpu")
                            state_dict.update(part_dict)
                            del part_dict
                            
                        # Load the combined state dict
                        model = comfy.sd.load_diffusion_model_state_dict(
                            state_dict, 
                            model_options={"dtype": dtype}
                        )
                    else:
                        # Single file, load directly
                        model_path = os.path.join(local_model_path, bin_files[0])
                        print(f"Loading single .bin file: {model_path}")
                        model = comfy.sd.load_diffusion_model(
                            model_path,
                            model_options={"dtype": dtype}
                        )
                else:
                    # Fallback: if no model files found locally, try to use FluxTransformer2DModel 
                    # as a last resort to get the model
                    print("No local model files found, falling back to FluxTransformer2DModel")
                    from diffusers import FluxTransformer2DModel

                    model = FluxTransformer2DModel.from_pretrained(
                        model_name, 
                        cache_dir=checkpoints_dir, 
                        torch_dtype=dtype
                    )
                    
                    # Convert to ComfyUI format if needed
                    if hasattr(model, "state_dict"):
                        print("Converting FluxTransformer2DModel to ComfyUI format")
                        state_dict = model.state_dict()
                        model = comfy.sd.load_diffusion_model_state_dict(
                            state_dict,
                            model_options={"dtype": dtype}
                        )
           
            
            return (model,)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Last resort fallback - try direct loading with diffusers
            try:
                from diffusers import FluxTransformer2DModel
                
                print("Falling back to direct diffusers loading")
                model = FluxTransformer2DModel.from_pretrained(
                    model_name, 
                    cache_dir=checkpoints_dir, 
                    torch_dtype=dtype
                )
                    
                return (model,)
            except Exception as e2:
                print(f"All loading methods failed: {e2}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}, {e2}")


# FluxFillModel Node
class TryOffFluxFillModelNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "transformer": ("MODEL",),
                "model_name": (["FLUX.1-dev"],),
                "device": (device_list,),
            },
            "optional": {"diffusers_config": ("diffusers_config",)},
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_pipeline"

    def load_pipeline(self, transformer, model_name, device, diffusers_config=None):
        model_path = os.path.join(checkpoints_dir, model_name)

        if diffusers_config:
            pipeline = FluxFillPipeline.from_pretrained(
                model_path,
                transformer=transformer,
                torch_dtype=DTYPE,
                quantization_config=diffusers_config,
                device_map="balanced",
            )
        else:
            pipeline = FluxFillPipeline.from_pretrained(
                model_path,
                transformer=transformer,
                torch_dtype=DTYPE,
            ).to(device)

            pipeline.enable_model_cpu_offload()
            pipeline.transformer.to(DTYPE)

        return (pipeline,)


class FluxFillPipelineNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "transformer": ("MODEL",),
                "device": (device_list,),
            },
            "optional": {
                "transformers_config": ("transformers_config",),
                "diffusers_config": ("diffusers_config",),
            },
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_pipeline"

    def load_pipeline(
        self, transformer, device, transformers_config=None, diffusers_config=None
    ):
        if transformers_config:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
                quantization_config=transformers_config,
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
                quantization_config=transformers_config,
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
                quantization_config=transformers_config,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
                quantization_config=transformers_config,
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=DTYPE,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        if diffusers_config:
            vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taef1",
                cache_dir=vae_dir,
                torch_dtype=DTYPE,
                quantization_config=diffusers_config,
            )
            pipeline = FluxFillPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
            )
        else:
            vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taef1", cache_dir=vae_dir, torch_dtype=DTYPE
            )
            pipeline = FluxFillPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                transformer=transformer,
            )
            pipeline.to(device)
        pipeline.enable_model_cpu_offload()

        return (pipeline,)


def tryon_off_inference(
    pipe,
    image_in,
    mask_in,
    try_on: bool,
    garment_in,
    prompt: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    width: int,
    height: int,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Resize and preprocess
    def convert_image(tnsr):
        return Image.fromarray(
            np.clip(255.0 * tnsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        ).convert("RGB")

    image = convert_image(image_in).resize((width, height))
    mask = convert_image(mask_in).resize((width, height))

    if try_on:
        garment = convert_image(garment_in).resize((width, height))

    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]  # Take only first channel
    if try_on:
        garment_tensor = transform(garment)
    else:
        garment_tensor = torch.zeros_like(image_tensor)
        image_tensor = image_tensor * mask_tensor

    # Create concatenated images
    inpaint_image = torch.cat(
        [garment_tensor, image_tensor], dim=2
    )  # Concatenate along width
    garment_mask = torch.zeros_like(mask_tensor)

    if try_on:
        extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
    else:
        extended_mask = torch.cat([1 - garment_mask, garment_mask], dim=2)

    # Run pipeline
    result = pipe(
        height=height,
        width=width * 2,
        image=inpaint_image,
        mask_image=extended_mask,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
        max_sequence_length=512,
        guidance_scale=guidance_scale,
        prompt=prompt,
    ).images[0]

    # Split result into garment and try-on images
    garment_result = result.crop((0, 0, width, height))
    try_result = result.crop((width, 0, width * 2, height))

    try_result = torch.tensor(
        np.array(try_result) / 255.0, dtype=torch.float32
    ).unsqueeze(0)
    garment_result = torch.tensor(
        np.array(garment_result) / 255.0, dtype=torch.float32
    ).unsqueeze(0)

    return (
        try_result,
        garment_result,
    )


# TryOffRun Node
class TryOffRunNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "pipe": ("MODEL",),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 1024, "step": 16},
                ),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5},
                ),
                "seed": ("INT", {"default": 42}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryoff_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(
        self,
        image_in,
        mask_in,
        pipe,
        width,
        height,
        num_steps,
        guidance_scale,
        seed,
        prompt,
    ):
        return tryon_off_inference(
            pipe,
            image_in,
            mask_in,
            False,
            None,
            prompt,
            num_steps,
            guidance_scale,
            seed,
            width,
            height,
        )


class TryOnRunNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "image_in": ("IMAGE",),
                "garment_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "pipe": ("MODEL",),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 1024, "step": 16},
                ),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5},
                ),
                "seed": ("INT", {"default": 42}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryon_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(
        self,
        image_in,
        garment_in,
        mask_in,
        pipe,
        width,
        height,
        num_steps,
        guidance_scale,
        seed,
        prompt,
    ):

        return tryon_off_inference(
            pipe,
            image_in,
            mask_in,
            True,
            garment_in,
            prompt,
            num_steps,
            guidance_scale,
            seed,
            width,
            height,
        )


class TryOnOffRunNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "pipe": ("MODEL",),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 1024, "step": 16},
                ),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 30.0, "min": 1.0, "max": 100.0, "step": 0.5},
                ),
                "seed": ("INT", {"default": 42}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; "
                        "[IMAGE1] Detailed product shot of clothing "
                        "[IMAGE2] The same clothing is worn by a model in a lifestyle setting.",
                    },
                ),
            },
            "optional": {
                "garment_in": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryon_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(
        self,
        image_in,
        mask_in,
        pipe,
        width,
        height,
        num_steps,
        guidance_scale,
        seed,
        prompt,
        garment_in=None,
    ):
        # TODO: type checking
        if garment_in is not None:
            return tryon_off_inference(
                pipe,
                image_in,
                mask_in,
                True,
                garment_in,
                prompt,
                num_steps,
                guidance_scale,
                seed,
                width,
                height,
            )
        else:
            return tryon_off_inference(
                pipe,
                image_in,
                mask_in,
                False,
                None,
                prompt,
                num_steps,
                guidance_scale,
                seed,
                width,
                height,
            )

