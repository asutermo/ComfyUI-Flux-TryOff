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

dtype = torch.bfloat16


def load_flux_transformer(model_directory, transformer_parts="all", weight_dtype=dtype):
    if not os.path.exists(model_directory):
        raise ValueError(f"Model directory does not exist: {model_directory}")

    if os.path.exists(os.path.join(model_directory, "transformer")):
        transformer_dir = os.path.join(model_directory, "transformer")
    else:
        # Assume we're already in the transformer directory
        transformer_dir = model_directory

    print(f"Loading Flux transformer from: {transformer_dir}")

    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    part_files = [
        f
        for f in os.listdir(transformer_dir)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    ]

    if not part_files:
        raise ValueError(f"No model part files found in {transformer_dir}")

    part_files.sort()

    if transformer_parts == "all":
        # Combine all parts
        print(f"Loading and combining all parts: {part_files}")
        state_dict = {}
        for part_file in part_files:
            part_path = os.path.join(transformer_dir, part_file)
            print(f"Loading part: {part_file}")
            part_dict = comfy.utils.load_torch_file(part_path, safe_load=True)
            state_dict.update(part_dict)
            del part_dict

        # Load the combined state dict
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options
        )
    else:
        part_num = int(transformer_parts.split("_")[1])
        part_pattern = f"diffusion_pytorch_model-0000{part_num}-of-"
        matching_files = [f for f in part_files if part_pattern in f]

        if not matching_files:
            raise ValueError(f"No file found for part {part_num}")

        part_file = os.path.join(transformer_dir, matching_files[0])
        print(f"Loading specific part file: {part_file}")
        model = comfy.sd.load_diffusion_model(part_file, model_options=model_options)

    print("Flux transformer model loaded successfully")
    return model


def load_flux_model_for_try_on(model_path, device="cuda", weight_dtype="default"):
    # Set the appropriate torch dtype
    if weight_dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
    elif weight_dtype == "default":
        dtype = dtype if device == "cuda" else torch.float32

    # Check if model_path is a directory or a file
    if os.path.isdir(model_path):
        # It's a directory, use the load_flux_transformer function
        model = load_flux_transformer(
            model_path, transformer_parts="all", weight_dtype=weight_dtype
        )
    else:
        # It's a file, load directly
        if model_path.endswith(".safetensors"):
            state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
            model = comfy.sd.load_diffusion_model_state_dict(
                state_dict, model_options={"dtype": dtype}
            )
        else:
            model = comfy.sd.load_diffusion_model(
                model_path, model_options={"dtype": dtype}
            )

    # Move to the specified device
    model.to(device)
    print(f"Flux model loaded on {device} with {weight_dtype} precision")

    return model


def load_and_merge_flux_model(model_directory, device="cuda", weight_dtype="default"):
    """
    Load and merge a Flux model split across multiple files specifically for ComfyUI integration.

    Args:
        model_directory (str): Path to the directory containing the Flux model files
        device (str): Device to load the model on - "cuda" or "cpu"
        weight_dtype (str): Model precision - "default", "fp8_e4m3fn", or "fp8_e5m2"

    Returns:
        The loaded and merged Flux model
    """
    # Verify model directory exists
    if not os.path.exists(model_directory):
        raise ValueError(f"Model directory does not exist: {model_directory}")

    # Find the transformer directory
    if os.path.exists(os.path.join(model_directory, "transformer")):
        transformer_dir = os.path.join(model_directory, "transformer")
    else:
        transformer_dir = model_directory

    # Set dtype
    if weight_dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
    elif weight_dtype == "default":
        dtype = torch.float16 if device == "cuda" else torch.float32

    model_options = {"dtype": dtype}

    # Find all model part files
    part_files = [
        f
        for f in os.listdir(transformer_dir)
        if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
    ]

    if not part_files:
        raise ValueError(f"No model part files found in {transformer_dir}")

    # Sort the files to ensure they're loaded in the correct order
    part_files.sort()

    print(f"Found {len(part_files)} model parts: {part_files}")

    # Combine all parts into a single state dictionary
    state_dict = {}
    for part_file in part_files:
        part_path = os.path.join(transformer_dir, part_file)
        print(f"Loading part: {part_file}")
        try:
            part_dict = comfy.utils.load_torch_file(part_path, safe_load=True)
            state_dict.update(part_dict)
            del part_dict
            # Clear CUDA cache to avoid OOM
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading part {part_file}: {e}")
            raise

    # Load the combined state dictionary
    try:
        print("Creating model from combined state dictionary")
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options
        )

        # Move to device
        model.to(device)
        print(f"Successfully loaded and merged Flux model on {device}")

        return model
    except Exception as e:
        print(f"Error creating model from state dictionary: {e}")
        # Try with lower precision if OOM occurs
        if (
            "CUDA out of memory" in str(e)
            and weight_dtype == "default"
            and device == "cuda"
        ):
            print("Attempting to load with lower precision...")
            torch.cuda.empty_cache()
            model_options = {"dtype": torch.float16}  # Force fp16
            model = comfy.sd.load_diffusion_model_state_dict(
                state_dict, model_options=model_options
            )
            model.to(device)
            return model
        raise


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
                "device": (device_list,),
            },
            "optional": {"transformers_config": ("transformers_config",)},
        }

    CATEGORY = "Models"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"

    def load_model(self, model_name, device, transformers_config=None):
        if transformers_config:
            model = FluxTransformer2DModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                cache_dir=checkpoints_dir,
                quantization_config=transformers_config,
            )
        else:
            model = FluxTransformer2DModel.from_pretrained(
                model_name, cache_dir=checkpoints_dir, torch_dtype=dtype
            ).to(device)

        return (model,)


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
                torch_dtype=dtype,
                quantization_config=diffusers_config,
                device_map="balanced",
            )
        else:
            pipeline = FluxFillPipeline.from_pretrained(
                model_path,
                transformer=transformer,
                torch_dtype=dtype,
            ).to(device)

            pipeline.enable_model_cpu_offload()
            pipeline.transformer.to(dtype)

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
                torch_dtype=dtype,
                quantization_config=transformers_config,
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
                quantization_config=transformers_config,
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
                quantization_config=transformers_config,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
                quantization_config=transformers_config,
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
            )
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
            )
            text_encoder_2 = T5EncoderModel.from_pretrained(
                "XLabs-AI/xflux_text_encoders",
                cache_dir=encoders_dir,
                torch_dtype=dtype,
            )

        scheduler = FlowMatchEulerDiscreteScheduler()

        if diffusers_config:
            vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taef1",
                cache_dir=vae_dir,
                torch_dtype=dtype,
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
                "madebyollin/taef1", cache_dir=vae_dir, torch_dtype=dtype
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


def comfy_tryon_off_inference(
    model,  # The loaded Flux model
    clip,  # ComfyUI CLIP text encoder
    vae,  # ComfyUI VAE
    image_in,  # Input model image tensor
    mask_in,  # Input mask tensor
    try_on=True,  # Whether to do try-on (True) or try-off (False)
    garment_in=None,  # Garment image tensor (only needed for try-on)
    prompt="",  # Prompt for generation
    negative_prompt="",  # Negative prompt
    steps=50,  # Number of sampling steps
    cfg_scale=7.5,  # Guidance scale
    scheduler="euler",  # Sampling scheduler
    seed=42,  # Generation seed
    width=576,  # Image width
    height=768,  # Image height
    denoise_strength=1.0,  # Denoising strength (1.0 = full denoise)
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

    def convert_image(tnsr):
        return Image.fromarray(
            np.clip(255.0 * tnsr.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        ).convert("RGB")

    image = convert_image(image_in).resize((width, height))
    mask = convert_image(mask_in).resize((width, height))

    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]  # Take only first channel

    if try_on and garment_in is not None:
        garment = convert_image(garment_in).resize((width, height))
        garment_tensor = transform(garment)
    else:
        garment_tensor = torch.zeros_like(image_tensor)
        if not try_on:
            image_tensor = image_tensor * mask_tensor

    inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)

    # Create appropriate mask
    garment_mask = torch.zeros_like(mask_tensor)
    if try_on:
        extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
    else:
        extended_mask = torch.cat([1 - garment_mask, garment_mask], dim=2)

    # Convert to ComfyUI format
    inpaint_image = inpaint_image.unsqueeze(0)  # Add batch dimension
    extended_mask = extended_mask.unsqueeze(0)

    # Generate random noise from seed
    torch.manual_seed(seed)
    noise = torch.randn(
        (1, model.unet.config.in_channels, height // 8, width * 2 // 8),
        device=model.device,
        dtype=model.dtype,
    )

    # Convert images to latent space using VAE
    samples = vae.encode(inpaint_image)

    # Process conditioning using CLIP
    positive_cond = clip.encode(prompt)
    negative_cond = clip.encode(negative_prompt) if negative_prompt else None

    # Set up model patches (similar to ComfyUI's approach)
    model_options = {
        "transformer_options": {},
    }

    if not isinstance(model, comfy.model_patcher.ModelPatcher):
        model = comfy.model_patcher.ModelPatcher(model)

    # Create sampler
    scheduler_map = {
        "euler": "euler",
        "euler_ancestral": "euler_ancestral",
        "dpm_2": "dpm_2",
        "dpm_2_ancestral": "dpm_2_ancestral",
        "dpmpp_2s_ancestral": "dpmpp_2s_ancestral",
        "dpmpp_sde": "dpmpp_sde",
        "dpmpp_2m": "dpmpp_2m",
        "ddim": "ddim",
    }
    sampler_name = scheduler_map.get(scheduler, "euler_ancestral")

    sampler = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=model.device,
        sampler=sampler_name,
        scheduler="karras",
        denoise=denoise_strength,
        model_options=model_options,
    )

    # Prepare inpainting conditioning
    latent_image = samples
    latent_mask = vae.encode_mask(extended_mask)

    # Sample
    samples = sampler.sample(
        noise,
        positive_cond,
        negative_cond,
        latent_image=latent_image,
        latent_mask=latent_mask,
        cfg_scale=cfg_scale,
    )

    # Decode the latents to images
    result = vae.decode(samples)

    result_image = transforms.ToPILImage()(result[0].cpu())

    # Split result into garment and try-on/off images
    garment_result = result_image.crop((0, 0, width, height))
    try_result = result_image.crop((width, 0, width * 2, height))

    # Convert back to tensors
    try_result_tensor = torch.tensor(
        np.array(try_result) / 255.0, dtype=torch.float32
    ).unsqueeze(0)

    garment_result_tensor = torch.tensor(
        np.array(garment_result) / 255.0, dtype=torch.float32
    ).unsqueeze(0)

    return (
        try_result_tensor,
        garment_result_tensor,
    )


class ComfyTryOnOffNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
                "width": ("INT", {"default": 576, "min": 128, "max": 1024, "step": 16}),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 1024, "step": 16},
                ),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 1.0, "max": 100.0, "step": 0.1},
                ),
                "scheduler": (
                    [
                        "euler",
                        "euler_ancestral",
                        "dpm_2",
                        "dpm_2_ancestral",
                        "dpmpp_2s_ancestral",
                        "dpmpp_sde",
                        "dpmpp_2m",
                        "ddim",
                    ],
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
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "low quality, bad anatomy, worst quality, low res",
                    },
                ),
                "try_on": (["true", "false"],),
                "denoise_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "garment_in": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("result_image", "garment_image")
    FUNCTION = "run_inference"
    CATEGORY = "Try-On/Off"

    def run_inference(
        self,
        model,
        clip,
        vae,
        image_in,
        mask_in,
        width,
        height,
        steps,
        cfg_scale,
        scheduler,
        seed,
        prompt,
        negative_prompt,
        try_on,
        denoise_strength,
        garment_in=None,
    ):
        try_on_mode = try_on == "true"

        if try_on_mode and garment_in is None:
            print(
                "Warning: Try-on mode selected but no garment provided. Defaulting to try-off mode."
            )
            try_on_mode = False

        return comfy_tryon_off_inference(
            model=model,
            clip=clip,
            vae=vae,
            image_in=image_in,
            mask_in=mask_in,
            try_on=try_on_mode,
            garment_in=garment_in,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            seed=seed,
            width=width,
            height=height,
            denoise_strength=denoise_strength,
        )
