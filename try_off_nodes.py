import os
from dataclasses import dataclass
from typing import List

import folder_paths
import numpy as np  # type: ignore
import torch  # type: ignore
from comfy import model_management
from diffusers import (  # type: ignore
    AutoencoderKL,
    AutoencoderTiny,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    FluxTransformer2DModel,
)
from PIL import Image
from torchvision import transforms  # type: ignore
from transformers import (  # type: ignore
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

# from diffusers.scripts import convert_diffusers_to_original_stable_diffusion
# from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint


__all__ = [
    "TryOffModelNode",
    "TryOffFluxFillModelNode",
    "TryOffRunNode",
    "TryOffQuantizerNode",
    "FluxFillPipelineNode",
]


device_list = ["cuda", "cpu"]
node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_dir = os.path.abspath(os.path.join(node_dir, "..", ".."))
models_dir = os.path.abspath(os.path.join(comfy_dir, "models"))
checkpoints_dir = os.path.abspath(os.path.join(models_dir, "checkpoints"))
encoders_dir = os.path.abspath(os.path.join(models_dir, "text_encoders"))
clip_dir = os.path.abspath(os.path.join(models_dir, "clip_vision"))
vae_dir = os.path.abspath(os.path.join(models_dir, "vae"))

dtype = torch.bfloat16

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


# TryOffModel Node
class TryOffModelNode:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "model_name": (["xiaozaa/cat-tryoff-flux"],),
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

            # just get inputs from encoders
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
        pipeline.transformer.to(dtype)

        return (pipeline,)


# TryOffModel Node
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


def inference(
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

    # Preprocessing transforms
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

    if garment_in:
        garment = convert_image(garment_in).resize((width, height))
        try_on = True
    else:
        try_on = False

    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]  # TT5EncoderModelake only first channel

    if try_on:
        garment_tensor = transform(garment)
    else:
        garment_tensor = torch.zeros_like(image_tensor)
        image_tensor = image_tensor * mask_tensor

    # Concatenate inputs for FluxFillPipeline
    inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
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
        num_inference_steps=num_steps,
        generator=torch.manual_seed(seed),
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

    return (
        tryoff_result,
        garment_result,
    )


# TryOffRun Node
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
    RETURN_NAMES = ("garment_image", "tryonoff_image")
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
        return inference(
            image_in,
            mask_in,
            pipe,
            width,
            height,
            num_steps,
            guidance_scale,
            seed,
            prompt,
            garment_in,
        )


# TryOffRun Node
class TryOnOffRunNodeAdvanced:
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "flux_catvton_model": ("MODEL",),
                "vae": (folder_paths.get_filename_list("vae"),),
                "clip_l_encoder": (folder_paths.get_filename_list("text_encoders")+folder_paths.get_filename_list("clip_vision"),),
                "t5_encoder": (folder_paths.get_filename_list("text_encoders")+folder_paths.get_filename_list("clip_vision"),),
                "image_in": ("IMAGE",),
                "mask_in": ("MASK",),
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
                "transformers_config": ("transformers_config",),
                "diffusers_config": ("diffusers_config",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("garment_image", "tryonoff_image")
    CATEGORY = "Processing"
    FUNCTION = "run_inference"

    def run_inference(
        self,
        flux_catvton_model,
        vae,
        clip_l_encoder,
        t5_encoder,
        image_in,
        mask_in,
        width,
        height,
        num_steps,
        guidance_scale,
        seed,
        prompt,
        garment_in=None,
        transformers_config=None,
        diffusers_config=None,
    ):
        targs = {"torch_dtype": dtype}
        if transformers_config:
            targs['quantization_config'] = transformers_config
        dargs = {"torch_dtype": dtype}
        if diffusers_config:
            dargs['quantization_config'] = diffusers_config

        def get_full_path(possible_paths: List[str], file_path: str):
            for path in possible_paths:
                full_path = os.path.join(path, file_path)
                print(f'Trying {full_path}')
                if os.path.exists(full_path):
                    print(f'Using {full_path}')
                    return full_path
            raise Exception("Model file {file_path} not found in any of the possible paths")
        
        clip_encoder_path = get_full_path([encoders_dir, clip_dir], clip_l_encoder)
        t5_encoder_path = get_full_path([encoders_dir, clip_dir], t5_encoder)
        vae_path = get_full_path([vae_dir], vae)

        clip_tokenizer = CLIPTokenizer()

        tokenizer = CLIPTokenizer.from_pretrained(clip_encoder_path, **targs)
        tokenizer_2 = T5TokenizerFast.from_pretrained(t5_encoder_path, **targs)

        text_encoder = CLIPTextModel.from_pretrained(clip_encoder_path, **targs)
        text_encoder_2 = T5EncoderModel.from_pretrained(t5_encoder_path, **targs)
        scheduler = FlowMatchEulerDiscreteScheduler()
        vae = AutoencoderKL.from_pretrained(vae_path, **dargs)
        pipe = FluxFillPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=flux_catvton_model,
        )
        pipe.enable_model_cpu_offload()
        pipe.transformer.to(dtype)
        return inference(
            image_in,
            mask_in,
            pipe,
            width,
            height,
            num_steps,
            guidance_scale,
            seed,
            prompt,
            garment_in,
        )
            