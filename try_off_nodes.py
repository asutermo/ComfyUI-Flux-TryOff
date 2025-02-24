import json
import os
from dataclasses import dataclass
from typing import List

import comfy.utils
import folder_paths
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F
from comfy import model_management
from comfy.model_base import Flux
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
from huggingface_hub import snapshot_download

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
                "conditioning": ("CONDITIONING",),
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
        conditioning,
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
            targs["quantization_config"] = transformers_config
        dargs = {"torch_dtype": dtype}
        if diffusers_config:
            dargs["quantization_config"] = diffusers_config

        def get_full_path(possible_paths: List[str], file_path: str):
            for path in possible_paths:
                full_path = os.path.join(path, file_path)
                print(f"Trying {full_path}")
                if os.path.exists(full_path):
                    print(f"Using {full_path}")
                    return full_path
            raise Exception(
                "Model file {file_path} not found in any of the possible paths"
            )

        vae_path = get_full_path([vae_dir], vae)

        clip_tokenizer = CLIPTokenizer()

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


class FluxModelWrapper(Flux):
    def __init__(self, model, model_type="FLUX"):
        unet_config = {
            "image_model": "flux",
            "guidance_embed": True,
            "in_channels": 96,
        }
        super().__init__(model_config = {"unet_config": unet_config})
        self.model = model
        self.device = model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
        self.dtype = model.dtype

    def forward(self, x, timesteps, context, **kwargs):
        x = x.to(self.device, dtype=self.dtype)
        timesteps = timesteps.to(self.device)
        context = context.to(self.device, dtype=self.dtype)

        return self.model(
            sample=x,
            timestep=timesteps,
            encoder_hidden_states=context,
            return_dict=False,
        )[0]

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def cleanup(self):
        if self.offload_device != torch.device("cpu"):
            self.to(self.offload_device)

    @torch.no_grad()
    def get_input_block_skip_connections(self):
        return []


CUSTOM_MODELS_DIR = os.path.join(folder_paths.models_dir, "catvton")
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)


class TryOnOffLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [
                        "xiaozaa/cat-tryoff-flux",
                        "xiaozaa/catvton-flux-beta",
                        "xiaozaa/catvton-flux-alpha",
                    ],
                ),
                "precision": (["full", "half"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    @classmethod
    def load_model(cls, model_name, precision):
        cache_dir = CUSTOM_MODELS_DIR
        local_dir = os.path.join(cache_dir, model_name)
        os.makedirs(local_dir, exist_ok=True)
        
        dtype = torch.float16 if precision == "half" else torch.float32

        quantization_config = TransformersBitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    
        # Load model using diffusers
        print(f"Loading {model_name} from HuggingFace or cache...")
        # flux_model = FluxTransformer2DModel.from_pretrained(
        #     model_name,
        #     cache_dir=cache_dir,
        #     torch_dtype=dtype,
        #     quantization_config=quantization_config
        # )
        snapshot_download(repo_id=model_name, local_dir=local_dir, cache_dir=cache_dir)
        index_path = os.path.join(cache_dir, model_name, "index.json")

        def load_from_index(index_path):
            print(f"Load_from_index: index_path: {index_path}")
            
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            weight_map = index_data['weight_map']
            state_dict = {}
            
            for key, file in weight_map.items():
                file_path = os.path.join(os.path.dirname(index_path), file)
                part_dict = comfy.utils.load_torch_file(file_path, safe_load=True)
                state_dict.update(part_dict)
                del part_dict
                #torch.cuda.empty_cache()
            
            print(f"Load from index function completed. Returning state_dict")
            return state_dict
        state_dict = load_from_index(index_path) 
        model_options = {}
        weight_dtype="fp8_e4m3fn"
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        
        model = cls.load_diffusion_model_from_state_dict(state_dict, model_options=model_options)
        return (model,)


class TryOnOffSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
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
                        "default": "The pair of images highlights clothing and its styling on a model, high resolution, 4K, 8K; [IMAGE1] Detailed product shot of clothing [IMAGE2] The same clothing is worn by a model in a lifestyle setting.",
                    },
                ),
            },
            "optional": {
                "garment_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("garment_result", "tryon_result", "latent")
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def prepare_image_tensor(self, image, width, height):
        # Ensure we're working with the right dimensions
        if len(image.shape) == 3:  # Add batch dimension if needed
            image = image.unsqueeze(0)
        
        # Convert from NHWC to NCHW format
        if image.shape[3] == 3 or image.shape[3] == 4:  # If in NHWC format
            image = image.permute(0, 3, 1, 2)
            if image.shape[1] == 4:  # If RGBA, convert to RGB
                image = image[:, :3]
        
        # Resize if needed
        if image.shape[2] != height or image.shape[3] != width:
            image = F.interpolate(image, size=(height, width), mode='bilinear')
        
        # Ensure normalized to [-1, 1] for VAE
        if image.max() > 1.0:
            image = image / 255.0
        image = image * 2.0 - 1.0
        
        return image

    def prepare_mask_tensor(self, mask, width, height):
        # Ensure mask has right dimensions
        if len(mask.shape) == 2:  # Add batch and channel dimensions if needed
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3 and mask.shape[2] == 1:  # If [H,W,1]
            mask = mask.permute(2, 0, 1).unsqueeze(0)  # To [1,1,H,W]
        elif len(mask.shape) == 3:  # If [B,H,W]
            mask = mask.unsqueeze(1)  # To [B,1,H,W]
        elif len(mask.shape) == 4 and mask.shape[3] == 1:  # If [B,H,W,1]
            mask = mask.permute(0, 3, 1, 2)  # To [B,1,H,W]
        
        # Resize if needed
        if mask.shape[2] != height or mask.shape[3] != width:
            mask = F.interpolate(mask, size=(height, width), mode='nearest')
        
        # Ensure binary values
        mask = (mask > 0.5).float()
        
        return mask



    def get_conditioning(self, clip, prompt):
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def sample(
        self,
        model,
        vae,
        clip,
        image,
        mask,
        width,
        height,
        num_steps,
        guidance_scale,
        seed,
        prompt,
        garment_image=None,
    ):
        device = model_management.get_torch_device()

        # Prepare conditioning
        positive = self.get_conditioning(clip, prompt)
        negative = self.get_conditioning(clip, "")

        # Prepare images
        image_tensor = self.prepare_image_tensor(image, width, height)
        mask_tensor = self.prepare_mask_tensor(mask, width, height)

        if garment_image is not None:
            garment_tensor = self.prepare_image_tensor(garment_image, width, height)
            try_on = True
        else:
            garment_tensor = torch.zeros_like(image_tensor)
            image_tensor = image_tensor * mask_tensor
            try_on = False

        # KEY FIX: Concatenate inputs along width dimension (dim=3 for NCHW format)
        # We need width*2 for the final image
        double_width = width * 2
        concat_height = height
        
        # Create a new tensor with the right dimensions for side-by-side images
        inpaint_image = torch.zeros((1, 3, concat_height, double_width), device=device)
        
        # Place the garment image on the left half
        inpaint_image[:, :, :, :width] = garment_tensor
        
        # Place the person image on the right half
        inpaint_image[:, :, :, width:] = image_tensor
        
        # Similarly for the mask
        extended_mask = torch.zeros((1, 1, concat_height, double_width), device=device)
        
        if try_on:
            # For try-on, mask is on the right side
            extended_mask[:, :, :, width:] = mask_tensor
        else:
            # For try-off, inverse mask on the left side
            extended_mask[:, :, :, :width] = 1 - torch.zeros_like(mask_tensor)

        # Encode to latent space
        latent = vae.encode(inpaint_image)

        # Set up noise
        generator = torch.manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device=device)

        # Sample
        samples = comfy.sample.sample(
            model,
            noise,
            steps=num_steps,
            cfg=guidance_scale,
            sampler_name="euler",
            scheduler="normal",
            positive=positive,
            negative=negative,
            denoise=1.0,
            disable_noise=False,
        )

        # Decode latents
        decoded = vae.decode(samples)

        # Split and process results
        garment_result = decoded[:, :, :, :width]
        tryon_result = decoded[:, :, :, width:]

        # Convert to ComfyUI format [B,H,W,C]
        garment_result = garment_result.permute(0, 2, 3, 1)
        tryon_result = tryon_result.permute(0, 2, 3, 1)

        # Normalize to [0, 1]
        garment_result = (garment_result + 1) / 2
        tryon_result = (tryon_result + 1) / 2

        return (garment_result, tryon_result, {"samples": samples})
