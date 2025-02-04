# ComfyUI-Flux-TryOff

Original Source: [catvton-flux](https://github.com/nftblackmagic/catvton-flux). I implemented their try-off inference code as ComfyUI nodes
There's a sample workflow in [Workflow](https://github.com/asutermo/ComfyUI-Flux-TryOff/tree/main/workflow) that uses SegFormer to generate the mask for you. I highly recommend this approach. Alternatively you can provide your own!

Please note, that this was tested with a 4080, and it's quite slow. You'll want a 4090 or better for performant execution as of right now.

This uses diffusers>=0.32.2 but you no longer need to approve on the Hugging Face site or use the Flux.1 Dev Model

```diff
- This is presently incompatible with Flux fp8 single file.
```

After heavy experimenting with Try-on, it's nice to have a [Try-Off, xiaozaa/cat-tryoff-flux](https://huggingface.co/xiaozaa/cat-tryoff-flux) model to work with.
All models will download automatically unless you use the legacy 'FluxFill Model Loader'. The quantized versions will work on lower end GPUs but this has not been verified for multi-gpu runs.

![Quantized Sample](./quantized_sample_4bit.png)

## TODO

- Multi-gpu testing
- Optimize, optimize, optimize.
- TryOn

## Legacy (<= v1.1)

1. Go to huggingface
2. Go to your settings and generate a 'write' token
3. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev and accept the terms
4. Open a prompt, go to your ComfyUI installation and do the following

Windows

```bat
SET HF_TOKEN=<token_from_above>
SET HUGGING_FACE_HUB_TOKEN=<token_from_above>
```

Linux

```sh
EXPORT HF_TOKEN=<token_from_above>
EXPORT HUGGING_FACE_HUB_TOKEN=<token_from_above>
```

Finally, download FLUX.1

```sh
cd ./models/checkpoints
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev
```

And run

```sh
cd ../..
python ./main.py
```
