# ComfyUI-Flux-TryOff

Original Source: [catvton-flux](https://github.com/nftblackmagic/catvton-flux). I implemented their try-off inference code as ComfyUI nodes.

After heavy experimenting with Try-on, it's nice to have a [Try-Off](https://huggingface.co/xiaozaa/cat-tryoff-flux) model to work with.

The cat-try-off-flux model will download automatically.


You'll need a huggingface token, and have gone through Flux's agreement to do the following. Go to the comfyui directory.

```sh
cd ./models/checkpoints
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev
```

- [xiaozaa/cat-tryoff-flux](https://huggingface.co/xiaozaa/cat-tryoff-flux)

## TODO

- Allow additional models
- Formatting/consistency
- Precision
