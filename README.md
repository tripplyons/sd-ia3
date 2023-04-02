# (IA)^3 for Stable Diffusion

Parameter-efficient fine-tuning of Stable Diffusion using (IA)^3.

| Before Fine-Tuning | After Fine-Tuning |
| --- | --- |
| ![Donald Trump](/docs/no-adapter-example.png) | ![Donald Trump](/docs/adapter-example.png) |

The prompt is "donald trump", and the model is fine-tuned on [pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)

Based on these papers:

- (IA)^3: [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)
- Stable Diffusion: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

Implemented in [diffusers](https://github.com/huggingface/diffusers) using an attention processor in [`attention.py`](/attention.py).

## Comparison to full fine-tuning

(IA)^3 has trade-offs similar to LoRA when comparing to full fine-tuning.

One major difference to LoRA is that (IA)^3 uses much less parameters. In general, it will most likely be faster and smaller, but less expressive.

- Faster training
- Smaller file size (~222 KB for Stable Diffusion 1.5)
- Can be swapped in and out of the base model during inference
- Can be loaded into fine-tuned models that have the same architecture
- Can be merged with the weights of the base model
  - Only possible when `learn_biases=False` without changing the architecture
  - Not currently implemented in this repo

## Installation

First create an environment and [install PyTorch](https://pytorch.org/get-started/locally/).

Then install the pip dependencies:

```bash
pip install -r requirements.txt
```

Currently, bitsandbytes only supports Linux, so fine-tuning on Windows requires more VRAM.

## Training

Training script in [`train.py`](/train.py). Based on [this example script for diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py).

Currently you can change the parameters by editing the variables at the top of the file and running the script:

```bash
python train.py
```

## Inference

Inference script in [`infer.py`](/infer.py) to load the changes and generate images.

Currently you can change the parameters by editing the variables at the top of the file and running the script:

```bash
python infer.py
```
