# (IA)^3 for Stable Diffusion

Parameter-efficient fine-tuning of Stable Diffusion using (IA)^3.

Based on these papers:

- [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

**This repository is currently a work in progress.**

Implemented in [diffusers](https://github.com/huggingface/diffusers) using an attention processor in [`attention.py`](/attention.py).

Training script in [`train.py`](/train.py). Based on [this example script for diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py).

Inference script in [`infer.py`](/infer.py) to load the changes and generate images.

## To-do

- [ ] Add CLI arguments for scripts
- [x] Save state dict instead of model
- [ ] Give example results and saved models
- [x] Reparameterize to initialize with all zeros to fix weight decay
- [ ] Add requirements.txt
