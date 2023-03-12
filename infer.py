import os
from diffusers import DiffusionPipeline
import pickle
import torch
from attention import load_attn_processors

model_name = 'runwayml/stable-diffusion-v1-5'
device = 'cuda'
prompt = 'robotic cat with wings'
num_images = 4

pipeline = DiffusionPipeline.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)

load_attn_processors(pipeline.unet, device, torch.float32, "output/attn_processors.pt")

generator = torch.Generator(device=device)
images = []
for _ in range(num_images):
    images.append(pipeline(prompt, num_inference_steps=30, generator=generator).images[0])


save_path = os.path.join('output', "infer-images")
os.makedirs(save_path, exist_ok=True)
for i, image in enumerate(images):
    image.save(os.path.join(save_path, f"{i}.png"))
