import os
from diffusers import DiffusionPipeline
import pickle
import torch

model_name = 'runwayml/stable-diffusion-v1-5'
device = 'cuda'
prompt = 'robotic cat with wings'
num_images = 4

pipeline = DiffusionPipeline.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)

with open("output/attn_processors.pkl", "rb") as f:
    attn_processors = pickle.load(f)
    pipeline.unet.set_attn_processor(attn_processors)


generator = torch.Generator(device=device)
images = []
for _ in range(num_images):
    images.append(pipeline(prompt, num_inference_steps=30, generator=generator).images[0])


save_path = os.path.join('output', "infer-images")
os.makedirs(save_path, exist_ok=True)
for i, image in enumerate(images):
    image.save(os.path.join(save_path, f"{i}.png"))
