import os
from diffusers import DiffusionPipeline
import torch
from attention import load_attn_processors


def main():
    model_name = 'runwayml/stable-diffusion-v1-5'
    device = 'cuda'
    prompt = 'robotic cat with wings'
    num_images = 4

    # load model
    pipeline = DiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)

    load_attn_processors(pipeline.unet, device,
                         torch.float32, "output/attn_processors.pt")
    
    # generate images
    generator = torch.Generator(device=device)
    images = []
    for _ in range(num_images):
        images.append(pipeline(prompt, num_inference_steps=30,
                      generator=generator).images[0])

    # save images
    save_path = os.path.join('output', "infer-images")
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join(save_path, f"{i}.png"))


if __name__ == '__main__':
    main()
