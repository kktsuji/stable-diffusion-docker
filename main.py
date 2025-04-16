import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
num_inference_steps = 50

for i in range(1):
    seed = i
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]

    image.save(f"./out/astronaut_horse_steps{num_inference_steps}_seed{seed}.png")
