import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

# Download the pretrained model from Hugging Face Hub
repo_id = "CompVis/stable-diffusion-v1-4"
local_model_path = f"./models/{repo_id}"
snapshot_download(repo_id=repo_id, local_dir=local_model_path)

# Load the model from the local path
pipe = StableDiffusionPipeline.from_pretrained(
    local_model_path, torch_dtype=torch.float16
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
