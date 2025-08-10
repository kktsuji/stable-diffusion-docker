from diffusers import StableDiffusionPipeline
import torch
from peft import PeftModel

import os

RESOLUTION = 512

lora_dir_name = f"cell-rois_lora_size-{RESOLUTION}_target-module"  # Edit
lora_path = f"./loras/{lora_dir_name}/unet/"
print(lora_path, os.path.exists(lora_path))

model_id = "CompVis/stable-diffusion-v1-4"
model_dir_path = "./models/" + model_id
print(model_dir_path, os.path.exists(model_dir_path))

out_dir = f"./out/{lora_dir_name}/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print(out_dir, os.path.exists(out_dir))

seed_num = 10
prompt = "circle"
lora_id = "rsd"
# prompt_lora = lora_id + " " + prompt
prompt_lora = lora_id

pipe = StableDiffusionPipeline.from_pretrained(
    model_dir_path, torch_dtype=torch.float32
)
pipe.to("cuda")

original_unet = pipe.unet


def generate_and_save(pipe, prompt, seed, out_dir, suffix):
    image = pipe(prompt, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    image.save(out_dir + f"seed{seed}_" + prompt.replace(" ", "_") + f"_{suffix}.png")


for seed in range(seed_num):
    print(f"seed: {seed} / {seed_num - 1}")

    generate_and_save(pipe, prompt, seed, out_dir, "ori")
#   generate_and_save(pipe, prompt_lora, seed, out_dir, "ori")
exit()
lora_unet = PeftModel.from_pretrained(original_unet, lora_path, adapter_name=lora_id)
pipe.unet = lora_unet
pipe.to("cuda")

for seed in range(seed_num):
    print(f"seed: {seed} / {seed_num - 1}")

    # generate_and_save(pipe, prompt, seed, out_dir, "lora")
    generate_and_save(pipe, prompt_lora, seed, out_dir, "lora")
