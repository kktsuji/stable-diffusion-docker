import os

RESOLUTION = "40"  # original size
# RESOLUTION = "256"
# RESOLUTION = "512"  # recommended for sd-1.x

out_dir_name = f"cell-rois_lora_size-{RESOLUTION}"  # Edit
out_dir_path = f"./loras/{out_dir_name}/"
if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)
    os.makedirs(out_dir_path + "logs")
print(out_dir_path, os.path.exists(out_dir_path))

base_data_dir_path = "./data/pseudo_rgb/"
print(base_data_dir_path, os.path.exists(base_data_dir_path))

model_id = "CompVis/stable-diffusion-v1-4"
model_dir_path = "./models/" + model_id
print(model_dir_path, os.path.exists(model_dir_path))

import sys

sys.path.append("./peft/examples/stable_diffusion")
import train_dreambooth

print(train_dreambooth.UNET_TARGET_MODULES)

# for "CompVis/stable-diffusion-v1-4"
unet_target_modules = [
    # Convolution layers
    "conv_in",
    "conv1",
    "conv2",
    "conv_out",
    "conv_shortcut",
    "conv",
    # Linear layers
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "linear_1",
    "linear_2",
    "proj_in",
    "proj_out",
    "proj",
    "time_emb_proj",
    "ff.net.0.proj",
    "ff.net.2",
]

train_dreambooth.UNET_TARGET_MODULES = unet_target_modules
print(train_dreambooth.UNET_TARGET_MODULES)

input_args = [
    "--pretrained_model_name_or_path",
    model_dir_path,
    "--instance_data_dir",
    base_data_dir_path,
    "--instance_prompt",
    "rds",
    "--seed",
    "0",
    "--resolution",
    RESOLUTION,
    "--output_dir",
    out_dir_path,
    "--num_train_epochs",
    "100",
    "--lr_scheduler",
    "cosine",
    "--lr_warmup_steps",
    "5",
    "--learning_rate",
    "1e-4",
    "--logging_dir",
    "logs",
    "--report_to",
    "tensorboard",
    "lora",
    "--unet_r",
    "16",
    "--unet_alpha",
    "16",
]

args = train_dreambooth.parse_args(input_args)
train_dreambooth.main(args)
