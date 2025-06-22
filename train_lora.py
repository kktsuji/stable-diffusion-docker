import os

# RESOLUTION = "40" # original size
# RESOLUTION = "256"
RESOLUTION = "512"  # recommended for sd-1.x

OUT_DIR_NAME = f"250622_cell_rois_lora_size-{RESOLUTION}"  # Edit
OUT_DIR_PATH = f"./loras/{OUT_DIR_NAME}/"
if not os.path.exists(OUT_DIR_PATH):
    os.makedirs(OUT_DIR_PATH)
print(OUT_DIR_PATH, os.path.exists(OUT_DIR_PATH))

BASE_DATA_DIR_PATH = "./data/pseudo_rgb/"
print(BASE_DATA_DIR_PATH, os.path.exists(BASE_DATA_DIR_PATH))

model_id = "CompVis/stable-diffusion-v1-4"
MODEL_DIR_PATH = "./models/" + model_id
print(MODEL_DIR_PATH, os.path.exists(MODEL_DIR_PATH))

import sys

sys.path.append("./peft/examples/stable_diffusion")
import train_dreambooth

print(train_dreambooth.UNET_TARGET_MODULES)

# for "CompVis/stable-diffusion-v1-4"
UNET_TARGET_MODULES = [
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

train_dreambooth.UNET_TARGET_MODULES = UNET_TARGET_MODULES
print(train_dreambooth.UNET_TARGET_MODULES)

input_args = [
    "--pretrained_model_name_or_path",
    MODEL_DIR_PATH,
    "--instance_data_dir",
    BASE_DATA_DIR_PATH,
    "--instance_prompt",
    "rds",
    "--seed",
    "0",
    "--resolution",
    RESOLUTION,
    "--output_dir",
    OUT_DIR_PATH,
    "--num_train_epochs",
    "100",
    "--lr_scheduler",
    "cosine",
    "--lr_warmup_steps",
    "5",
    "--learning_rate",
    "1e-4",
    "lora",
    "--unet_r",
    "16",
    "--unet_alpha",
    "16",
]

args = train_dreambooth.parse_args(input_args)
train_dreambooth.main(args)
