from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import collections
from PIL import Image
import numpy as np
import os

# docker run --rm -it --gpus all --network=host -v $PWD:/work -w /work pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime ./venv/bin/python load_data.py

data_path = "./data/train"
val_dataset = "./data/val"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(data_path, transform=transform)
val_dataset = datasets.ImageFolder(val_dataset, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("=== DataLoader Details ===")
print(f"Train loader batch size: {train_loader.batch_size}")
print(f"Train loader shuffle: {train_loader.sampler}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Train dataset classes: {train_dataset.classes}")
print(f"Number of batches: {len(train_loader)}")

# print("\n=== Sample Batch ===")
# for images, labels in train_loader:
#     print(
#         f"Batch images shape: {images.shape}"
#     )  # [batch_size, channels, height, width]
#     print(f"Batch labels shape: {labels.shape}")  # [batch_size]
#     print(f"Labels in this batch: {labels}")

print("\n=== Images per Class (Method 1) ===")
label_counts = collections.Counter(train_dataset.targets)
for class_idx, count in label_counts.items():
    class_name = train_dataset.classes[class_idx]
    print(f"Class {class_idx} ({class_name}): {count} images")


def save_transformed_as_rgb(dataset, save_dir="./out/transformed_rgb", num_samples=5):
    """Save transformed images as RGB images"""
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(num_samples, len(dataset))):
        image_tensor, label = dataset[i]
        class_name = dataset.classes[label]

        # Convert tensor back to PIL Image (RGB format)
        # Tensor is in CHW format [0, 1] range
        image_array = image_tensor.permute(1, 2, 0).numpy()  # CHW -> HWC
        image_array = (image_array * 255).astype("uint8")  # [0,1] -> [0,255]

        # Create RGB PIL Image
        if image_array.shape[2] == 3:  # Already RGB
            rgb_image = Image.fromarray(image_array, mode="RGB")
        elif image_array.shape[2] == 1:  # Grayscale
            # Convert grayscale to RGB
            rgb_image = Image.fromarray(image_array.squeeze(), mode="L").convert("RGB")
        else:
            print(f"Unexpected number of channels: {image_array.shape[2]}")
            continue

        # Save as RGB image
        filename = f"{save_dir}/rgb_{i}_{class_name}.jpg"
        rgb_image.save(filename, format="JPEG", quality=95)
        print(f"Saved RGB image: {filename}")


# Save transformed images as RGB
save_transformed_as_rgb(
    train_dataset, save_dir="./out/train_transformed_rgb", num_samples=200
)
save_transformed_as_rgb(
    val_dataset, save_dir="./out/val_transformed_rgb", num_samples=200
)
