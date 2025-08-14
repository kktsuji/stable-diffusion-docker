from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import collections

data_path = "./data"
# val_dataset = "./data/normal_cells/"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(data_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
