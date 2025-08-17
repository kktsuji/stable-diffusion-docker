from add_new_class_lora_resnet import ResNetWithNewClasses

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import collections


def approach_4_new_classes_only():
    """
    Approach 4: New classes only (discard original ImageNet classes)
    - Replace classifier to handle only your custom classes
    - No ImageNet classes, much smaller and more efficient
    - Best for: Specific tasks where ImageNet classes are not needed
    """
    print("\n" + "=" * 60)
    print("APPROACH 4: New Classes Only (Discard ImageNet)")
    print("=" * 60)

    # Create model without extending the classifier initially
    model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,
        new_num_classes=None,  # Don't extend in init
        use_lora=False,
    )

    # Replace classifier to handle only 2 custom classes
    model.new_classes_only(num_new_classes=2)

    # Freeze backbone for transfer learning
    model.freeze_backbone(freeze=True)

    # Get optimizer
    optimizer = model.get_optimizer(learning_rate=1e-3)

    print("✓ New classes only model ready!")
    print(f"✓ Original ImageNet classes: DISCARDED")
    print(f"✓ New custom classes: 5 classes (0-4)")
    print(f"✓ Model size: Much smaller and more efficient")
    print(f"✓ Class mapping:")
    print(f"   - Class 0: Your custom class 1")
    print(f"   - Class 1: Your custom class 2")
    print(f"   - Class 2: Your custom class 3")
    print(f"   - Class 3: Your custom class 4")
    print(f"   - Class 4: Your custom class 5")

    print("\n✓ Use cases:")
    print("   - Medical image classification (normal, disease1, disease2)")
    print("   - Quality control (good, defect_type1, defect_type2)")
    print("   - Custom object detection (apple, orange, banana)")
    print("   - Document classification (invoice, receipt, contract)")

    return model


def train_resnet(train_data_path, val_data_path, num_epochs, model_out_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Demonstrate different approaches
        model = approach_4_new_classes_only()
        model.model.to(device)
        print(f"Model moved to {device}")

        # Show training example
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
        val_dataset = datasets.ImageFolder(val_data_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        print("=== DataLoader Details ===")
        print(f"Train loader batch size: {train_loader.batch_size}")
        print(f"Train loader shuffle: {train_loader.sampler}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Train dataset classes: {train_dataset.classes}")
        print(f"Number of batches: {len(train_loader)}")

        print("\n=== Images per Class ===")
        label_counts = collections.Counter(train_dataset.targets)
        for class_idx, count in label_counts.items():
            class_name = train_dataset.classes[class_idx]
            print(f"Class {class_idx} ({class_name}): {count} images")

        print("\n3. Training loop:")
        optimizer = model.get_optimizer(learning_rate=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            train_loss, train_acc = model.train_step(
                train_loader, optimizer, criterion, epoch
            )
            val_loss, val_acc = model.validate(val_loader, criterion)

        print("\n4. Save the model:")
        model.save_model(model_out_path)

        # Show prediction example
        # prediction_model = prediction_example()

    except Exception as e:
        print(
            f"Note: This is a demonstration script. Some imports may not be available."
        )
        print(f"Error: {e}")
        print(
            "To run this code, ensure you have: torch, torchvision, timm, peft installed"
        )


if __name__ == "__main__":
    train_resnet(
        train_data_path="./data/val",
        val_data_path="./data/val",
        num_epochs=50,
        model_out_path="./models/resnet50_epochs10.pth",
    )
