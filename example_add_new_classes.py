#!/usr/bin/env python3
"""
Example script demonstrating how to add new classes to a pretrained ResNet model.
This script shows three different approaches for extending ResNet models.
"""

import sys
import os
from add_new_class_lora_resnet import ResNetWithNewClasses


def create_dummy_dataset():
    """
    Create a dummy dataset for demonstration purposes.
    In practice, you would replace this with your actual dataset.
    """
    print("Creating dummy dataset for demonstration...")

    # This is where you would load your actual data
    # For example:
    # - New medical images for a new disease class
    # - New animal species images
    # - New product categories

    # Example dataset structure:
    dataset_info = {
        "original_classes": {
            "range": "0-999",
            "description": "ImageNet classes (dog, cat, car, etc.)",
        },
        "new_classes": {
            "1000": "custom_class_1",  # Your new class 1
            "1001": "custom_class_2",  # Your new class 2
            "1002": "custom_class_3",  # Your new class 3
        },
    }

    return dataset_info


def approach_1_simple_extension():
    """
    Approach 1: Simple classifier extension
    - Replace the final classifier layer with a larger one
    - Copy weights for original classes, initialize new ones
    - Best for: Adding a few new classes, simple implementation
    """
    print("\n" + "=" * 60)
    print("APPROACH 1: Simple Classifier Extension")
    print("=" * 60)

    # Create model with 3 additional classes
    model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,  # ImageNet
        new_num_classes=1003,  # Add 3 new classes
        use_lora=False,
    )

    # Option A: Freeze backbone (recommended for small datasets)
    print("Freezing backbone for transfer learning...")
    model.freeze_backbone(freeze=True)

    # Option B: Fine-tune entire model (for larger datasets)
    # model.freeze_backbone(freeze=False)

    # Get optimizer
    optimizer = model.get_optimizer(learning_rate=1e-3)

    print("✓ Model ready for training!")
    print(f"✓ Original classes: {model.original_num_classes}")
    print(f"✓ New total classes: {model.new_num_classes}")
    print(f"✓ Added classes: {model.new_num_classes - model.original_num_classes}")

    return model


def approach_2_lora_extension():
    """
    Approach 2: LoRA (Low-Rank Adaptation) extension
    - Parameter-efficient fine-tuning
    - Only trains a small number of additional parameters
    - Best for: Limited computational resources, preserving original model
    """
    print("\n" + "=" * 60)
    print("APPROACH 2: LoRA Parameter-Efficient Extension")
    print("=" * 60)

    # Create model with LoRA
    model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,
        new_num_classes=1005,  # Add 5 new classes
        use_lora=True,
    )

    # Get optimizer (only LoRA parameters will be trained)
    optimizer = model.get_optimizer(learning_rate=1e-4)

    print("✓ LoRA model ready!")
    print("✓ Only a small fraction of parameters will be trained")
    print("✓ Original model weights are preserved")

    return model


def approach_3_dual_heads():
    """
    Approach 3: Dual heads approach
    - Keep original classifier separate from new classifier
    - Useful when you want to distinguish between original and new classes
    - Best for: When you need to track which head made the prediction
    """
    print("\n" + "=" * 60)
    print("APPROACH 3: Dual Heads Approach")
    print("=" * 60)

    # Create model with dual heads
    model = ResNetWithNewClasses(
        base_model_name="resnet18",  # Using smaller model for demo
        original_num_classes=1000,
        new_num_classes=1002,  # Add 2 new classes
    )

    # Apply dual heads modification
    model.approach_2_dual_heads()

    print("✓ Dual heads model ready!")
    print("✓ Original ImageNet head: outputs classes 0-999")
    print("✓ New classes head: outputs classes 1000-1001")
    print("✓ Final output: concatenated predictions")

    return model


def training_example():
    """
    Example of how you would train the model with new classes.
    """
    print("\n" + "=" * 60)
    print("TRAINING EXAMPLE")
    print("=" * 60)

    # For this example, we'll use approach 1
    model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,
        new_num_classes=1003,
        use_lora=False,
    )

    # Freeze backbone for transfer learning
    model.freeze_backbone(freeze=True)

    print("Training setup:")
    print("1. Prepare your dataset with new classes")
    print("   - Class 0-999: ImageNet classes")
    print("   - Class 1000: Your new class 1")
    print("   - Class 1001: Your new class 2")
    print("   - Class 1002: Your new class 3")

    print("\n2. Create data loaders:")
    print("   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)")
    print("   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)")

    print("\n3. Training loop:")
    print("   optimizer = model.get_optimizer(learning_rate=1e-3)")
    print("   criterion = nn.CrossEntropyLoss()")
    print("   ")
    print("   for epoch in range(num_epochs):")
    print(
        "       train_loss, train_acc = model.train_step(train_loader, optimizer, criterion, epoch)"
    )
    print("       val_loss, val_acc = model.validate(val_loader, criterion)")

    print("\n4. Save the model:")
    print("   model.save_model('path/to/your/model.pth')")

    return model


def prediction_example():
    """
    Example of how to use the trained model for predictions.
    """
    print("\n" + "=" * 60)
    print("PREDICTION EXAMPLE")
    print("=" * 60)

    # Assuming you have a trained model
    model = ResNetWithNewClasses(
        base_model_name="resnet50", original_num_classes=1000, new_num_classes=1003
    )

    # Define class names (including new ones)
    class_names = []

    # Add ImageNet class names (0-999)
    # In practice, you'd load these from imagenet_classes.txt
    class_names.extend([f"imagenet_class_{i}" for i in range(1000)])

    # Add your new class names (1000-1002)
    class_names.extend(["my_new_class_1", "my_new_class_2", "my_new_class_3"])

    print("Prediction usage:")
    print("# Load an image and predict")
    print("result = model.predict('path/to/image.jpg', class_names)")
    print("print(f'Predicted class: {result[\"predicted_class_name\"]}')")
    print("print(f'Confidence: {result[\"confidence\"]:.3f}')")

    return model


def main():
    """
    Main demonstration function.
    """
    print("ResNet New Class Addition - Complete Guide")
    print("=" * 60)

    # Show dataset structure
    dataset_info = create_dummy_dataset()
    print(f"Dataset structure: {dataset_info}")

    try:
        # Demonstrate different approaches
        model1 = approach_1_simple_extension()
        model2 = approach_2_lora_extension()
        model3 = approach_3_dual_heads()

        # Show training example
        training_model = training_example()

        # Show prediction example
        prediction_model = prediction_example()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ Three approaches demonstrated:")
        print("  1. Simple Extension: Replace classifier, copy weights")
        print("  2. LoRA Extension: Parameter-efficient fine-tuning")
        print("  3. Dual Heads: Separate heads for original vs new classes")

        print("\n✓ Key considerations:")
        print("  - Dataset size: Small→freeze backbone, Large→fine-tune all")
        print("  - Computational resources: Limited→use LoRA")
        print("  - Class relationships: Related→simple extension, Unrelated→dual heads")

        print("\n✓ Next steps:")
        print("  1. Prepare your dataset with proper class labels")
        print("  2. Choose the approach that fits your use case")
        print("  3. Implement data loading and training loop")
        print("  4. Train and evaluate your model")

    except Exception as e:
        print(
            f"Note: This is a demonstration script. Some imports may not be available."
        )
        print(f"Error: {e}")
        print(
            "To run this code, ensure you have: torch, torchvision, timm, peft installed"
        )


if __name__ == "__main__":
    main()
