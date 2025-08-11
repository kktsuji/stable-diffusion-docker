# Adding New Classes to Pretrained ResNet Models

## Overview

Yes, you can definitely add new classes to a pretrained ResNet model! This repository demonstrates four different approaches for extending pretrained ResNet models with additional classes while leveraging transfer learning.

## Approaches

### 1. Simple Classifier Extension (`approach_1_simple_extension`)
**Best for:** Adding a few new classes with straightforward implementation

**How it works:**
- Replace the final classifier layer with a larger one
- Copy existing weights for original classes (0-999 for ImageNet)
- Initialize new class weights with small random values
- Fine-tune the classifier while optionally freezing the backbone

**Pros:**
- Simple to implement and understand
- Preserves original class knowledge
- Fast training when backbone is frozen

**Cons:**
- Requires retraining the entire classifier
- May forget some original knowledge if not careful

### 2. LoRA (Low-Rank Adaptation) Extension (`approach_2_lora_extension`)
**Best for:** Limited computational resources, preserving original model

**How it works:**
- Uses Parameter-Efficient Fine-Tuning (PEFT)
- Adds small trainable matrices to existing layers
- Original model weights remain frozen
- Only trains ~1% of the parameters

**Pros:**
- Extremely parameter-efficient
- Fast training and low memory usage
- Original model completely preserved
- Can easily switch between different LoRA adapters

**Cons:**
- Slightly more complex setup
- May have limited capacity for very different new classes

### 3. Dual Heads Approach (`approach_3_dual_heads`)
**Best for:** When you need to distinguish between original and new classes

**How it works:**
- Keeps original classifier separate from new classifier
- Two separate heads: one for ImageNet classes, one for new classes
- Concatenates outputs at inference time
- Can track which head made the prediction

**Pros:**
- Clear separation between original and new knowledge
- Can apply different training strategies to each head
- Useful for domain-specific applications

**Cons:**
- More complex architecture
- Requires careful handling of the dual outputs

### 4. New Classes Only (`approach_4_new_classes_only`)
**Best for:** Specific tasks where ImageNet classes are not needed

**How it works:**
- Completely replaces the classifier with one for custom classes only
- Discards all ImageNet classes (0-999)
- Creates a much smaller and more efficient model
- Maps your classes directly to indices 0, 1, 2, etc.

**Pros:**
- Extremely efficient and lightweight
- Fast inference and training
- Clean class mapping (no need to handle ImageNet indices)
- Perfect for domain-specific applications
- Much smaller model size

**Cons:**
- Cannot classify original ImageNet classes
- Less versatile than other approaches

**Use Cases:**
- Medical image classification (normal, disease1, disease2)
- Quality control (good, defect_type1, defect_type2)
- Custom object detection (apple, orange, banana)
- Document classification (invoice, receipt, contract)

## File Structure

```
├── add_new_class_lora_resnet.py    # Main implementation with all three approaches
├── example_add_new_classes.py      # Complete usage examples and demonstrations
├── identify_resnet.py              # Original ResNet classification code
└── requirements.txt                # Required dependencies
```

## Usage Examples

### Quick Start
```python
from add_new_class_lora_resnet import ResNetWithNewClasses

# Add 5 new classes to ImageNet pretrained ResNet50
model = ResNetWithNewClasses(
    base_model_name="resnet50",
    original_num_classes=1000,  # ImageNet
    new_num_classes=1005,       # Add 5 new classes
    use_lora=False
)

# Freeze backbone for transfer learning
model.freeze_backbone(freeze=True)

# Get optimizer
optimizer = model.get_optimizer(learning_rate=1e-3)
```

### LoRA Approach
```python
# Parameter-efficient fine-tuning with LoRA
lora_model = ResNetWithNewClasses(
    base_model_name="resnet50",
    original_num_classes=1000,
    new_num_classes=1005,
    use_lora=True  # Enable LoRA
)

# Only LoRA parameters will be trained
optimizer = lora_model.get_optimizer(learning_rate=1e-4)
```

### New Classes Only Approach
```python
# For domain-specific tasks where ImageNet classes aren't needed
new_only_model = ResNetWithNewClasses(
    base_model_name="resnet50",
    original_num_classes=1000,
    new_num_classes=None,  # Don't extend in init
    use_lora=False,
)

# Replace classifier to handle only 5 custom classes
new_only_model.new_classes_only(num_new_classes=5)
new_only_model.freeze_backbone(freeze=True)

# Your classes are now mapped as:
# Class 0: Your custom class 1
# Class 1: Your custom class 2
# Class 2: Your custom class 3
# Class 3: Your custom class 4
# Class 4: Your custom class 5
```

### Training Loop
```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # Training
    train_loss, train_acc = model.train_step(
        train_loader, optimizer, criterion, epoch
    )
    
    # Validation
    val_loss, val_acc = model.validate(val_loader, criterion)
    
    print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Save the model
model.save_model("path/to/your/extended_model.pth")
```

### Prediction
```python
# For extended classes (approaches 1-3)
class_names = []
class_names.extend([f"imagenet_class_{i}" for i in range(1000)])  # Original classes
class_names.extend(["my_new_class_1", "my_new_class_2", "my_new_class_3"])  # New classes

result = model.predict("path/to/image.jpg", class_names)
print(f"Predicted: {result['predicted_class_name']} (confidence: {result['confidence']:.3f})")

# For new classes only (approach 4)
custom_class_names = ["apple", "orange", "banana", "grape", "strawberry"]
result = new_only_model.predict("path/to/image.jpg", custom_class_names)
print(f"Predicted: {result['predicted_class_name']} (confidence: {result['confidence']:.3f})")
```

## Key Considerations

### Dataset Size
- **Small dataset (< 1000 images per new class):** Freeze backbone, only train classifier
- **Medium dataset (1000-10000 images):** Use LoRA or train with very low learning rate
- **Large dataset (> 10000 images):** Fine-tune entire model with different learning rates

### Computational Resources
- **Limited resources:** Use LoRA approach
- **Moderate resources:** Freeze backbone + simple extension
- **High resources:** Full fine-tuning with all approaches available

### Class Relationships
- **Related to ImageNet classes:** Simple extension works well
- **Very different from ImageNet:** Consider dual heads or domain adaptation
- **Mix of related and unrelated:** Dual heads approach
- **Completely unrelated/domain-specific:** New classes only approach

### Task Requirements
- **Need ImageNet classes:** Use approaches 1, 2, or 3
- **Only custom classes needed:** Use approach 4 (new classes only)
- **Want maximum efficiency:** Use approach 4 for smallest model size
- **Need flexibility:** Use dual heads to switch between different classification modes

## Installation

```bash
pip install torch torchvision timm peft pillow
```

## Dependencies

- `torch`: PyTorch framework
- `torchvision`: Computer vision utilities
- `timm`: Pretrained models library
- `peft`: Parameter-Efficient Fine-Tuning library
- `pillow`: Image processing

## Advanced Features

### Custom Learning Rates
The implementation supports different learning rates for backbone and classifier:
```python
optimizer = model.get_optimizer(learning_rate=1e-3)
# Backbone gets 1e-4, classifier gets 1e-3
```

### Model Saving/Loading
```python
# Save
model.save_model("extended_resnet.pth")

# Load
model.load_model("extended_resnet.pth")
```

### Gradual Unfreezing
```python
# Start with frozen backbone
model.freeze_backbone(freeze=True)
# Train for a few epochs...

# Then unfreeze for fine-tuning
model.freeze_backbone(freeze=False)
# Continue training with lower learning rate...
```

## Tips for Success

1. **Start Simple:** Begin with the simple extension approach
2. **Use Transfer Learning:** Always start with frozen backbone for small datasets
3. **Monitor Overfitting:** Use validation set to prevent overfitting on new classes
4. **Class Balance:** Ensure balanced representation of old and new classes during training
5. **Data Augmentation:** Use appropriate augmentation for your new classes
6. **Gradual Learning:** Start with frozen backbone, then gradually unfreeze layers

## Troubleshooting

### Common Issues
1. **Memory errors:** Reduce batch size or use LoRA approach
2. **Poor performance on original classes:** Use lower learning rates or more balanced training
3. **Slow convergence:** Check learning rates and ensure proper data preprocessing
4. **LoRA not working:** Verify PEFT installation and configuration

### Performance Tips
- Use mixed precision training for faster computation
- Implement proper data augmentation for new classes
- Consider using larger image resolutions for better accuracy
- Monitor training with proper logging and checkpointing

## Contributing

Feel free to contribute improvements, additional approaches, or bug fixes to this implementation.

## License

This code is provided for educational and research purposes.
