import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import timm
from peft import LoraConfig, get_peft_model, TaskType
import os
from PIL import Image
import json


class ResNetWithNewClasses:
    """
    A class to extend a pretrained ResNet model with new classes using different approaches.
    """

    def __init__(
        self,
        base_model_name="resnet50",
        original_num_classes=1000,
        new_num_classes=None,
        use_lora=False,
    ):
        """
        Initialize ResNet with capability to add new classes.

        Args:
            base_model_name (str): Name of the base ResNet model
            original_num_classes (int): Number of classes in original model (ImageNet = 1000)
            new_num_classes (int): Total number of classes after adding new ones
            use_lora (bool): Whether to use LoRA for parameter-efficient fine-tuning
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.original_num_classes = original_num_classes
        self.new_num_classes = new_num_classes
        self.use_lora = use_lora

        # Load pretrained model
        self.model = timm.create_model(base_model_name, pretrained=True)
        print(
            f"Loaded pretrained {base_model_name} with {self.model.num_classes} classes"
        )

        # Store original classifier for reference
        self.original_classifier = self.model.get_classifier()

        if new_num_classes:
            self._extend_classifier()

        if use_lora:
            self._apply_lora()

        self.model = self.model.to(self.device)

        # Define preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _extend_classifier(self):
        """
        Extend the classifier to accommodate new classes.
        Three approaches are implemented:
        1. Replace entire classifier (approach 1)
        2. Add new head while keeping original (approach 2)
        3. Expand existing classifier (approach 3)
        """
        print(
            f"Extending classifier from {self.original_num_classes} to {self.new_num_classes} classes"
        )

        # Get the feature dimension
        if hasattr(self.model, "fc"):
            feature_dim = self.model.fc.in_features
        elif hasattr(self.model, "classifier"):
            feature_dim = self.model.classifier.in_features
        elif hasattr(self.model, "head"):
            feature_dim = self.model.head.in_features
        else:
            raise ValueError("Could not find classifier layer")

        # Approach 1: Replace entire classifier (simplest)
        new_classifier = nn.Linear(feature_dim, self.new_num_classes)

        # Initialize new classifier weights
        # Copy weights for original classes if possible
        if self.new_num_classes > self.original_num_classes:
            with torch.no_grad():
                # Copy original weights
                new_classifier.weight[: self.original_num_classes] = (
                    self.original_classifier.weight
                )
                new_classifier.bias[: self.original_num_classes] = (
                    self.original_classifier.bias
                )

                # Initialize new class weights (small random values)
                nn.init.normal_(
                    new_classifier.weight[self.original_num_classes :], std=0.01
                )
                nn.init.constant_(new_classifier.bias[self.original_num_classes :], 0)

        # Replace the classifier
        if hasattr(self.model, "fc"):
            self.model.fc = new_classifier
        elif hasattr(self.model, "classifier"):
            self.model.classifier = new_classifier
        elif hasattr(self.model, "head"):
            self.model.head = new_classifier

    def _apply_lora(self):
        """
        Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
        """
        print("Applying LoRA configuration...")

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,  # scaling parameter
            target_modules=["fc", "classifier", "head"],  # target the classifier layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA applied successfully")

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def approach_2_dual_heads(self):
        """
        Approach 2: Keep original classifier and add a new head for new classes.
        This allows the model to handle both original and new classes separately.
        """
        if hasattr(self.model, "fc"):
            feature_dim = self.model.fc.in_features
            # Keep original classifier
            self.model.original_fc = self.model.fc
            # Add new classifier for new classes only
            num_new_classes = self.new_num_classes - self.original_num_classes
            self.model.new_fc = nn.Linear(feature_dim, num_new_classes)
            # Remove the original fc to avoid conflicts
            self.model.fc = None

        # Override forward method
        def forward_dual_heads(x):
            # Get features from backbone
            x = self.model.forward_features(x)
            x = self.model.global_pool(x)
            if self.model.drop_rate:
                x = self.model.dropout(x)

            # Get predictions from both heads
            original_logits = self.model.original_fc(x)
            new_logits = self.model.new_fc(x)

            # Concatenate logits
            return torch.cat([original_logits, new_logits], dim=1)

        self.model.forward = forward_dual_heads

    def new_classes_only(self, num_new_classes):
        """
        Approach 3: Replace classifier with one that only handles new classes.
        Original ImageNet classes are discarded.

        Args:
            num_new_classes (int): Number of new classes to classify
        """
        print(f"Replacing classifier to handle only {num_new_classes} new classes")

        # Get the feature dimension
        if hasattr(self.model, "fc"):
            feature_dim = self.model.fc.in_features
            # Replace with new classifier for new classes only
            self.model.fc = nn.Linear(feature_dim, num_new_classes)
        elif hasattr(self.model, "classifier"):
            feature_dim = self.model.classifier.in_features
            self.model.classifier = nn.Linear(feature_dim, num_new_classes)
        elif hasattr(self.model, "head"):
            feature_dim = self.model.head.in_features
            self.model.head = nn.Linear(feature_dim, num_new_classes)
        else:
            raise ValueError("Could not find classifier layer")

        # Initialize weights with small random values
        with torch.no_grad():
            if hasattr(self.model, "fc"):
                nn.init.normal_(self.model.fc.weight, std=0.01)
                nn.init.constant_(self.model.fc.bias, 0)
            elif hasattr(self.model, "classifier"):
                nn.init.normal_(self.model.classifier.weight, std=0.01)
                nn.init.constant_(self.model.classifier.bias, 0)
            elif hasattr(self.model, "head"):
                nn.init.normal_(self.model.head.weight, std=0.01)
                nn.init.constant_(self.model.head.bias, 0)

        print(f"New classifier initialized with {num_new_classes} output classes")

    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze the backbone (feature extraction layers).

        Args:
            freeze (bool): Whether to freeze the backbone
        """
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = not freeze

        # Unfreeze classifier layers
        if hasattr(self.model, "fc") and self.model.fc is not None:
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True

        # If using dual heads approach
        if hasattr(self.model, "original_fc"):
            for param in self.model.original_fc.parameters():
                param.requires_grad = True
        if hasattr(self.model, "new_fc"):
            for param in self.model.new_fc.parameters():
                param.requires_grad = True

        print(f"Backbone {'frozen' if freeze else 'unfrozen'}")

    def get_optimizer(self, learning_rate=1e-4, weight_decay=1e-4):
        """
        Get optimizer for training.

        Args:
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if self.use_lora:
            # Only optimize LoRA parameters
            optimizer = optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            # Different learning rates for backbone and classifier
            backbone_params = []
            classifier_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "fc" in name or "classifier" in name or "head" in name:
                        classifier_params.append(param)
                    else:
                        backbone_params.append(param)

            optimizer = optim.AdamW(
                [
                    {
                        "params": backbone_params,
                        "lr": learning_rate * 0.1,
                    },  # Lower LR for backbone
                    {
                        "params": classifier_params,
                        "lr": learning_rate,
                    },  # Higher LR for classifier
                ],
                weight_decay=weight_decay,
            )

        return optimizer

    def train_step(self, dataloader, optimizer, criterion, epoch):
        """
        Training step for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(dataloader)
        print(
            f"Epoch {epoch} - Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        return avg_loss, accuracy

    def validate(self, dataloader, criterion):
        """
        Validation step.

        Args:
            dataloader: Validation data loader
            criterion: Loss function
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

        return avg_loss, accuracy

    def predict(self, image_path, class_names=None):
        """
        Predict class for a single image.

        Args:
            image_path (str): Path to the image
            class_names (list): List of class names

        Returns:
            dict: Prediction results
        """
        self.model.eval()

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[predicted_class].item()

        result = {
            "predicted_class_id": predicted_class,
            "confidence": confidence,
            "all_probabilities": probabilities.cpu().numpy(),
        }

        if class_names:
            result["predicted_class_name"] = class_names[predicted_class]

        return result

    def save_model(self, save_path):
        """
        Save the model.

        Args:
            save_path (str): Path to save the model
        """
        if self.use_lora:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
            # Save entire model
            torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        Load the model.

        Args:
            load_path (str): Path to load the model from
        """
        if self.use_lora:
            # This would require proper LoRA loading implementation
            print("LoRA model loading - implement based on your specific needs")
        else:
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"Model loaded from {load_path}")


# Example usage and demonstration
def example_usage():
    """
    Example of how to use the ResNetWithNewClasses.
    """
    print("=== ResNet New Class Addition Example ===")

    # Example 1: Add new classes to ImageNet pretrained model
    print("\n1. Adding new classes to pretrained ResNet50...")
    model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,  # ImageNet classes
        new_num_classes=1010,  # Add 10 new classes
        use_lora=False,
    )

    # Freeze backbone for transfer learning
    model.freeze_backbone(freeze=True)

    # Example 2: Using LoRA for parameter-efficient fine-tuning
    print("\n2. Using LoRA for parameter-efficient fine-tuning...")
    lora_model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,
        new_num_classes=1005,  # Add 5 new classes
        use_lora=True,
    )

    # Example 3: Dual heads approach
    print("\n3. Dual heads approach...")
    dual_model = ResNetWithNewClasses(
        base_model_name="resnet18",
        original_num_classes=1000,
        new_num_classes=1003,  # Add 3 new classes
    )
    dual_model.approach_2_dual_heads()

    # Example 4: New classes only (discard original ImageNet classes)
    print("\n4. New classes only approach...")
    new_only_model = ResNetWithNewClasses(
        base_model_name="resnet50",
        original_num_classes=1000,
        new_num_classes=None,  # Don't extend in init
        use_lora=False,
    )
    # Replace classifier to handle only 5 custom classes
    new_only_model.new_classes_only(num_new_classes=5)
    # Freeze backbone for transfer learning
    new_only_model.freeze_backbone(freeze=True)

    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    example_usage()
