import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import json
import urllib.request
import argparse
import os


class ImageClassifier:
    def __init__(self, model_name="resnet50", use_gpu=True):
        """
        Initialize the image classifier with a pre-trained ResNet model.

        Args:
            model_name (str): Name of the ResNet model ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            use_gpu (bool): Whether to use GPU if available
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        print(f"Using device: {self.device}")

        # Load pre-trained model
        if model_name == "resnet18":
            self.model = timm.create_model("resnet18", pretrained=True)
        elif model_name == "resnet34":
            self.model = timm.create_model("resnet34", pretrained=True)
        elif model_name == "resnet50":
            self.model = timm.create_model("resnet50", pretrained=True)
        elif model_name == "resnet101":
            self.model = timm.create_model("resnet101", pretrained=True)
        elif model_name == "resnet152":
            self.model = timm.create_model("resnet152", pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Define image preprocessing
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

        # Load ImageNet class labels
        self.class_labels = self._load_imagenet_labels()

    def _load_imagenet_labels(self):
        """Load ImageNet class labels."""
        try:
            # Try to load from local file first
            if os.path.exists("imagenet_classes.txt"):
                with open("imagenet_classes.txt", "r") as f:
                    return [line.strip() for line in f.readlines()]
            else:
                # Download from internet if not available locally
                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                with urllib.request.urlopen(url) as response:
                    return [
                        line.decode("utf-8").strip() for line in response.readlines()
                    ]
        except Exception as e:
            print(f"Warning: Could not load ImageNet labels: {e}")
            # Return dummy labels if download fails
            return [f"class_{i}" for i in range(1000)]

    def preprocess_image(self, image_path):
        """
        Preprocess an image for ResNet input.

        Args:
            image_path (str): Path to the image file

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
            return input_batch.to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")

    def predict(self, image_path, top_k=5):
        """
        Predict the class of an image.

        Args:
            image_path (str): Path to the image file
            top_k (int): Number of top predictions to return

        Returns:
            list: List of tuples containing (class_name, confidence_score)
        """
        # Preprocess the image
        input_batch = self.preprocess_image(image_path)

        # Make prediction
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top k predictions
        top_prob, top_indices = torch.topk(probabilities, top_k)

        predictions = []
        for i in range(top_k):
            class_idx = top_indices[i].item()
            confidence = top_prob[i].item()
            class_name = (
                self.class_labels[class_idx]
                if class_idx < len(self.class_labels)
                else f"class_{class_idx}"
            )
            predictions.append((class_name, confidence))

        return predictions

    def predict_batch(self, image_paths, top_k=5):
        """
        Predict classes for multiple images.

        Args:
            image_paths (list): List of image file paths
            top_k (int): Number of top predictions to return for each image

        Returns:
            dict: Dictionary mapping image paths to their predictions
        """
        results = {}
        for image_path in image_paths:
            try:
                predictions = self.predict(image_path, top_k)
                results[image_path] = predictions
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = []

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Identify images using pre-trained ResNet model"
    )
    parser.add_argument("image_path", help="Path to the image file or directory")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="ResNet model to use (default: resnet50)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")

    args = parser.parse_args()

    # Initialize classifier
    classifier = ImageClassifier(model_name=args.model, use_gpu=not args.no_gpu)

    # Check if input is a file or directory
    if os.path.isfile(args.image_path):
        # Single image
        print(f"Analyzing image: {args.image_path}")
        try:
            predictions = classifier.predict(args.image_path, args.top_k)
            print(f"\nTop {args.top_k} predictions:")
            for i, (class_name, confidence) in enumerate(predictions, 1):
                print(f"{i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
        except Exception as e:
            print(f"Error: {e}")

    elif os.path.isdir(args.image_path):
        # Directory of images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            os.path.join(args.image_path, f)
            for f in os.listdir(args.image_path)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]

        if not image_files:
            print(f"No image files found in {args.image_path}")
            return

        print(f"Found {len(image_files)} images in {args.image_path}")
        results = classifier.predict_batch(image_files, args.top_k)

        for image_path, predictions in results.items():
            print(f"\n--- {os.path.basename(image_path)} ---")
            if predictions:
                for i, (class_name, confidence) in enumerate(predictions, 1):
                    print(
                        f"{i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)"
                    )
            else:
                print("Failed to process this image")

    else:
        print(f"Error: {args.image_path} is not a valid file or directory")


def run_classification(image_path, model="resnet50", top_k=5, use_gpu=True):
    """
    Run image classification with manual parameters (no argparse).

    Args:
        image_path (str): Path to image file or directory
        model (str): Model name ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        top_k (int): Number of top predictions to show
        use_gpu (bool): Whether to use GPU if available
    """
    # Initialize classifier
    classifier = ImageClassifier(model_name=model, use_gpu=use_gpu)

    # Check if input is a file or directory
    if os.path.isfile(image_path):
        # Single image
        print(f"Analyzing image: {image_path}")
        try:
            predictions = classifier.predict(image_path, top_k)
            print(f"\nTop {top_k} predictions:")
            for i, (class_name, confidence) in enumerate(predictions, 1):
                print(f"{i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
        except Exception as e:
            print(f"Error: {e}")

    elif os.path.isdir(image_path):
        # Directory of images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]

        if not image_files:
            print(f"No image files found in {image_path}")
            return

        print(f"Found {len(image_files)} images in {image_path}")
        results = classifier.predict_batch(image_files, top_k)

        for image_path, predictions in results.items():
            print(f"\n--- {os.path.basename(image_path)} ---")
            if predictions:
                for i, (class_name, confidence) in enumerate(predictions, 1):
                    print(
                        f"{i}. {class_name}: {confidence:.4f} ({confidence*100:.2f}%)"
                    )
            else:
                print("Failed to process this image")

    else:
        print(f"Error: {image_path} is not a valid file or directory")


if __name__ == "__main__":
    run_classification(
        image_path="data/human/a-girl_masterpiece_street_casual-cloth_cool_human_upper-body.png",
        model="resnet50",
        top_k=3,
        use_gpu=True,
    )
