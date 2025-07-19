# Import necessary libraries and modules
import torchvision.models as models
from training.model_trainer import train_model_on_cifar10
import sys
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Define the transformation for the dataset
    transform = transforms.Compose([
        transforms.Resize(224), # Resize to 224x224
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1] image = (image - mean) / std
    ])

    # Begin training the model
    print(f"Starting training...")
    sys.stdout.flush()
    model=models.vgg16(weights=None)
    train_model_on_cifar10(
        model=model, 
        model_name="vgg16_cifar10_pytorch",
        transform=transform,)
    print(f"Completed training.\n")
    sys.stdout.flush()