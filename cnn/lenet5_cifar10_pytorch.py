# Import necessary libraries and modules
from models.lenet5_model import LeNet5Model
from training.model_trainer import train_model_on_cifar10
import sys
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Define the transformation for the dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(28), # Resize to 28x28
        transforms.CenterCrop(28), # Center crop to 28x28
        transforms.ToTensor(), # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] image = (image - mean) / std
    ])

    # Begin training the model
    print(f"Starting training...")
    sys.stdout.flush()
    model=LeNet5Model()
    train_model_on_cifar10(
        model=model, 
        model_name="lenet5_cifar10_pytorch",
        transform=transform)
    print(f"Completed training.\n")
    sys.stdout.flush()