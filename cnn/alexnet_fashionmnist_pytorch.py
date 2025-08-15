# Import necessary libraries and modules
from models.alexnet_model import AlexNetModel
from training.model_trainer import train_model_on_fashionmnist
import sys
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Define the transformation for the dataset
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Iterate for each version from 1 to 5
    for i in range(1, 6):
        version = str(i)
        print(f"Starting experiment for version {version}...")
        sys.stdout.flush()
        model=AlexNetModel()
        train_model_on_fashionmnist(
            model=model, 
            model_name="alexnet_relu_fashionmnist_pytorch", 
            transform=transform,
            version=version)
        print(f"Completed experiment for version {version}.\n")
        sys.stdout.flush()