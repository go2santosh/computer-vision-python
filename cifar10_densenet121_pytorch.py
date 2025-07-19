# Import necessary libraries and modules
import torchvision.models as models
from training.model_trainer import train_model_on_cifar10
import sys
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Define the transformation for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # DenseNet requires 224x224 input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize
    ])
    
    # Begin training the model
    print(f"Starting training...")
    sys.stdout.flush()
    model=models.densenet121(pretrained=False)
    train_model_on_cifar10(
        model=model, 
        model_name="cifar10_densenet121_pytorch", 
        transform=transform)
    print(f"Completed training.\n")
    sys.stdout.flush()