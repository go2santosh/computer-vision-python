# Import necessary libraries and modules
from models.lenet5_model import LeNet5Model
from training.model_trainer import train_model_on_fashionmnist
import sys

if __name__ == "__main__":
    # Iterate for each version from 1 to 5
    for i in range(1, 6):
        version = str(i)
        print(f"Starting experiment for version {version}...")
        sys.stdout.flush()
        model=LeNet5Model()
        train_model_on_fashionmnist(
            model=model, 
            model_name="lenet5_relu_fashionmnist_pytorch", 
            version=version)
        print(f"Completed experiment for version {version}.\n")
        sys.stdout.flush()