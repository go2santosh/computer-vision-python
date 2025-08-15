# Import necessary libraries and modules
from models.simple_mlp_model import SimpleMLPModel
from training.model_trainer import train_model_on_fashionmnist
import sys

if __name__ == "__main__":
    # Iterate for each version from 1 to 5
    for i in range(1, 6):
        version = str(i)
        print(f"Starting experiment for version {version}...")
        sys.stdout.flush()
        model=SimpleMLPModel()
        train_model_on_fashionmnist(
            model=model, 
            model_name="fashionmnist_mlp_relu_pytorch", 
            version=version)
        print(f"Completed experiment for version {version}.\n")
        sys.stdout.flush()