import argparse
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.fashionmnist_dataset import FashionMNISTDataset

from training.optimization_loop import OptimizationLoop
from torch.utils.tensorboard import SummaryWriter

# Private method to use program arguments
def __get_args():
    parser = argparse.ArgumentParser(description='Trains a Deep Learning model on a dataset.')
    # Add arguments
    parser.add_argument('--version', type=str, default=None, help='Version of the model.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer. Default is 1e-3.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model. Default is 10.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training. Default is 32.')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce'], help='Loss function to use for training. Default is CrossEntropyLoss (ce).')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'], help='Optimizer to use for training. Default is adam.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training if available.')
    parser.add_argument('--use_tensorboard', action='store_true', help='Use TensorBoard for logging.')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model after training.')
    
    # Parse the arguments
    args = parser.parse_args()

    # Print the parsed arguments
    print("Parsed arguments:")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Loss Function: {args.loss}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Use GPU: {args.use_gpu}")
    print(f"Use TensorBoard: {args.use_tensorboard}")
    print(f"Save Model: {args.save_model}")
    sys.stdout.flush()

    # Return the parsed arguments
    return args

def train_model_on_fashionmnist(model, model_name, transform=None, version=None):
    # Parse command line arguments
    args = __get_args()
    epochs = args.epochs
    batch_size = args.batch_size
    loss_fn_name = args.loss
    optimizer_name = args.optimizer
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu
    use_tensorboard = args.use_tensorboard
    save_model = args.save_model

    # Use default transformation if none is provided
    if transform is None:
        # Define the transformation for the dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

    # Load the training and validation datasets
    print("Creating FashionMNIST datasets...")
    custom_data_root = './datasets'
    train_dataset = torchvision.datasets.FashionMNIST(root=custom_data_root, train=True, transform=transform, download=False)
    validate_dataset = torchvision.datasets.FashionMNIST(root=custom_data_root, train=False, transform=transform, download=False)

    # Create DataLoaders for the datasets
    print("Creating DataLoaders for training and validation datasets...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    # Create an instance of ModelTrainer
    print("Initializing ModelTrainer...")
    if version is None:
        model_trainer = ModelTrainer(
            model_name=model_name, 
            model=model,
            epochs=epochs,
            loss_fn_name=loss_fn_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            use_tensorboard=args.use_tensorboard,
            save_model=save_model)
    else:
        model_trainer = ModelTrainer(
            model_name=model_name, 
            model=model, 
            epochs=epochs,
            loss_fn_name=loss_fn_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            use_tensorboard=use_tensorboard,
            save_model=save_model,
            version=version)

    # Begin training with the model name and model instance
    model_trainer.begin_training(train_dataloader=train_dataloader, validate_dataloader=validate_dataloader)

def train_model_on_cifar10(model, model_name, transform=None, version=None):
    # Parse command line arguments
    args = __get_args()
    epochs = args.epochs
    loss_fn_name = args.loss
    optimizer_name = args.optimizer
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu
    use_tensorboard = args.use_tensorboard
    batch_size = args.batch_size
    save_model = args.save_model

    # Use default transformation if none is provided
    if transform is None:
        # Define the transformation for the dataset
        transform = transforms.Compose([
            transforms.ToTensor(), # Convert to tensor
            transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] image = (image - mean) / std
        ])
        
    # Create train_dataloader and validate_dataloader
    print("Loading CIFAR-10 dataset...")
    data_root = 'datasets/cifar-10'
    import torchvision
    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    validate_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    # Create DataLoaders for the datasets
    print("Creating DataLoaders for training and validation datasets...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    
    # Create an instance of ModelTrainer
    print("Initializing ModelTrainer...")
    if version is None:
        model_trainer = ModelTrainer(
            model_name=model_name, 
            model=model,
            epochs=epochs,
            loss_fn_name=loss_fn_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            use_tensorboard=args.use_tensorboard,
            save_model=save_model)
    else:
        model_trainer = ModelTrainer(
            model_name=model_name, 
            model=model, 
            epochs=epochs,
            loss_fn_name=loss_fn_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            use_tensorboard=use_tensorboard,
            save_model=save_model,
            version=version)

    # Begin training with the model name and model instance
    model_trainer.begin_training(train_dataloader=train_dataloader, validate_dataloader=validate_dataloader)

# Class to train a deep learning model
class ModelTrainer:
    def __init__(self, model_name, model, epochs, loss_fn_name, optimizer_name, learning_rate,
                 use_gpu=False, use_tensorboard=False, save_model=False, version=None):
        sys.stdout.flush()
        self.model_name = model_name
        self.model = model
        self.epochs = epochs
        self.loss_fn_name = loss_fn_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.use_tensorboard = use_tensorboard
        self.save_model = save_model
        self.version = version
        # Get the experiment name
        self.experiment_name = self.__get_experiment_name(self.model_name, loss_fn_name, optimizer_name, self.version)
        # Initialize TensorBoard writer if enabled
        self.writer = self.__get_writer(use_tensorboard, self.experiment_name)
        # Configure the device to use cpu or gpu
        self.device = self.__get_device(use_gpu)
        # Move the model to the appropriate device
        self.model = model.to(self.device)
        # Obtain loss function and optimizer
        self.loss_fn = self.__get_loss_fn(loss_fn_name)
        self.optimizer = self.__get_optimizer(self.model, optimizer_name, learning_rate)

    # Private method to configure the device
    def __get_device(self, use_gpu):
        # Check if a GPU is available
        device = "cpu"
        if use_gpu:
            if torch.accelerator.is_available():
                device = torch.accelerator.current_accelerator().type
                print(f"Accelerator device available is: {device}")
                sys.stdout.flush()
            else:
                print("Accelerator device not available.")
                sys.stdout.flush()
        else:
            print("Accelerator device not requested.")
            sys.stdout.flush()
        return device
    
    # Private method to get loss function
    def __get_loss_fn(self, loss):
        # Define loss function
        if loss == 'ce':
            loss_fn = torch.nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss as the loss function.")
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            print(f"Cannot use loss function: {loss}. Defaulting to CrossEntropyLoss.")
        return loss_fn

    # Private method to get the optimizer
    def __get_optimizer(self, model, optimizer_name, learning_rate): 
        # Define optimizer
        optimizer = None
        if optimizer_name == 'sgd':
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name == 'adamw':
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        elif optimizer_name == 'adam':
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            print(f"Cannot use optimizer: {optimizer_name}. Defaulting to Adam.")
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        sys.stdout.flush()
        return optimizer

    def __get_experiment_name(self, model_name, loss_fn_name, optimizer_name, version=None):
        # Define the experiment name
        experiment_name = model_name + "_" + loss_fn_name + "_" + optimizer_name
        if version:
            experiment_name += "_" + version
        print("Experiment name:", experiment_name)
        sys.stdout.flush()
        return experiment_name

    def __get_writer(self, use_tensorboard, experiment_name):
        # To view, start TensorBoard on the command line with:
        #   tensorboard --logdir=runs
        # ...and open a browser tab to http://localhost:6006/

        # Initialize TensorBoard writer if enabled
        writer = None
        # Create SummaryWriter for TensorBoard logging
        if use_tensorboard:
            print("Using TensorBoard for logging.")
            writer = SummaryWriter('runs/' + experiment_name)
        else:
            print("Not using TensorBoard for logging.")
        sys.stdout.flush()
        return writer

    def begin_training(self, train_dataloader, validate_dataloader):
        print("Starting training...")
        sys.stdout.flush()
        optimizationLoop = OptimizationLoop(self.model, self.loss_fn, self.device)
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = optimizationLoop.train_loop(train_dataloader, self.optimizer)
            eval_loss, eval_accuracy = optimizationLoop.evaluate_loop(validate_dataloader)

            # Log the training and evaluation loss to TensorBoard
            print('Epoch:', t+1, 'Training loss:', train_loss, 'Evaluation loss:', eval_loss, 'Accuracy:', eval_accuracy)
            sys.stdout.flush()

            if self.writer:
                self.writer.add_scalars('Training vs Evaluation Loss', {'Training': train_loss, 'Evaluation': eval_loss}, t+1)
                self.writer.add_scalar('Training Accuracy', eval_accuracy, t+1)
                self.writer.flush()
        print("Training complete.")
        sys.stdout.flush()

        # Save the model if specified
        if self.save_model:
            self.__save_model()

    def __save_model(self):
        # Save the trained model to the specified path
        # Print the model's parameters and their sizes
        print("Trained Model Parameters:")
        for name, param in self.model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:]} \n")
        sys.stdout.flush()

        # Save the model
        model_file_name = self.experiment_name + '_model_weights.pth'
        torch.save(self.model.state_dict(), model_file_name)
        print("Model saved successfully as:", model_file_name)
