# Importing necessary modules
import torch
from torch.utils.data import DataLoader
from datasets.fashionmnist_dataset import FashionMNISTDataset
from models.alexnet_model import AlexNetModel
from training.optimization_loop import OptimizationLoop
import torchvision.transforms as transforms
import sys

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# Check if a GPU is available and set the device accordingly
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
print(f"Using {device} device")

# Initialize the model, loss function, and optimizer
print("Initializing the model, loss function, and optimizer...")
model = AlexNetModel().to(device)
# Define hyperparameters
learning_rate = 1e-3
epochs = 50
batch_size=32

# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss()  

# Define optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Define the experiment name
experiment_name = "alexnet_fashionmnist_pytorch_ce_adam"

# Load the training and validation datasets
train_dataset = FashionMNISTDataset(csv_file='datasets/fashion-mnist/fashion-mnist_train.csv', transform=transform)
validate_dataset = FashionMNISTDataset(csv_file='datasets/fashion-mnist/fashion-mnist_validate.csv', transform=transform)

# Create DataLoaders for the datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

# Train the model for optimal performance
print("Training the model for optimal performance...")
optimal_epochs = 14
optimizationLoop = OptimizationLoop(model, loss_fn, device)
for t in range(optimal_epochs):
    print(f"Optimal Epoch {t+1}\n-------------------------------")
    train_loss = optimizationLoop.train_loop(train_dataloader, optimizer, batch_size)
    eval_loss, eval_accuracy = optimizationLoop.evaluate_loop(validate_dataloader)
    # print the training and evaluation loss 
    print('Epoch:', t+1, 'Training loss:', train_loss, 'Evaluation loss:', eval_loss, 'Accuracy:', eval_accuracy)
    sys.stdout.flush()

# Print the model's parameters and their sizes
print("Trained Model Parameters:")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:]} \n")
sys.stdout.flush()

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), experiment_name + '_model_weights.pth')

# Load the model
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.to(device)
    return model

# Load the model from the saved state
print("Loading the model...")
model = load_model(AlexNetModel(), experiment_name + '_model_weights.pth', device)

# Load the test dataset
test_dataset = FashionMNISTDataset(csv_file='datasets/fashion-mnist/fashion-mnist_test.csv', transform=transform)

# Create DataLoader for the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test the model
print("Starting testing...")
optimizationLoop = OptimizationLoop(model, loss_fn, device)
test_loss, test_accuracy = optimizationLoop.evaluate_loop(test_dataloader)

# Create SummaryWriter for TensorBoard logging
writer = SummaryWriter('runs/' + experiment_name)

# Write the test loss and accuracy to TensorBoard
writer.add_scalar('Test Loss', test_loss)
writer.add_scalar('Test Accuracy', test_accuracy)
writer.flush()
print("Testing complete.")
sys.stdout.flush()