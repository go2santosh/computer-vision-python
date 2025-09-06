# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import sys
import os

# Experiment settings
batch_size = 64 # Batch size
image_size = 64
image_dim = image_size * image_size * 3
learning_rate = 0.0002 # Learning rate
z_dim = 100 # Dimension of the noise vector
max_epochs = 200 # Number of epochs to train the model
dataset_root = '../datasets/celeba' # Root directory for the CelebA dataset
saved_models_dir = '../saved_models' # Model weight save directory
experiment_name = 'gan_celeba_pytorch' # Name of the experiment
output_dir = '../data/' + experiment_name + '_output' # Output directory for generated images
tensorboard_log_dir = '../runs/' + experiment_name # TensorBoard log directory

# Check if a GPU is available and set the device accordingly
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
print(f"Using {device} device")

# Data transformation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset for CelebA training images
train_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

# DataLoader for CelebA training images
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator model
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# Initialize the generator and discriminator
generator_model = Generator(g_input_dim = z_dim, g_output_dim = image_dim).to(device)
discriminator_model = Discriminator(image_dim).to(device)

# Show the generator and discriminator model architectures
print(generator_model)
print(discriminator_model) 
sys.stdout.flush()

# Loss function
criterion = nn.BCELoss() 

# Optimizers
generator_optimizer = optim.Adam(generator_model.parameters(), lr = learning_rate)
discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr = learning_rate)

# Function to train the discriminator
def D_train(x):
    # Reset gradients of model
    discriminator_model.zero_grad()

    # Take the real images
    current_batch_size = x.size(0)
    x_real = x.view(current_batch_size, -1)
    y_real = torch.ones(current_batch_size, 1, device=x.device)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    # Let the discriminator predict the real images
    D_output = discriminator_model(x_real)
    D_real_loss = criterion(D_output, y_real)

    # Let the discriminator predict the fake images 
    z = Variable(torch.randn(batch_size, z_dim).to(device))
    x_fake, y_fake = generator_model(z), Variable(torch.zeros(batch_size, 1).to(device))
    D_output = discriminator_model(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # Calculate the total discriminator loss, backpropagate, and update the parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    discriminator_optimizer.step()
    
    # Return the discriminator loss
    return  D_loss.data.item()

# Function to train the generator
def G_train(x):
    # Reset gradients of model
    generator_model.zero_grad()

    # Generate random noise with label 1 (fake)
    z = Variable(torch.randn(batch_size, z_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    # Let the generator generate fake images
    G_output = generator_model(z)

    # Let the discriminator predict the fake images
    D_output = discriminator_model(G_output)

    # Calculate the generator loss
    G_loss = criterion(D_output, y)

    # Gradient backpropagation and optimization of the generator's parameters
    G_loss.backward()
    generator_optimizer.step()
        
    # Return the generator loss
    return G_loss.data.item()

# Training loop
writer = SummaryWriter(tensorboard_log_dir)
for epoch in range(1, max_epochs+1):    
    # Lists to store losses       
    D_losses, G_losses = [], []

    # Iterate through the training data 
    for batch_idx, (x, _) in enumerate(train_loader):
        # Train the discriminator and store the loss 
        discriminator_loss = D_train(x)
        D_losses.append(discriminator_loss)
        # Train the generator and store the loss
        generator_loss = G_train(x)
        G_losses.append(generator_loss)
        # Print both the losses at batch_idx
        print('Batch [%d]: loss_d: %.3f, loss_g: %.3f' % (
            (batch_idx), discriminator_loss, generator_loss))

    # Print the average losses for the epoch
    print('Epoch [%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), max_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    sys.stdout.flush()

    # Log the losses to TensorBoard
    writer.add_scalars('Discriminator vs Generator Loss', {'Discriminator': torch.mean(torch.FloatTensor(D_losses)), 'Generator': torch.mean(torch.FloatTensor(G_losses))}, epoch)
    writer.flush()

    # Save the generator and discriminator models for every epoch
    torch.save(generator_model.state_dict(), saved_models_dir + '/' + experiment_name + '_G_epoch_' + str(epoch) + '.pth')
    torch.save(discriminator_model.state_dict(), saved_models_dir + '/' + experiment_name + '_D_epoch_' + str(epoch) + '.pth')

    # Save image for every epoch
    with torch.no_grad():
        test_z = Variable(torch.randn(batch_size, z_dim).to(device))
        generated = generator_model(test_z)
        os.makedirs(output_dir, exist_ok=True)
        save_image(generated.view(generated.size(0), 3, image_size, image_size),  output_dir + '/' + experiment_name + '_output_' + str(epoch) + '.png')

# Close the TensorBoard writer
writer.close()
print("Training complete. Models and images saved.")
sys.stdout.flush()