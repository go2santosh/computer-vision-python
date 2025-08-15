# Create a Conditional GAN model
from email import generator
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils
import os
from torch.utils.tensorboard import SummaryWriter
import sys

# Set configurations
mnist_data_root = './datasets' # Root directory for the MNIST dataset
experiment_name = 'conditional_gan_mnist_pytorch' # Name of the experiment
saved_models_dir = './saved_models/' + experiment_name # Model weight save directory
output_dir = './data/' + experiment_name + '_output' # Output directory for generated images
tensorboard_log_dir = './runs/' + experiment_name # TensorBoard log directory

# Check if a GPU is available and set the device accordingly
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
print(f"Using {device} device")
sys.stdout.flush()

# Define a conditional generator model
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim):
        super(ConditionalGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise and labels
        input = torch.cat((noise, labels), dim=1)
        return self.model(input)
    
# Define a conditional discriminator model
class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Concatenate image and labels
        input = torch.cat((x, labels), dim=1)
        return self.model(input)
    
# Train the Conditional GAN model
def train_conditional_gan(generator, discriminator, data_loader, num_epochs=100):
    # Move models to the appropriate device
    generator.to(device)
    discriminator.to(device)

    # Define optimizers outside the training loop
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    writer = SummaryWriter(tensorboard_log_dir)
    for epoch in range(num_epochs):
        # Lists to store losses       
        D_losses, G_losses = [], []

        for real_images, labels in data_loader:
            # Move data to the appropriate device
            real_images = real_images.to(device)
            labels = torch.eye(10, device=device)[labels]  # Convert labels to one-hot encoding

            # Train the discriminator
            noise = torch.randn(real_images.size(0), 100, device=device)
            fake_images = generator(noise, labels)
            d_loss = train_discriminator(discriminator, real_images, fake_images, labels, optimizer_d)

            # Train the generator
            g_loss = train_generator(generator, discriminator, noise, labels, optimizer_g)

            # Store losses
            D_losses.append(d_loss)
            G_losses.append(g_loss)

        # Print the average losses for the epoch
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch), num_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
        sys.stdout.flush()

        # Log the losses to TensorBoard
        writer.add_scalars('Discriminator vs Generator Loss', {'Discriminator': torch.mean(torch.FloatTensor(D_losses)), 'Generator': torch.mean(torch.FloatTensor(G_losses))}, epoch)
        writer.flush()

        # Save the trained generator and discriminator models every 5 epochs
        if (epoch + 1) % 5 == 0:
            if not os.path.exists(saved_models_dir):
                os.makedirs(saved_models_dir)
            torch.save(generator.state_dict(), f'{saved_models_dir}/conditional_gan_generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'{saved_models_dir}/conditional_gan_discriminator_epoch_{epoch+1}.pth')

        # Save the image generated at every 5 epochs
        if (epoch + 1) % 5 == 0:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with torch.no_grad():
                noise = torch.randn(80, 100, device=device)  # 80 = 10 digits * 8 images per digit
                # Generate labels for digits 0 to 9, repeated 8 times each
                labels = torch.eye(10, device=device).repeat_interleave(8, dim=0)
                fake_images = generator(noise, labels)
                fake_images = fake_images.view(-1, 1, 28, 28)
                # Save the images in 10 rows (one row per digit) and 8 columns
                torchvision.utils.save_image(fake_images, f'{output_dir}/fake_images_epoch_{epoch+1}.png', nrow=8, normalize=True)

# Function to train the discriminator
def train_discriminator(discriminator, real_images, fake_images, labels, optimizer):
    criterion = nn.BCELoss()

    # Prepare labels
    real_labels = torch.ones(real_images.size(0), 1, device=real_images.device)
    fake_labels = torch.zeros(fake_images.size(0), 1, device=fake_images.device)

    # Forward pass for real images
    outputs = discriminator(real_images.view(real_images.size(0), -1), labels)
    d_loss_real = criterion(outputs, real_labels)

    # Forward pass for fake images
    outputs = discriminator(fake_images.detach().view(fake_images.size(0), -1), labels)
    d_loss_fake = criterion(outputs, fake_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer.step()

    return d_loss.item()

# Function to train the generator
def train_generator(generator, discriminator, noise, labels, optimizer):
    criterion = nn.BCELoss()

    # Prepare labels
    valid_labels = torch.ones(noise.size(0), 1, device=noise.device)

    # Forward pass
    fake_images = generator(noise, labels)
    outputs = discriminator(fake_images, labels)
    g_loss = criterion(outputs, valid_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    g_loss.backward()
    optimizer.step()

    return g_loss.item()    

# main function to run the Conditional GAN training
if __name__ == "__main__":
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust for 1 channel
    ])

    # Dataset for MNIST training images
    train_dataset = datasets.MNIST(root=mnist_data_root, train=True, transform=transform, download=True)

    # DataLoader for MNIST training images
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    noise_dim = 100
    label_dim = 10  # MNIST has 10 classes (0-9)
    generator = ConditionalGenerator(noise_dim, label_dim)
    discriminator = ConditionalDiscriminator(784, label_dim)

    # Train the Conditional GAN
    max_epochs = 70
    print(f"Training Conditional GAN on MNIST dataset for {max_epochs} epochs.")
    sys.stdout.flush()
    train_conditional_gan(generator, discriminator, train_loader, num_epochs=max_epochs)
