# Import necessary modules
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import sys

# Enable cudnn autotuner for optimized performance on GPU
cudnn.benchmark = True

# Experiment settings
epochs = 200  # Number of training epochs
learning_rate = 0.0005 # Learning rate
num_in_channel=3  # Number of channels in the training images (RGB)
num_gpu = 1 # Number of GPUs available (set to 1 for single GPU)
z_dim = 100 # Size of the latent z vector (input noise to generator)
num_feature_maps_G = 64 # Number of feature maps in generator
num_feature_maps_D = 64 # Number of feature maps in discriminator
cifar10_data_root = './datasets' # Root directory for the CIFAR10 dataset
saved_models_dir = './saved_models/gan_cifar10_pytorch' # Model weight save directory
experiment_name = 'gan_cifar10_pytorch' # Name of the experiment
output_dir = './data/' + experiment_name + '_output' # Output directory for generated images
tensorboard_log_dir = './runs/' + experiment_name # TensorBoard log directory

# Set manual seed for reproducibility
randomSeed = random.randint(1, 10000)
print("Random Seed: ", randomSeed)
random.seed(randomSeed)
torch.manual_seed(randomSeed)

# Load the CIFAR10 dataset with resizing and normalization
# Images are resized to 64x64 and normalized to [-1, 1]
dataset = dset.CIFAR10(root="./datasets/cifar-10", download=False,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create DataLoader for batching and shuffling
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# Check if a GPU is available and set the device accordingly
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
else:
    device = "cpu"
print(f"Using {device} device")
sys.stdout.flush()

# Custom weights initialization for generator and discriminator
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, num_gpu):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # Input is Z (latent vector), going into a transposed convolution
            nn.ConvTranspose2d(z_dim, num_feature_maps_G * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_feature_maps_G * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_feature_maps_G * 8, num_feature_maps_G * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_G * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_feature_maps_G * 4, num_feature_maps_G * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_G * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_feature_maps_G * 2, num_feature_maps_G, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_G),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(num_feature_maps_G, num_in_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output state size: (nc) x 64 x 64
        )

    def forward(self, input):
        # Forward pass for generator
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.num_gpu))
        else:
            output = self.main(input)
            return output
        
# Create generator instance and apply weights initialization
generator_model = Generator(num_gpu).to(device)
generator_model.apply(weights_init)
print(generator_model)
sys.stdout.flush()

class Discriminator(nn.Module):
    def __init__(self, num_gpu):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # Input is (nc) x 64 x 64 (image)
            nn.Conv2d(num_in_channel, num_feature_maps_D, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(num_feature_maps_D, num_feature_maps_D * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_D * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(num_feature_maps_D * 2, num_feature_maps_D * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_D * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(num_feature_maps_D * 4, num_feature_maps_D * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_feature_maps_D * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(num_feature_maps_D * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # Forward pass for discriminator
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.num_gpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
# Create discriminator instance and apply weights initialization
discriminator_model = Discriminator(num_gpu).to(device)
discriminator_model.apply(weights_init)
print(discriminator_model)
sys.stdout.flush()

# Binary Cross Entropy loss for GAN
criterion = nn.BCELoss()

# Setup Adam optimizers for both generator and discriminator
optimizerD = optim.Adam(discriminator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Fixed noise for generating samples during training
fixed_noise = torch.randn(128, z_dim, 1, 1, device=device)
real_label = 1  # Label for real images
fake_label = 0  # Label for fake images

G_losses = []  # List to store generator loss
D_losses = []  # List to store discriminator loss

# TensorBoard writer for logging
writer = SummaryWriter(tensorboard_log_dir)

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with real images
        discriminator_model.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

        output = discriminator_model(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake images generated by the generator
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake = generator_model(noise)
        label.fill_(fake_label)
        output = discriminator_model(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator_model.zero_grad()
        label.fill_(real_label)  # Fake labels are real for generator cost
        output = discriminator_model(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Print training stats for each batch
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        sys.stdout.flush()

        # Log the losses to TensorBoard
        writer.add_scalars('Discriminator vs Generator Loss', {'Discriminator': errD.item(), 'Generator': errG.item()}, epoch)
        writer.flush()

        # Save real and fake images every 100 batches
        if i % 100 == 0:
            print('Saving the output for epoch %d, batch %d' % (epoch, i))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            vutils.save_image(real_cpu, output_dir + '/gan_cifar10_pytorch_real_samples.png',normalize=True)
            fake = generator_model(fixed_noise)
            vutils.save_image(fake.detach(), output_dir + '/gan_cifar10_pytorch_fake_samples_epoch_%03d.png' % (epoch),normalize=True)

    # Save model checkpoints after every epoch
    print('Saving models for epoch %d' % epoch)
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
    torch.save(generator_model.state_dict(), saved_models_dir + '/gan_cifar10_pytorch_G_epoch_%d.pth' % (epoch))
    torch.save(discriminator_model.state_dict(), saved_models_dir + '/gan_cifar10_pytorch_D_epoch_%d.pth' % (epoch))

# Close the TensorBoard writer
writer.close()
print("Training complete. Models and images saved.")
sys.stdout.flush()