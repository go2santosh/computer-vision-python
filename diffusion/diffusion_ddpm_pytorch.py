import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define the DDPM model
class DDPM(nn.Module):
    def __init__(self, image_size, channels):
        super(DDPM, self).__init__()
        self.image_size = image_size
        self.channels = channels
        # Define a simple UNet-like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the forward diffusion process
def forward_diffusion(x, t, noise_schedule):
    noise = torch.randn_like(x)
    alpha_t = noise_schedule[t]
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

# Training loop
def train_ddpm():
    # Hyperparameters
    epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    image_size = 32
    channels = 3

    # Noise schedule
    timesteps = 1000
    beta = torch.linspace(1e-4, 0.02, timesteps)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, and loss function
    model = DDPM(image_size, channels).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # Training
    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            t = torch.randint(0, timesteps, (images.size(0),), device=images.device)
            noisy_images, noise = forward_diffusion(images, t, alpha_hat)

            # Predict noise
            predicted_noise = model(noisy_images)
            loss = mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "ddpm_cifar10.pth")

if __name__ == "__main__":
    train_ddpm()