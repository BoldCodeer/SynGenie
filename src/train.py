from src.gan_model import Generator, Discriminator
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def train_gan(real_data, generator, discriminator, num_epochs=200, batch_size=32, noise_dim=16):
    """
    Train a GAN on the real dataset.

    Args:
        real_data (pd.DataFrame): The real dataset to model.
        generator (Generator): Generator model instance.
        discriminator (Discriminator): Discriminator model instance.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        noise_dim (int): Dimension of random noise input.

    Returns:
        Generator: Trained generator model.
    """
    data = torch.tensor(real_data.values, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for real_batch, in dataloader:
            batch_size = real_batch.size(0)

            # Labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            outputs = discriminator(real_batch)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, noise_dim)
            fake_data = generator(z)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    return generator
