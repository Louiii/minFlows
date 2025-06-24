"""Minified example for TrigFlow noise schedule and scaling in the EDM framwork.

TrigFlow: [Lu & Song 2025](https://openreview.net/forum?id=LyJi5ugyJx)
EDM: [Karras et al. 2022](https://arxiv.org/abs/2206.00364)
"""
import math
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Folder to write plots to.
PLOTDIR = Path("outputs")
PLOTDIR.mkdir(exist_ok=True, parents=True)


class NetworkImage(Enum):
    """The 'image' or output space of the neural network."""
    DATA = auto()          # EDM
    NOISE = auto()         # Score-matching
    VELOCITY = auto()      # Flow-Matching


def heun_sampler(net, prior_sample, num_steps: int = 100, S_noise: float = 1.0):
    """Second order sampler inspired by Karras et al.
    
    Note: I replaced the noise scaling with the difference in variance of the noise
    between the two time steps. This is because the their scaling is incompatible with
    the TrigFlow equations."""
    t_steps = net.time_steps(num_steps, device=prior_sample.device)

    # Main sampling loop
    x_cur = prior_sample * net.sigma_data # Initial sample x_{pi/2} is pure noise with std=sigma_data
    for i in tqdm(range(len(t_steps) - 1), total=num_steps, desc="Sampling"):
        t, t_next = t_steps[i:i+2]

        # Note: in the forward marginal t | t_next, we have this, but we are in reverse.
        delta_var = net.conditional_variance(t, t_cond=t_next)
        x_hat = x_cur + delta_var.sqrt() * S_noise * torch.randn_like(x_cur)

        d_cur = net.direction(x_hat, t)

        # Euler step
        x_next = x_hat + (t_next - t) * d_cur

        # Apply 2nd order correction (Heun's method)
        if i < num_steps - 1:
            d_prime = net.direction(x_next, t_next)
            x_next = x_hat + (t_next - t) * 0.5 * (d_cur + d_prime)
        x_cur = x_next
    return x_cur


class FourierEmbedding(torch.nn.Module):
    """Embed real values into a Fourier basis.
    Taken from `https://github.com/NVlabs/edm/blob/main/training/networks.py#L212`"""
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class DiffusionNet(nn.Module):
    """Simple MLP with a fourier embedding for time conditioning variable."""
    def __init__(self, in_dims=2, fourier_num_channels=128, hidden=512, out_dims=2):
        super().__init__()
        self.fourier_features = FourierEmbedding(num_channels=fourier_num_channels)

        # The network now takes the input dimensions + fourier feature dimensions
        self.net = nn.Sequential(
            nn.Linear(in_dims + fourier_num_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dims)
        )

    def forward(self, x, time_cond):
        return self.net(torch.cat([x, self.fourier_features(time_cond)], dim=1))


class TrigFlow(nn.Module):
    """TrigFlow PF-ODE equations and network scaling functions."""
    def __init__(
        self,
        output_space: NetworkImage,
        sigma_data: float = 1.0,
        P_mean_val: float = -1.2,
        P_std_val: float = 1.2
    ):
        super().__init__()
        self.model = DiffusionNet() 
        self.sigma_data = sigma_data
        self.output_space = output_space
        self.P_mean_val = P_mean_val
        self.P_std_val = P_std_val

    def c_skip(self, t):
        return torch.cos(t)

    def c_out(self, t):
        return -self.sigma_data * torch.sin(t)

    def c_in(self, t):
        return torch.ones_like(t) / self.sigma_data

    def c_noise(self, t):
        return t

    def forward(self, x, t):
        """Forward pass of the network; with scaling functions as in EDM, Karras et al."""
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        # Ensure t has a channel dimension for broadcasting
        t_b = t.view(-1, 1) if t.ndim > 0 else t.view(1, 1)

        c_skip_val = self.c_skip(t_b)
        c_out_val = self.c_out(t_b)
        c_in_val = self.c_in(t_b)
        c_noise_val = self.c_noise(t)
        
        x_in = c_in_val * x
        F_x = self.model(x_in, c_noise_val)

        match self.output_space:
            case NetworkImage.DATA:
                denoised = c_skip_val * x + c_out_val * F_x
                output = denoised
            case NetworkImage.NOISE:
                output = F_x
            case NetworkImage.VELOCITY:
                output = F_x

        return output

    def conditional_variance(self, t, t_cond):
        """Var[x_t|x_t_cond]; can be derived from p(xt|x0,t) and p(x_tcond|x0,tcond) by
        subbing in for x0, to give p(xt|x_tcond) then compute the variance."""
        return (torch.sin(t_cond - t) * self.sigma_data / torch.cos(t_cond))**2

    def forward_marginal(self, x0, t, z):
        """Transform sample z~N(0,data_var) to xt~p(xt|x0,t)=N(xt|cos(t)x0,sin^2(t)*data_var)."""
        return torch.cos(t) * x0 + torch.sin(t) * z

    def instantaneous_forward_marginal_velocity(self, x0, t, z):
        """Compute velocity dxt/dt for xt = cos(t)x0 + sin(t)z; the instantenous flow."""
        return -torch.sin(t) * x0 + torch.cos(t) * z

    def loss(self, x0):
        """Take data, apply forward marginal, predict data/noise/flow, compute error."""
        # Sample t uniformly from [epsilon, pi/2] to avoid t=0 issues
        tau = torch.randn(x0.shape[0], device=x0.device) * self.P_std_val + self.P_mean_val
        t = torch.arctan(tau.exp() / self.sigma_data)

        # Sample noise z ~ N(0, sigma_d^2 * I)
        eps = torch.randn_like(x0)
        z = eps * self.sigma_data

        # Create the noisy sample x_t = cos(t)x_0 + sin(t)z
        t_b = t.view(-1, 1)
        xt = self.forward_marginal(x0, t_b, z)
        prediction = self(xt, t)

        match self.output_space:
            case NetworkImage.DATA:
                target = x0
            case NetworkImage.NOISE:
                target = eps
            case NetworkImage.VELOCITY:
                target = self.instantaneous_forward_marginal_velocity(x0, t_b, z)

        return ((prediction - target)**2).mean()

    def time_steps(self, num_steps: int, safe_tan_shift: float = 0.1, *, device):
        """Give the time steps from pi/2 -> 0, but the noise prediction has an asymtote
        at t = pi/2 so shift it down slightly."""
        # Time step discretization from pi/2 to 0
        top = math.pi / 2
        if self.output_space == NetworkImage.NOISE:
            top -= safe_tan_shift  # This is base tan blows up at pi/2 
        return torch.linspace(top, 0, num_steps + 1, device=device)

    def direction(self, x, t):
        """The network has learnt a mapping between noise and data; this gives the
        direction at a point (xt, t)."""
        match self.output_space:
            case NetworkImage.DATA:
                # Predict the clean data x_0
                denoised = self(x.double(), t.double()).to(torch.float64)

                # Use the TrigFlow derivative for dx/dt ---
                direction = (torch.cos(t) * x - denoised) / torch.sin(t)
            case NetworkImage.NOISE:
                # Get the network's prediction of the noise which we will use as a score
                predicted_noise = self(x.double(), t.double())

                # dx/dt for the score-based PF-ODE.
                direction = -x * torch.tan(t) + self.sigma_data * predicted_noise / torch.cos(t)
            case NetworkImage.VELOCITY:
                # Get the network's prediction of the velocity
                direction = self(x.double(), t.double())
        return direction

def train_and_sample(network_output_space):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 3000
    batch_size = 512
    learning_rate = 3e-4
    P_mean_val = -1.2
    P_std_val = 1.2
    n_samples = 3000
    data_noise = 0.07
    sampler_steps = 50
    S_noise = 0.5

    # --- Create Dataset ---
    print("Generating and normalizing 'many moons' dataset...")
    moons, _ = make_moons(n_samples=n_samples, noise=data_noise, random_state=42)
    data = torch.tensor(moons, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    # --- Initialise Model ---
    print("Initialising model...")
    # Pass sigma_data to the network for pre-conditioning
    net = TrigFlow(
        output_space=network_output_space,
        sigma_data=data.std(),
        P_mean_val=P_mean_val,
        P_std_val=P_std_val,
    ).to(device)
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    # --- Training ---
    print("Starting training...")
    net.train()
    losses = []
    progress_bar = tqdm(range(num_epochs), total=num_epochs)
    for _ in progress_bar:
        for batch_data in loader:
            clean_data = batch_data[0].to(device)
            optimiser.zero_grad()
            loss = net.loss(clean_data)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        progress_bar.set_description(f"Loss: {np.mean(losses[-10000:]):.4f}")

    # --- Inference ---
    print("Generating new samples with the trained model...")
    net.eval()
    net.double()
    with torch.no_grad():
        prior_samples = torch.randn(n_samples, 2, device=device)
        model_samples = heun_sampler(
            net, prior_samples, num_steps=sampler_steps, S_noise=S_noise
        ).cpu().numpy()

    # --- Visualise ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(*data.T, c='blue', s=10, alpha=0.5, label="Original Data")
    ax.scatter(*model_samples.T, c='red', s=10, alpha=0.5, label="Generated Samples")
    ax.set_title(f"Predicting {network_output_space.name}.")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(PLOTDIR / network_output_space.name)
    plt.close(fig)


def main():
    for network_output_space in NetworkImage:
        train_and_sample(network_output_space)


if __name__ == "__main__":
    main()