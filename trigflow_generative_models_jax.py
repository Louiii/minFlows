"""Minified example for TrigFlow noise schedule and scaling in the EDM framwork.

TrigFlow: [Lu & Song 2025](https://openreview.net/forum?id=LyJi5ugyJx)
EDM: [Karras et al. 2022](https://arxiv.org/abs/2206.00364)
"""
import argparse
import math
from enum import Enum, auto
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from jax import Array
from jynx import layers as nn
from jynx.fit import TrainState, make_train_step
from jynx.pytree import PyTree, static
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_moons
from tqdm import tqdm

# Folder to write plots to.
PLOTDIR = Path("outputs")
PLOTDIR.mkdir(exist_ok=True, parents=True)


class NetworkImage(Enum):
    """The 'image' or output space of the neural network."""
    DATA = auto()          # EDM
    NOISE = auto()         # Score-matching
    VELOCITY = auto()      # Flow-Matching


def euler_step(net, x_cur, t, t_next, S_noise, key):
    # Note: in the forward marginal t | t_next, we have this, but we are in reverse.
    delta_var = net.conditional_variance(t, t_cond=t_next)
    x_hat = x_cur + (delta_var**0.5) * S_noise * jr.normal(key, x_cur.shape)
    d_cur = net.direction(x_hat, t)
    return x_hat + (t_next - t) * d_cur


def heun_step(net, x_cur, t, t_next, S_noise, key):
    # Note: in the forward marginal t | t_next, we have this, but we are in reverse.
    delta_var = net.conditional_variance(t, t_cond=t_next)
    x_hat = x_cur + (delta_var**0.5) * S_noise * jr.normal(key, x_cur.shape)
    d_cur = net.direction(x_hat, t)
    x_next = x_hat + (t_next - t) * d_cur
    d_prime = net.direction(x_next, t_next)
    return x_hat + (t_next - t) * 0.5 * (d_cur + d_prime)


def heun_sampler(key, net, prior_sample, num_steps: int = 100, S_noise: float = 1.0):
    """Second order sampler inspired by Karras et al.
    
    Note: I replaced the noise scaling with the difference in variance of the noise
    between the two time steps. This is because the their scaling is incompatible with
    the TrigFlow equations."""
    t_steps = net.time_steps(num_steps, device=prior_sample.device)
    x = prior_sample * net.sigma_data # Initial sample x_{pi/2} is pure noise with std=sigma_data

    def step(x, aux):
        t, t_next, subkey = aux
        x = heun_step(net, x, t, t_next, S_noise, subkey)
        return x, x

    key, subkey = jr.split(key)
    x, path = jax.lax.scan(
        step, init=x, xs=(t_steps[:-2], t_steps[1:-1], jr.split(key, num_steps-1))
    )
    x = euler_step(net, x, t_steps[-2], t_steps[-1], S_noise, subkey)
    return x, jnp.concatenate([path, x[None]], axis=0)


def fourier_embedding(x, freqs):
    """Embed real values into a Fourier basis.
    Inspired by `https://github.com/NVlabs/edm/blob/main/training/networks.py#L212`"""
    x = x[:, None] * (2 * np.pi * freqs)[None, :]
    return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)


class DiffusionNet(PyTree):
    """Simple MLP with a fourier embedding for time conditioning variable."""
    net: nn.Module
    fourier_frequencies: Array = static()

    def __call__(self, x, time_cond):
        t_freqs = fourier_embedding(time_cond, self.fourier_frequencies)
        return self.net(jnp.concatenate([x, t_freqs], axis=1))

    @classmethod
    def init(
        cls, in_dims=2, fourier_num_channels=128, hidden=512, out_dims=2, scale=16, *, key
    ):
        k1, k2 = jr.split(key)
        return cls(
            net=nn.mlp(
                [in_dims + fourier_num_channels, hidden, hidden, hidden, out_dims],
                activation=jax.nn.silu,
                key=k1,
            ),
            fourier_frequencies=jr.normal(k2, fourier_num_channels // 2) * scale,
        )


class TrigFlow(PyTree):
    """TrigFlow PF-ODE equations and network scaling functions."""
    model: nn.Module
    sigma_data: float = static()
    output_space: NetworkImage = static()
    P_std_val: float = static()
    P_mean_val: float = static()
    
    def c_skip(self, t):
        return jnp.cos(t)

    def c_out(self, t):
        return -self.sigma_data * jnp.sin(t)

    def c_in(self, t):
        return jnp.ones_like(t) / self.sigma_data

    def c_noise(self, t):
        return t

    def __call__(self, x, t):
        """Forward pass of the network; with scaling functions as in EDM, Karras et al."""
        if t.ndim == 0:
            t = jnp.broadcast_to(t[None], x.shape[:1])

        # Ensure t has a channel dimension for broadcasting
        t_b = t[..., None] if t.ndim > 0 else t[None, None]

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
        return (jnp.sin(t_cond - t) * self.sigma_data / jnp.cos(t_cond))**2

    def forward_marginal(self, x0, t, z):
        """Transform sample z~N(0,data_var) to xt~p(xt|x0,t)=N(xt|cos(t)x0,sin^2(t)*data_var)."""
        return jnp.cos(t) * x0 + jnp.sin(t) * z

    def instantaneous_forward_marginal_velocity(self, x0, t, z):
        """Compute velocity dxt/dt for xt = cos(t)x0 + sin(t)z; the instantenous flow."""
        return -jnp.sin(t) * x0 + jnp.cos(t) * z

    def loss(self, x0, *, key):
        """Take data, apply forward marginal, predict data/noise/flow, compute error."""
        k1, k2 = jr.split(key)

        # Sample t uniformly from [epsilon, pi/2] to avoid t=0 issues
        tau = jr.normal(k1, x0.shape[0]) * self.P_std_val + self.P_mean_val
        t = jnp.arctan(jnp.exp(tau) / self.sigma_data)

        # Sample noise z ~ N(0, sigma_d^2 * I)
        eps = jr.normal(k2, x0.shape)
        z = eps * self.sigma_data

        # Create the noisy sample x_t = cos(t)x_0 + sin(t)z
        t_b = t[..., None]
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
        return jnp.linspace(top, 0, num_steps + 1, device=device)

    def direction(self, x, t):
        """The network has learnt a mapping between noise and data; this gives the
        direction at a point (xt, t)."""
        match self.output_space:
            case NetworkImage.DATA:
                # Predict the clean data x_0
                denoised = self(x, t)

                # Use the TrigFlow derivative for dx/dt ---
                direction = (jnp.cos(t) * x - denoised) / jnp.sin(t)
            case NetworkImage.NOISE:
                # Get the network's prediction of the noise which we will use as a score
                predicted_noise = self(x, t)

                # dx/dt for the score-based PF-ODE.
                direction = -x * jnp.tan(t) + self.sigma_data * predicted_noise / jnp.cos(t)
            case NetworkImage.VELOCITY:
                # Get the network's prediction of the velocity
                direction = self(x, t)
        return direction

    @classmethod
    def init(
        cls,
        output_space: NetworkImage,
        sigma_data: float = 1.0,
        P_mean_val: float = -1.2,
        P_std_val: float = 1.2,
        *,
        key,
    ):
        return cls(
            model=DiffusionNet.init(key=key),
            sigma_data=sigma_data,
            output_space=output_space,
            P_mean_val=P_mean_val,
            P_std_val=P_std_val,
        )


def dataloader(data, batch_size):
    batch = []
    data_copy = np.array(data)
    np.random.shuffle(data_copy)
    for item in data_copy:
        batch.append(item)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = []


def train_and_sample(network_output_space: NetworkImage, plot_net: bool) -> None:
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

    # --- Initialise Model ---
    print("Initialising model...")
    # Pass sigma_data to the network for pre-conditioning
    net = TrigFlow.init(
        output_space=network_output_space,
        sigma_data=jnp.std(moons),
        P_mean_val=P_mean_val,
        P_std_val=P_std_val,
        key=jr.key(0)
    )
    optimiser = optax.adam(learning_rate=learning_rate)
    train_step = make_train_step(lambda state, batch, key: state.loss(batch, key=key), optimiser)
    state = TrainState(net, jax.tree.map(jnp.zeros_like, net), optimiser.init(net))

    # --- Training ---
    print("Starting training...")
    losses = []
    progress_bar = tqdm(range(num_epochs), total=num_epochs)
    key = jr.key(0)
    for _ in progress_bar:
        for batch_data in dataloader(moons, batch_size=batch_size):
            key, sk = jr.split(key)
            state, loss, _ = train_step(state, batch_data, sk)
            losses.append(loss.item())
        progress_bar.set_description(f"Loss: {np.mean(losses[-10000:]):.4f}")

    # --- Inference ---
    print("Generating new samples with the trained model...")
    prior_samples = jr.normal(jr.key(0), (n_samples, 2))
    model_samples, path = heun_sampler(
        jr.key(0), state.params, prior_samples, num_steps=sampler_steps, S_noise=S_noise
    )

    # --- Visualise ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(*moons.T, c='blue', s=10, alpha=0.5, label="Original Data")
    ax.scatter(*np.array(model_samples).T, c='red', s=10, alpha=0.5, label="Generated Samples")
    ax.set_title(f"Predicting {network_output_space.name}.")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(PLOTDIR / network_output_space.name)
    plt.close(fig)

    if plot_net:
        path = np.array(path)
        net = state.params
        ts = net.time_steps(path.shape[0], device=model_samples.device)[:-1]
        num = 32
        xs = np.linspace(np.percentile(path[..., 0], 3), np.percentile(path[..., 0], 97), num)
        ys = np.linspace(np.percentile(path[..., 1], 3), np.percentile(path[..., 1], 97), num)
        x = np.stack(np.meshgrid(xs, ys), axis=-1)
        field = np.array(jax.lax.map(lambda t: jax.vmap(net.direction, (0, None))(x, t), ts))

        print("Animating...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        cmap = sns.color_palette("rocket", as_cmap=True)

        def update(frame):
            """
            This function is called for each frame of the animation.
            It clears the previous frame and draws the new one.
            """
            ax.clear()
            if frame >= ts.shape[0]:
                frame = -1

            # Get data for the current time step
            u = field[frame, :, :, 0]
            v = field[frame, :, :, 1]
            current_path = path[frame, :, :]
            current_time = ts[frame]

            # Calculate the magnitude of the vectors
            magnitude = np.sqrt(u**2 + v**2)

            # Plotting the vector field
            ax.quiver(x[:, :, 0], x[:, :, 1], u, v, magnitude, cmap=cmap, alpha=0.8, scale=50)

            # Plotting the particle paths
            ax.scatter(current_path[:, 0], current_path[:, 1], s=5, color="#960DFE", alpha=0.8, edgecolors='w', linewidth=0.5)

            ax.set_facecolor('#1C1C1E')
            ax.set_title(f'Vector Field and Particle Paths\nTime: {current_time:.2f}s', fontsize=16, color='white')
            ax.set_xlabel('X-coordinate', fontsize=12, color='white')
            ax.set_ylabel('Y-coordinate', fontsize=12, color='white')
            ax.set_xlim(np.min(x[:,:,0]), np.max(x[:,:,0]))
            ax.set_ylim(np.min(x[:,:,1]), np.max(x[:,:,1]))
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555')

        ani = FuncAnimation(fig, update, frames=np.arange(len(ts) + 10), interval=50, blit=False)
        ani.save(
            PLOTDIR / f'{network_output_space.name}.mp4',
            writer="imagemagick",
            fps=15,
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train simple generative models")
    parser.add_argument("--network_output", type=str, help="The output type of the network.")
    parser.add_argument("--plot_net", action="store_true", help="Shift the directory up by one.")
    args = parser.parse_args()
    train_and_sample(NetworkImage.__members__[args.network_output], plot_net=args.plot_net)


if __name__ == "__main__":
    main()