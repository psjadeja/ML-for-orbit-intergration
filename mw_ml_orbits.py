"""
=============================================================================
Milky Way Orbit Integration & Substructure Detection via Machine Learning
=============================================================================
Combines:
  - Classical orbit integration using galpy (for ground truth / training data)
  - Neural ODE (torchdiffeq) for ML-based orbit integration
  - Physics-Informed Neural Network (PINN) for potential learning
  - HDBSCAN + GMM + Autoencoder for MW substructure detection

Dependencies:
    pip install galpy numpy scipy matplotlib torch torchdiffeq
    pip install scikit-learn hdbscan astropy umap-learn

Author: Generated for Milky Way stellar dynamics research
=============================================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# galpy – classical orbit integration (ground truth)
from galpy.orbit import Orbit
from galpy.potential import (MWPotential2014, NFWPotential,
                              MiyamotoNagaiPotential, PowerSphericalPotentialwCutoff)
from galpy.util import conversion
import astropy.units as u

# ML / clustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixturex as GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import hdbscan

# torchdiffeq – Neural ODE solver
try:
    from torchdiffeq import odeint
    NEURAL_ODE_AVAILABLE = True
except ImportError:
    print("torchdiffeq not installed. Neural ODE section will be skipped.")
    NEURAL_ODE_AVAILABLE = False

# reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ─────────────────────────────────────────────
# PART 1: CLASSICAL ORBIT INTEGRATION (galpy)
#         Used as ground-truth training data
# ─────────────────────────────────────────────

def generate_galpy_orbits(n_orbits: int = 500,
                           t_end_Gyr: float = 5.0,
                           n_steps: int = 1000) -> dict:
    """
    Integrate N stellar orbits in MWPotential2014 using galpy.

    Returns a dict with:
        positions  : (N, T, 3) – x,y,z in kpc
        velocities : (N, T, 3) – vx,vy,vz in km/s
        times      : (T,)      – time array in Gyr
        initial    : (N, 6)    – initial phase-space [R,vR,vT,z,vz,phi]
    """
    pot = MWPotential2014

    # Random initial conditions (disk-like distribution)
    R   = np.random.uniform(4,   14,  n_orbits)   # kpc
    vR  = np.random.normal(0,    20,  n_orbits)   # km/s
    vT  = np.random.normal(220,  30,  n_orbits)   # km/s (circular ~220)
    z   = np.random.normal(0,    0.5, n_orbits)   # kpc
    vz  = np.random.normal(0,    20,  n_orbits)   # km/s
    phi = np.random.uniform(0, 2*np.pi, n_orbits) # rad

    # Unit conversion factors (galpy natural units)
    ro, vo = 8.0, 220.0  # kpc, km/s

    ts = np.linspace(0, t_end_Gyr / conversion.time_in_Gyr(vo, ro), n_steps)

    all_x, all_y, all_z   = [], [], []
    all_vx, all_vy, all_vz = [], [], []

    print(f"Integrating {n_orbits} orbits in MWPotential2014...")
    for i in range(n_orbits):
        o = Orbit([R[i]/ro, vR[i]/vo, vT[i]/vo, z[i]/ro, vz[i]/vo, phi[i]])
        o.integrate(ts, pot)

        all_x.append(o.x(ts) * ro)
        all_y.append(o.y(ts) * ro)
        all_z.append(o.z(ts) * ro)
        all_vx.append(o.vx(ts) * vo)
        all_vy.append(o.vy(ts) * vo)
        all_vz.append(o.vz(ts) * vo)

    times_Gyr = ts * conversion.time_in_Gyr(vo, ro)

    return {
        'positions':  np.stack([all_x, all_y, all_z],  axis=2),   # (N,T,3)
        'velocities': np.stack([all_vx, all_vy, all_vz], axis=2),  # (N,T,3)
        'times':      times_Gyr,
        'initial':    np.column_stack([R, vR, vT, z, vz, phi]),
    }


def plot_sample_orbits(data: dict, n_show: int = 10):
    """Plot a handful of galpy orbits in x-y and R-z planes."""
    pos = data['positions']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = cm.plasma(np.linspace(0, 1, n_show))

    for i in range(n_show):
        axes[0].plot(pos[i, :, 0], pos[i, :, 1], color=colors[i], alpha=0.7, lw=0.8)
        R = np.sqrt(pos[i, :, 0]**2 + pos[i, :, 1]**2)
        axes[1].plot(R, pos[i, :, 2], color=colors[i], alpha=0.7, lw=0.8)

    for ax, xl, yl, t in zip(
        axes,
        ['x (kpc)', 'R (kpc)'],
        ['y (kpc)', 'z (kpc)'],
        ['Orbits: X-Y plane', 'Orbits: R-z plane']
    ):
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/galpy_orbits.png', dpi=150)
    plt.show()
    print("Saved: galpy_orbits.png")


# ─────────────────────────────────────────────
# PART 2: NEURAL ODE ORBIT INTEGRATOR
#         Learns the MW gravitational force field
#         and integrates orbits end-to-end
# ─────────────────────────────────────────────

class MWForceNet(nn.Module):
    """
    Neural network that learns the gravitational acceleration field
    f(x,y,z) → (ax, ay, az) of the Milky Way.

    Architecture: residual MLP with sinusoidal positional encoding.
    """
    def __init__(self, hidden: int = 256, n_layers: int = 6):
        super().__init__()
        # Fourier feature encoding for better spatial resolution
        self.freq = nn.Parameter(
            torch.randn(3, 64) * 2.0, requires_grad=False
        )
        in_dim = 128  # sin + cos of 64 frequencies

        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 3)]  # output: ax, ay, az

        self.net = nn.Sequential(*layers)

        # Residual skip
        self.skip = nn.Linear(in_dim, 3)

    def fourier_encode(self, x):
        proj = x @ self.freq          # (B, 64)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B,128)

    def forward(self, x):
        feat = self.fourier_encode(x)
        return self.net(feat) + self.skip(feat)


class NeuralODEOrbit(nn.Module):
    """
    Neural ODE wrapper: state = [x, y, z, vx, vy, vz]
    d/dt [pos] = vel
    d/dt [vel] = force_net(pos)
    """
    def __init__(self, force_net: MWForceNet):
        super().__init__()
        self.force_net = force_net

    def forward(self, t, state):
        # state: (B, 6)
        pos = state[:, :3]   # x, y, z
        vel = state[:, 3:]   # vx, vy, vz
        acc = self.force_net(pos)
        return torch.cat([vel, acc], dim=-1)


def prepare_training_data(galpy_data: dict, n_train: int = 2000,
                           device: str = 'cpu'):
    """
    Build (position, acceleration) pairs from galpy orbits via finite difference.
    """
    pos = galpy_data['positions']   # (N,T,3) kpc
    vel = galpy_data['velocities']  # (N,T,3) km/s
    times = galpy_data['times']     # (T,) Gyr

    # Convert to consistent units: kpc, km/s, Gyr
    dt = np.diff(times)  # (T-1,)

    # Acceleration via finite differences on velocity (km/s / Gyr → km/s²)
    acc = np.diff(vel, axis=1) / dt[None, :, None]  # (N,T-1,3)
    mid_pos = 0.5 * (pos[:, :-1] + pos[:, 1:])       # (N,T-1,3)

    # Flatten and subsample
    N, T1, _ = acc.shape
    flat_pos = mid_pos.reshape(-1, 3)
    flat_acc = acc.reshape(-1, 3)

    idx = np.random.choice(len(flat_pos), min(n_train, len(flat_pos)), replace=False)
    X = torch.tensor(flat_pos[idx], dtype=torch.float32).to(device)
    Y = torch.tensor(flat_acc[idx], dtype=torch.float32).to(device)
    return X, Y


def train_force_network(X: torch.Tensor, Y: torch.Tensor,
                         epochs: int = 300, lr: float = 1e-3,
                         device: str = 'cpu') -> MWForceNet:
    """Train the MWForceNet to predict gravitational acceleration."""
    net = MWForceNet().to(device)
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(X, Y), batch_size=512, shuffle=True)
    losses = []

    print("Training Force Network...")
    for ep in range(1, epochs + 1):
        ep_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()
        avg = ep_loss / len(loader)
        losses.append(avg)
        if ep % 50 == 0:
            print(f"  Epoch {ep:4d}/{epochs} | Loss: {avg:.4e}")

    # Plot training curve
    plt.figure(figsize=(8, 4))
    plt.semilogy(losses)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Force Network Training Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/force_net_loss.png', dpi=150)
    print("Saved: force_net_loss.png")

    return net


def integrate_neural_orbit(force_net: MWForceNet,
                            init_conditions: np.ndarray,
                            t_span_Gyr: tuple = (0, 5),
                            n_steps: int = 500,
                            device: str = 'cpu') -> np.ndarray:
    """
    Integrate one orbit using the Neural ODE.

    init_conditions: array [x0, y0, z0, vx0, vy0, vz0]
    Returns: (n_steps, 6) trajectory
    """
    if not NEURAL_ODE_AVAILABLE:
        raise RuntimeError("torchdiffeq not installed.")

    node = NeuralODEOrbit(force_net).to(device)
    node.eval()

    ts = torch.linspace(*t_span_Gyr, n_steps, dtype=torch.float32).to(device)
    y0 = torch.tensor(init_conditions, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        traj = odeint(node, y0, ts, method='rk4')  # (T, 1, 6)

    return traj.squeeze(1).cpu().numpy()  # (T, 6)


def compare_orbits(galpy_data: dict, force_net: MWForceNet,
                   orbit_idx: int = 0, device: str = 'cpu'):
    """Compare galpy (truth) vs Neural ODE orbit."""
    pos = galpy_data['positions'][orbit_idx]   # (T,3)
    vel = galpy_data['velocities'][orbit_idx]  # (T,3)
    t   = galpy_data['times']

    # Initial condition from galpy data
    ic = np.concatenate([pos[0], vel[0]])  # [x,y,z,vx,vy,vz]

    if NEURAL_ODE_AVAILABLE:
        neural_traj = integrate_neural_orbit(
            force_net, ic,
            t_span_Gyr=(t[0], t[-1]),
            n_steps=len(t),
            device=device
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, xi, yi, label in zip(
            [axes[0], axes[1]],
            [0, None], [1, None],
            ['X-Y Plane', 'R-z Plane']
        ):
            if label == 'X-Y Plane':
                ax.plot(pos[:, 0], pos[:, 1], 'b-', lw=1.5, label='galpy (truth)')
                ax.plot(neural_traj[:, 0], neural_traj[:, 1], 'r--', lw=1.5, label='Neural ODE')
                ax.set_xlabel('x (kpc)'); ax.set_ylabel('y (kpc)')
            else:
                R_g = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
                R_n = np.sqrt(neural_traj[:, 0]**2 + neural_traj[:, 1]**2)
                ax.plot(R_g, pos[:, 2], 'b-', lw=1.5, label='galpy (truth)')
                ax.plot(R_n, neural_traj[:, 2], 'r--', lw=1.5, label='Neural ODE')
                ax.set_xlabel('R (kpc)'); ax.set_ylabel('z (kpc)')
            ax.set_title(label)
            ax.legend(); ax.grid(alpha=0.3)

        plt.suptitle('galpy vs Neural ODE Orbit Integration', fontsize=13)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/orbit_comparison.png', dpi=150)
        print("Saved: orbit_comparison.png")
    else:
        print("Skipping orbit comparison (torchdiffeq not available).")


# ─────────────────────────────────────────────
# PART 3: PHYSICS-INFORMED NEURAL NETWORK (PINN)
#         Learn gravitational potential Φ(x,y,z)
#         such that ∇²Φ = 4πGρ (Poisson eq.)
# ─────────────────────────────────────────────

class PotentialPINN(nn.Module):
    """
    Neural network representing the gravitational potential Φ(x,y,z).
    Physics constraint: ∇²Φ = 4πGρ enforced via automatic differentiation.
    """
    def __init__(self, hidden: int = 128, n_layers: int = 5):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # scalar potential per point

    def gradient(self, x):
        """Compute ∇Φ = [∂Φ/∂x, ∂Φ/∂y, ∂Φ/∂z]."""
        x = x.requires_grad_(True)
        phi = self.forward(x)
        grad = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        return grad  # gravitational acceleration = -∇Φ

    def laplacian(self, x):
        """Compute ∇²Φ for Poisson equation constraint."""
        x = x.requires_grad_(True)
        phi = self.forward(x)
        grad = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        lap = sum(
            torch.autograd.grad(grad[:, i].sum(), x, create_graph=True)[0][:, i]
            for i in range(3)
        )
        return lap.unsqueeze(1)


def train_pinn(X_pos: torch.Tensor, Y_acc: torch.Tensor,
               epochs: int = 500, lr: float = 5e-4,
               lambda_poisson: float = 0.1,
               device: str = 'cpu') -> PotentialPINN:
    """
    Train PINN with two loss terms:
      1. Data loss:    |∇Φ - observed_acc|²
      2. Physics loss: |∇²Φ - 4πGρ|²  (Poisson eq., ρ from MW model)
    """
    G = 4.302e-3  # pc M_sun^-1 (km/s)^2  → scale appropriately
    model = PotentialPINN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []

    # Simple density model: exponential disk + NFW halo (analytic)
    def density_model(x):
        R = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
        z = x[:, 2]
        # Disk: Miyamoto-Nagai profile
        rho_disk = 0.1 * torch.exp(-R / 3.0 - torch.abs(z) / 0.3)
        # Halo: NFW profile
        r = torch.sqrt(R**2 + z**2)
        rs = 20.0
        rho_halo = 0.01 / ((r / rs) * (1 + r / rs)**2 + 1e-6)
        return rho_disk + rho_halo

    loader = DataLoader(TensorDataset(X_pos, Y_acc), batch_size=256, shuffle=True)
    print("Training PINN...")

    for ep in range(1, epochs + 1):
        ep_loss = 0
        for xb, acc_b in loader:
            opt.zero_grad()

            # Data loss: predicted force = -∇Φ
            grad_phi = model.gradient(xb)
            data_loss = nn.MSELoss()(-grad_phi, acc_b)

            # Physics loss: Poisson equation
            lap = model.laplacian(xb)
            rho = density_model(xb).unsqueeze(1)
            poisson_loss = nn.MSELoss()(lap, 4 * np.pi * G * rho)

            loss = data_loss + lambda_poisson * poisson_loss
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        avg = ep_loss / len(loader)
        losses.append(avg)
        if ep % 100 == 0:
            print(f"  PINN Epoch {ep:4d}/{epochs} | Loss: {avg:.4e}")

    plt.figure(figsize=(8, 4))
    plt.semilogy(losses)
    plt.xlabel('Epoch'); plt.ylabel('Total Loss')
    plt.title('PINN Training Loss (Data + Poisson)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pinn_loss.png', dpi=150)
    print("Saved: pinn_loss.png")
    return model


def visualize_pinn_potential(model: PotentialPINN, device: str = 'cpu'):
    """Visualize the learned potential in the x-y plane (z=0)."""
    grid = np.linspace(-15, 15, 100)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    X = torch.tensor(pts, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        phi = model(X).cpu().numpy().reshape(100, 100)

    plt.figure(figsize=(8, 7))
    plt.contourf(xx, yy, phi, levels=50, cmap='RdBu_r')
    plt.colorbar(label='Φ (learned potential)')
    plt.xlabel('x (kpc)'); plt.ylabel('y (kpc)')
    plt.title('PINN Learned MW Gravitational Potential (z=0)')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pinn_potential.png', dpi=150)
    print("Saved: pinn_potential.png")


# ─────────────────────────────────────────────
# PART 4: MW SUBSTRUCTURE DETECTION
#         Streams, clusters, moving groups
#         from 6D phase space using ML
# ─────────────────────────────────────────────

def compute_integrals_of_motion(positions: np.ndarray,
                                 velocities: np.ndarray,
                                 pot=None) -> np.ndarray:
    """
    Compute conserved quantities for substructure detection:
      - Energy E = KE + PE
      - Angular momentum Lz = x*vy - y*vx
      - |L| = sqrt(Lx²+Ly²+Lz²)

    Returns feature matrix (N, 5): [E, Lz, |L|, Lx, Ly]
    """
    if pot is None:
        pot = MWPotential2014

    x, y, z   = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]

    # Kinetic energy (per unit mass, km²/s²)
    KE = 0.5 * (vx**2 + vy**2 + vz**2)

    # Angular momentum components (kpc·km/s)
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    L  = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Potential energy from galpy (evaluated at each position)
    ro, vo = 8.0, 220.0
    from galpy.potential import evaluatePotentials
    R = np.sqrt(x**2 + y**2) / ro
    _z = z / ro
    PE_nat = evaluatePotentials(pot, R, _z)
    PE = PE_nat * vo**2  # km²/s²

    E = KE + PE  # total energy

    return np.column_stack([E, Lz, L, Lx, Ly])


def generate_mock_catalog(galpy_data: dict,
                           n_stars: int = 5000,
                           n_streams: int = 3) -> dict:
    """
    Generate a mock stellar catalog with:
      - Field stars drawn from galpy orbits (smooth background)
      - Injected stellar streams (tight groups in phase space)
    """
    N_field = n_stars - n_streams * 100
    idx = np.random.choice(galpy_data['positions'].shape[0], N_field, replace=True)
    t_idx = np.random.randint(0, galpy_data['positions'].shape[1], N_field)

    field_pos = galpy_data['positions'][idx, t_idx]    # (N_field, 3)
    field_vel = galpy_data['velocities'][idx, t_idx]   # (N_field, 3)
    labels_field = np.zeros(N_field, dtype=int)

    # Inject streams: tight clumps in 6D phase space
    stream_pos, stream_vel, stream_labels = [], [], []
    stream_centers_pos = np.array([
        [8, 0, 2], [-5, 7, 1], [3, -10, -1]
    ])
    stream_centers_vel = np.array([
        [-30, 200, 10], [50, -180, 20], [10, 210, -15]
    ])
    for s in range(n_streams):
        n_s = 100
        sp = stream_centers_pos[s] + np.random.randn(n_s, 3) * np.array([0.5, 0.5, 0.2])
        sv = stream_centers_vel[s] + np.random.randn(n_s, 3) * np.array([5, 5, 3])
        stream_pos.append(sp)
        stream_vel.append(sv)
        stream_labels.append(np.full(n_s, s + 1))

    all_pos = np.vstack([field_pos] + stream_pos)
    all_vel = np.vstack([field_vel] + stream_vel)
    all_labels = np.concatenate([labels_field] + stream_labels)

    return {'positions': all_pos, 'velocities': all_vel, 'true_labels': all_labels}


class PhaseSpaceAutoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction of 6D phase space.
    Useful for substructure detection in a compressed latent space.
    """
    def __init__(self, input_dim: int = 6, latent_dim: int = 3,
                 hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X: np.ndarray, epochs: int = 200,
                       lr: float = 1e-3,
                       device: str = 'cpu') -> PhaseSpaceAutoencoder:
    """Train autoencoder on phase-space data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Xt = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model = PhaseSpaceAutoencoder(input_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xt), batch_size=256, shuffle=True)

    print("Training Phase-Space Autoencoder...")
    losses = []
    for ep in range(1, epochs + 1):
        ep_loss = 0
        for (xb,) in loader:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(loader))
        if ep % 50 == 0:
            print(f"  AE Epoch {ep:3d}/{epochs} | Loss: {losses[-1]:.4e}")

    model.eval()
    with torch.no_grad():
        _, latent = model(Xt)
    latent_np = latent.cpu().numpy()
    return model, latent_np, scaler


def detect_substructures_hdbscan(features: np.ndarray,
                                   labels_true: np.ndarray = None,
                                   min_cluster_size: int = 15) -> np.ndarray:
    """
    Run HDBSCAN on phase-space / IoM features to find substructures.
    Returns predicted labels (-1 = noise/field).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    pred_labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    n_noise    = np.sum(pred_labels == -1)
    print(f"HDBSCAN: {n_clusters} clusters found | {n_noise} noise points")

    if labels_true is not None and n_clusters > 0:
        # Recovery rate: fraction of true stream stars recovered
        for s in np.unique(labels_true[labels_true > 0]):
            mask = labels_true == s
            pred_here = pred_labels[mask]
            best = np.bincount(pred_here[pred_here >= 0] + 1).argmax() - 1 \
                   if np.any(pred_here >= 0) else -1
            if best >= 0:
                purity = np.mean(pred_labels[pred_labels == best] == -1)
                recall = np.mean(pred_here == best)
                print(f"  Stream {s}: best cluster {best}, recall={recall:.2f}")

    return pred_labels


def detect_substructures_gmm(features: np.ndarray,
                               max_components: int = 8) -> np.ndarray:
    """
    Gaussian Mixture Model with BIC model selection.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    bics = []
    models = []
    ks = range(2, max_components + 1)
    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=42, max_iter=300)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))
        models.append(gmm)

    best_idx = np.argmin(bics)
    best_gmm = models[best_idx]
    print(f"GMM best k={ks[best_idx]} (BIC={bics[best_idx]:.1f})")

    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), bics, 'o-')
    plt.xlabel('Number of components'); plt.ylabel('BIC')
    plt.title('GMM Model Selection (BIC)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/gmm_bic.png', dpi=150)

    return best_gmm.predict(X_scaled)


def plot_substructures(features_iom: np.ndarray,
                        pred_labels: np.ndarray,
                        true_labels: np.ndarray,
                        method: str = 'HDBSCAN'):
    """Plot detected substructures in E–Lz space."""
    E, Lz = features_iom[:, 0], features_iom[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Ground truth
    for label in np.unique(true_labels):
        mask = true_labels == label
        color = 'gray' if label == 0 else None
        alpha = 0.3 if label == 0 else 0.9
        lbl   = 'Field' if label == 0 else f'Stream {label}'
        axes[0].scatter(Lz[mask], E[mask], s=5, alpha=alpha,
                        label=lbl, color=color)
    axes[0].set_xlabel('Lz (kpc·km/s)'); axes[0].set_ylabel('E (km²/s²)')
    axes[0].set_title('Ground Truth Labels')
    axes[0].legend(markerscale=3)
    axes[0].grid(alpha=0.3)

    # Predictions
    unique_pred = np.unique(pred_labels)
    cmap = cm.tab10
    for i, label in enumerate(unique_pred):
        mask = pred_labels == label
        color = 'gray' if label == -1 else cmap(i / max(len(unique_pred), 1))
        alpha = 0.2 if label == -1 else 0.8
        lbl   = 'Noise' if label == -1 else f'Cluster {label}'
        axes[1].scatter(Lz[mask], E[mask], s=5, alpha=alpha,
                        label=lbl, c=[color] * mask.sum())
    axes[1].set_xlabel('Lz (kpc·km/s)'); axes[1].set_ylabel('E (km²/s²)')
    axes[1].set_title(f'{method} Detected Substructures')
    axes[1].legend(markerscale=3)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/mnt/user-data/outputs/substructures_{method.lower()}.png', dpi=150)
    print(f"Saved: substructures_{method.lower()}.png")


def plot_latent_space(latent: np.ndarray, labels: np.ndarray):
    """Visualize autoencoder latent space colored by substructure."""
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    cmap = cm.tab10
    for label in np.unique(labels):
        mask = labels == label
        color = 'lightgray' if label == -1 else cmap(label / 10)
        alpha = 0.2 if label == -1 else 0.8
        name  = 'Field/Noise' if label <= 0 else f'Stream {label}'
        ax.scatter(latent[mask, 0], latent[mask, 1], latent[mask, 2],
                   s=5, alpha=alpha, c=[color] * mask.sum(), label=name)
    ax.set_xlabel('Latent 1'); ax.set_ylabel('Latent 2'); ax.set_zlabel('Latent 3')
    ax.set_title('Autoencoder Latent Space (colored by HDBSCAN labels)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/latent_space.png', dpi=150)
    print("Saved: latent_space.png")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)

    # ── Step 1: Generate galpy orbits (ground truth) ──────────────
    print("\n[1] Generating galpy orbits...")
    galpy_data = generate_galpy_orbits(n_orbits=300, t_end_Gyr=5.0, n_steps=500)
    plot_sample_orbits(galpy_data, n_show=15)

    # ── Step 2: Train Force Network ────────────────────────────────
    print("\n[2] Training Neural Force Field...")
    X_pos, Y_acc = prepare_training_data(galpy_data, n_train=50000, device=device)
    force_net = train_force_network(X_pos, Y_acc, epochs=300, device=device)

    # ── Step 3: Neural ODE orbit integration ──────────────────────
    print("\n[3] Neural ODE Orbit Integration...")
    compare_orbits(galpy_data, force_net, orbit_idx=0, device=device)

    # ── Step 4: PINN for potential learning ───────────────────────
    print("\n[4] Training PINN (Physics-Informed Potential)...")
    pinn = train_pinn(X_pos, Y_acc, epochs=300, device=device)
    visualize_pinn_potential(pinn, device=device)

    # ── Step 5: Build mock stellar catalog ────────────────────────
    print("\n[5] Building mock stellar catalog with injected streams...")
    catalog = generate_mock_catalog(galpy_data, n_stars=3000, n_streams=3)

    # ── Step 6: Compute Integrals of Motion ───────────────────────
    print("\n[6] Computing Integrals of Motion (E, Lz, |L|)...")
    iom_features = compute_integrals_of_motion(
        catalog['positions'], catalog['velocities']
    )
    print(f"  IoM feature matrix shape: {iom_features.shape}")

    # 6D phase space features
    phase6d = np.hstack([catalog['positions'], catalog['velocities']])

    # ── Step 7: HDBSCAN Substructure Detection ────────────────────
    print("\n[7] HDBSCAN Substructure Detection (IoM space)...")
    hdb_labels = detect_substructures_hdbscan(
        iom_features, catalog['true_labels'], min_cluster_size=20
    )
    plot_substructures(iom_features, hdb_labels, catalog['true_labels'], 'HDBSCAN')

    # ── Step 8: GMM Substructure Detection ───────────────────────
    print("\n[8] GMM Substructure Detection...")
    gmm_labels = detect_substructures_gmm(iom_features, max_components=8)
    plot_substructures(iom_features, gmm_labels, catalog['true_labels'], 'GMM')

    # ── Step 9: Autoencoder + HDBSCAN in latent space ────────────
    print("\n[9] Autoencoder Latent Space + HDBSCAN...")
    ae_model, latent, scaler = train_autoencoder(phase6d, epochs=200, device=device)
    latent_labels = detect_substructures_hdbscan(latent, catalog['true_labels'])
    plot_latent_space(latent, latent_labels)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. Check /mnt/user-data/outputs/ for all plots.")
    print("=" * 60)


if __name__ == '__main__':
    main()
