"""Inverse ERT PINN training with hard boundary constraints.

Boundary conditions are encoded by construction in the potential network, so the
loss contains only data misfit + PDE residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from forward_physics import (
    DipoleCurrentSource,
    evaluate_poisson_residual,
    mean_squared_residual,
)

Tensor = torch.Tensor


class MLP(nn.Module):
    """Coordinate MLP used as unconstrained core network."""

    def __init__(self, in_features: int = 3, width: int = 128, depth: int = 5) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_features, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        return self.net(coords)


class ConductivityNet(nn.Module):
    """Positive conductivity model sigma(x, y, z)."""

    def __init__(self, width: int = 128, depth: int = 5, sigma_floor: float = 1e-4) -> None:
        super().__init__()
        self.backbone = MLP(in_features=3, width=width, depth=depth)
        self.softplus = nn.Softplus()
        self.sigma_floor = sigma_floor

    def forward(self, coords: Tensor) -> Tensor:
        return self.softplus(self.backbone(coords)) + self.sigma_floor


@dataclass(frozen=True)
class BoxDomain:
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0
    z_min: float = 0.0
    z_max: float = 5.0


@dataclass(frozen=True)
class HardLossWeights:
    data: float = 100.0
    pde: float = 1.0


class HardPotentialNet(nn.Module):
    """Potential model with exact BC enforcement via distance-function ansatz.

    We build:
        u(x, y, z) = phi_D(x, y, z) * g(x, y, z^2)

    where:
    - g is unconstrained.
    - z^2 makes g even in z, so dg/dz = 0 at z=0.
    - phi_D is built from distance-like factors and is exactly zero on
      Dirichlet boundaries (deep + lateral limits in this box example).

    The z factor in phi_D is chosen as (1 - (z/z_max)^2), which is equivalent to
    ((z_max - z) * (z_max + z)) / z_max^2. It vanishes at z=z_max and has zero
    derivative at z=0, so Neumann on the top surface is preserved exactly.
    """

    def __init__(self, domain: BoxDomain, width: int = 128, depth: int = 5) -> None:
        super().__init__()
        self.domain = domain
        self.core = MLP(in_features=3, width=width, depth=depth)

    def _distance_envelope(self, coords: Tensor) -> Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]

        lx = self.domain.x_max - self.domain.x_min
        ly = self.domain.y_max - self.domain.y_min

        d_x_left = x - self.domain.x_min
        d_x_right = self.domain.x_max - x
        d_y_front = y - self.domain.y_min
        d_y_back = self.domain.y_max - y

        # Lateral distance factors: zero exactly on x/y boundaries.
        phi_x = (d_x_left * d_x_right) / ((0.5 * lx) ** 2)
        phi_y = (d_y_front * d_y_back) / ((0.5 * ly) ** 2)

        # Deep-boundary factor from distance functions in z.
        phi_z = 1.0 - (z / self.domain.z_max) ** 2

        return phi_x * phi_y * phi_z

    def forward(self, coords: Tensor) -> Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]

        # Using z^2 forces even dependence wrt z and yields du/dz=0 at z=0.
        z_even = z ** 2
        core_input = torch.cat([x, y, z_even], dim=1)

        raw_u = self.core(core_input)
        envelope = self._distance_envelope(coords)
        return envelope * raw_u


def _as_leaf(coords: Tensor) -> Tensor:
    return coords.detach().clone().requires_grad_(True)


def sample_interior_points(n_points: int, domain: BoxDomain, device: torch.device) -> Tensor:
    rand = torch.rand(n_points, 3, device=device)
    x = domain.x_min + (domain.x_max - domain.x_min) * rand[:, 0:1]
    y = domain.y_min + (domain.y_max - domain.y_min) * rand[:, 1:2]
    z = domain.z_min + (domain.z_max - domain.z_min) * rand[:, 2:3]
    return torch.cat([x, y, z], dim=1)


def sample_surface_points(n_points: int, domain: BoxDomain, device: torch.device) -> Tuple[Tensor, Tensor]:
    rand_xy = torch.rand(n_points, 2, device=device)
    x = domain.x_min + (domain.x_max - domain.x_min) * rand_xy[:, 0:1]
    y = domain.y_min + (domain.y_max - domain.y_min) * rand_xy[:, 1:2]
    z = torch.full((n_points, 1), domain.z_min, device=device)
    coords = torch.cat([x, y, z], dim=1)

    normals = torch.zeros(n_points, 3, device=device)
    normals[:, 2] = 1.0
    return coords, normals


def compute_hard_loss(
    u_net: nn.Module,
    sigma_net: nn.Module,
    source_model: Callable[[Tensor], Tensor],
    collocation_coords: Tensor,
    electrode_coords: Tensor,
    observed_voltages: Tensor,
    weights: HardLossWeights,
) -> Tuple[Tensor, Dict[str, float]]:
    collocation_coords = _as_leaf(collocation_coords)

    pde_residual = evaluate_poisson_residual(
        u_network=u_net,
        sigma_network=sigma_net,
        coords=collocation_coords,
        source_fn=source_model,
    )
    loss_pde = mean_squared_residual(pde_residual)

    u_electrodes = u_net(electrode_coords)
    loss_data = torch.mean((u_electrodes - observed_voltages) ** 2)

    total_loss = weights.data * loss_data + weights.pde * loss_pde

    metrics = {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
    }
    return total_loss, metrics


def train_inverse_hard(
    u_net: nn.Module,
    sigma_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_model: Callable[[Tensor], Tensor],
    electrode_coords: Tensor,
    observed_voltages: Tensor,
    collocation_sampler: Callable[[int], Tensor],
    weights: HardLossWeights,
    epochs: int = 2000,
    n_collocation: int = 4096,
    print_every: int = 200,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        collocation_coords = collocation_sampler(n_collocation)

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = compute_hard_loss(
            u_net=u_net,
            sigma_net=sigma_net,
            source_model=source_model,
            collocation_coords=collocation_coords,
            electrode_coords=electrode_coords,
            observed_voltages=observed_voltages,
            weights=weights,
        )
        total_loss.backward()
        optimizer.step()

        history.append(metrics)
        if epoch == 1 or epoch % print_every == 0:
            print(
                f"[Hard] Epoch {epoch:05d} | "
                f"total={metrics['loss_total']:.6e} | "
                f"data={metrics['loss_data']:.6e} | "
                f"pde={metrics['loss_pde']:.6e}"
            )

    return history


if __name__ == "__main__":
    torch.manual_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain = BoxDomain(x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0, z_min=0.0, z_max=10.0)

    u_model = HardPotentialNet(domain=domain, width=96, depth=5).to(device)
    sigma_model = ConductivityNet(width=96, depth=5).to(device)

    source = DipoleCurrentSource(
        electrode_a=(-2.0, 0.0, 0.0),
        electrode_b=(2.0, 0.0, 0.0),
        current=1.0,
        epsilon=0.2,
    )

    electrode_coords, _ = sample_surface_points(n_points=32, domain=domain, device=device)
    observed_voltages = 0.25 * torch.sin(0.4 * electrode_coords[:, 0:1]) * torch.exp(-0.1 * electrode_coords[:, 1:2] ** 2)

    optimizer = torch.optim.Adam(
        list(u_model.parameters()) + list(sigma_model.parameters()),
        lr=1e-3,
    )

    train_inverse_hard(
        u_net=u_model,
        sigma_net=sigma_model,
        optimizer=optimizer,
        source_model=source,
        electrode_coords=electrode_coords,
        observed_voltages=observed_voltages,
        collocation_sampler=lambda n: sample_interior_points(n, domain, device),
        weights=HardLossWeights(data=100.0, pde=1.0),
        epochs=500,
        n_collocation=2048,
        print_every=100,
    )
