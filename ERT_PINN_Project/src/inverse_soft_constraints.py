"""Inverse ERT PINN training with soft boundary constraints.

The model learns continuous fields u(x, y, z) and sigma(x, y, z) directly from
coordinates, without meshes or matrix-based inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from forward_physics import (
    DipoleCurrentSource,
    evaluate_poisson_residual,
    gradient_scalar,
    mean_squared_residual,
)

Tensor = torch.Tensor


class MLP(nn.Module):
    """Simple fully-connected network over spatial coordinates (x, y, z)."""

    def __init__(self, in_features: int = 3, width: int = 128, depth: int = 5) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_features, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        return self.net(coords)


class PotentialNet(nn.Module):
    """Approximates electric potential u(x, y, z)."""

    def __init__(self, width: int = 128, depth: int = 5) -> None:
        super().__init__()
        self.backbone = MLP(in_features=3, width=width, depth=depth)

    def forward(self, coords: Tensor) -> Tensor:
        return self.backbone(coords)


class ConductivityNet(nn.Module):
    """Approximates conductivity sigma(x, y, z), constrained to be positive."""

    def __init__(self, width: int = 128, depth: int = 5, sigma_floor: float = 1e-4) -> None:
        super().__init__()
        self.backbone = MLP(in_features=3, width=width, depth=depth)
        self.softplus = nn.Softplus()
        self.sigma_floor = sigma_floor

    def forward(self, coords: Tensor) -> Tensor:
        return self.softplus(self.backbone(coords)) + self.sigma_floor


@dataclass(frozen=True)
class BoxDomain:
    """Continuous box domain used only to sample collocation/boundary points."""

    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0
    z_min: float = 0.0
    z_max: float = 5.0


@dataclass(frozen=True)
class SoftLossWeights:
    """Weights for soft-constraint objective."""

    data: float = 100.0
    pde: float = 1.0
    dirichlet: float = 10.0
    neumann: float = 10.0


def _as_leaf(coords: Tensor) -> Tensor:
    """Create a leaf tensor so autograd can differentiate wrt coordinates."""
    return coords.detach().clone().requires_grad_(True)


def sample_interior_points(n_points: int, domain: BoxDomain, device: torch.device) -> Tensor:
    rand = torch.rand(n_points, 3, device=device)
    x = domain.x_min + (domain.x_max - domain.x_min) * rand[:, 0:1]
    y = domain.y_min + (domain.y_max - domain.y_min) * rand[:, 1:2]
    z = domain.z_min + (domain.z_max - domain.z_min) * rand[:, 2:3]
    return torch.cat([x, y, z], dim=1)


def sample_surface_points(n_points: int, domain: BoxDomain, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Sample topographic surface points (flat topography z=z_min in this example)."""
    rand_xy = torch.rand(n_points, 2, device=device)
    x = domain.x_min + (domain.x_max - domain.x_min) * rand_xy[:, 0:1]
    y = domain.y_min + (domain.y_max - domain.y_min) * rand_xy[:, 1:2]
    z = torch.full((n_points, 1), domain.z_min, device=device)
    coords = torch.cat([x, y, z], dim=1)

    normals = torch.zeros(n_points, 3, device=device)
    normals[:, 2] = 1.0
    return coords, normals


def sample_deep_boundary_points(n_points: int, domain: BoxDomain, device: torch.device) -> Tensor:
    rand_xy = torch.rand(n_points, 2, device=device)
    x = domain.x_min + (domain.x_max - domain.x_min) * rand_xy[:, 0:1]
    y = domain.y_min + (domain.y_max - domain.y_min) * rand_xy[:, 1:2]
    z = torch.full((n_points, 1), domain.z_max, device=device)
    return torch.cat([x, y, z], dim=1)


def compute_soft_loss(
    u_net: nn.Module,
    sigma_net: nn.Module,
    source_model: Callable[[Tensor], Tensor],
    collocation_coords: Tensor,
    electrode_coords: Tensor,
    observed_voltages: Tensor,
    surface_coords: Tensor,
    surface_normals: Tensor,
    deep_boundary_coords: Tensor,
    weights: SoftLossWeights,
) -> Tuple[Tensor, Dict[str, float]]:
    collocation_coords = _as_leaf(collocation_coords)
    surface_coords = _as_leaf(surface_coords)

    pde_residual = evaluate_poisson_residual(
        u_network=u_net,
        sigma_network=sigma_net,
        coords=collocation_coords,
        source_fn=source_model,
    )
    loss_pde = mean_squared_residual(pde_residual)

    u_electrodes = u_net(electrode_coords)
    loss_data = torch.mean((u_electrodes - observed_voltages) ** 2)

    u_deep = u_net(deep_boundary_coords)
    loss_dirichlet = torch.mean(u_deep ** 2)

    u_surface = u_net(surface_coords)
    sigma_surface = sigma_net(surface_coords)
    grad_u_surface = gradient_scalar(u_surface, surface_coords)

    # No-current boundary: n . (sigma * grad(u)) = 0.
    flux_surface = sigma_surface * grad_u_surface
    normal_flux = torch.sum(flux_surface * surface_normals, dim=1, keepdim=True)
    loss_neumann = torch.mean(normal_flux ** 2)

    total_loss = (
        weights.data * loss_data
        + weights.pde * loss_pde
        + weights.dirichlet * loss_dirichlet
        + weights.neumann * loss_neumann
    )

    metrics = {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu()),
        "loss_pde": float(loss_pde.detach().cpu()),
        "loss_dirichlet": float(loss_dirichlet.detach().cpu()),
        "loss_neumann": float(loss_neumann.detach().cpu()),
    }
    return total_loss, metrics


def train_inverse_soft(
    u_net: nn.Module,
    sigma_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_model: Callable[[Tensor], Tensor],
    electrode_coords: Tensor,
    observed_voltages: Tensor,
    collocation_sampler: Callable[[int], Tensor],
    surface_sampler: Callable[[int], Tuple[Tensor, Tensor]],
    deep_boundary_sampler: Callable[[int], Tensor],
    weights: SoftLossWeights,
    epochs: int = 2000,
    n_collocation: int = 4096,
    n_surface: int = 512,
    n_deep: int = 512,
    print_every: int = 200,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        collocation_coords = collocation_sampler(n_collocation)
        surface_coords, surface_normals = surface_sampler(n_surface)
        deep_coords = deep_boundary_sampler(n_deep)

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = compute_soft_loss(
            u_net=u_net,
            sigma_net=sigma_net,
            source_model=source_model,
            collocation_coords=collocation_coords,
            electrode_coords=electrode_coords,
            observed_voltages=observed_voltages,
            surface_coords=surface_coords,
            surface_normals=surface_normals,
            deep_boundary_coords=deep_coords,
            weights=weights,
        )
        total_loss.backward()
        optimizer.step()

        history.append(metrics)
        if epoch == 1 or epoch % print_every == 0:
            print(
                f"[Soft] Epoch {epoch:05d} | "
                f"total={metrics['loss_total']:.6e} | "
                f"data={metrics['loss_data']:.6e} | "
                f"pde={metrics['loss_pde']:.6e} | "
                f"dir={metrics['loss_dirichlet']:.6e} | "
                f"neu={metrics['loss_neumann']:.6e}"
            )

    return history


if __name__ == "__main__":
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain = BoxDomain(x_min=-8.0, x_max=8.0, y_min=-8.0, y_max=8.0, z_min=0.0, z_max=10.0)

    u_model = PotentialNet(width=96, depth=5).to(device)
    sigma_model = ConductivityNet(width=96, depth=5).to(device)

    source = DipoleCurrentSource(
        electrode_a=(-2.0, 0.0, 0.0),
        electrode_b=(2.0, 0.0, 0.0),
        current=1.0,
        epsilon=0.2,
    )

    electrode_coords, _ = sample_surface_points(n_points=32, domain=domain, device=device)

    # Synthetic observed voltages at electrode coordinates (replace with field data).
    observed_voltages = 0.25 * torch.sin(0.4 * electrode_coords[:, 0:1]) * torch.exp(-0.1 * electrode_coords[:, 1:2] ** 2)

    optimizer = torch.optim.Adam(
        list(u_model.parameters()) + list(sigma_model.parameters()),
        lr=1e-3,
    )

    train_inverse_soft(
        u_net=u_model,
        sigma_net=sigma_model,
        optimizer=optimizer,
        source_model=source,
        electrode_coords=electrode_coords,
        observed_voltages=observed_voltages,
        collocation_sampler=lambda n: sample_interior_points(n, domain, device),
        surface_sampler=lambda n: sample_surface_points(n, domain, device),
        deep_boundary_sampler=lambda n: sample_deep_boundary_points(n, domain, device),
        weights=SoftLossWeights(data=100.0, pde=1.0, dirichlet=20.0, neumann=20.0),
        epochs=500,
        n_collocation=2048,
        n_surface=256,
        n_deep=256,
        print_every=100,
    )
