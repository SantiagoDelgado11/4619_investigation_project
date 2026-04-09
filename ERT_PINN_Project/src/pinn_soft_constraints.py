"""PINN con restricciones suaves para el problema inverso 3D de ERT.

Ecuación gobernante (Poisson con fuentes/sumideros puntuales):
    -∇·(σ∇u) = I δ(r-r_A) - I δ(r-r_B)

Este módulo descarta la forma discreta algebraica F(m)=P A(m)^-1 q y trabaja
exclusivamente en el continuo, aproximando u(x,y,z) y σ(x,y,z) con redes
neuronales y operadores diferenciales calculados con autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass
class DomainConfig:
    """Configuración geométrica del dominio 3D."""

    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 0.0
    z_max: float = 10.0


@dataclass
class LossWeights:
    """Pesos de la función de pérdida compuesta (soft constraints)."""

    data: float = 100.0
    pde: float = 1.0
    dirichlet: float = 10.0
    neumann: float = 10.0


class MLP(nn.Module):
    """MLP base para aproximar campos continuos en 3D."""

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 128,
        hidden_layers: int = 6,
        out_features: int = 1,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_features, hidden_features), activation()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_features, hidden_features), activation()])
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Tensor) -> Tensor:
        return self.net(coords)


class ConductivityNet(nn.Module):
    """Red para σ(x,y,z) con positividad garantizada."""

    def __init__(self, **mlp_kwargs) -> None:
        super().__init__()
        self.backbone = MLP(**mlp_kwargs)
        self.softplus = nn.Softplus(beta=2.0)

    def forward(self, coords: Tensor) -> Tensor:
        raw = self.backbone(coords)
        return self.softplus(raw) + 1e-6


class PotentialNetSoft(nn.Module):
    """Red para u(x,y,z) sin imposición dura de frontera."""

    def __init__(self, **mlp_kwargs) -> None:
        super().__init__()
        self.backbone = MLP(**mlp_kwargs)

    def forward(self, coords: Tensor) -> Tensor:
        return self.backbone(coords)


def gradient(field: Tensor, coords: Tensor) -> Tensor:
    """∇field usando autograd."""

    return torch.autograd.grad(
        field,
        coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]


def gaussian_delta_3d(coords: Tensor, center: Tuple[float, float, float], epsilon: float) -> Tensor:
    """Aproximación gaussiana normalizada de δ(r-r0) en 3D."""

    center_t = coords.new_tensor(center).view(1, 3)
    r2 = torch.sum((coords - center_t) ** 2, dim=1, keepdim=True)
    norm = 1.0 / ((2.0 * math.pi * epsilon**2) ** 1.5)
    return norm * torch.exp(-r2 / (2.0 * epsilon**2))


def poisson_residual(
    u_model: nn.Module,
    sigma_model: nn.Module,
    coords: Tensor,
    pos_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
    current: float = 1.0,
    epsilon: float = 0.2,
) -> Tensor:
    """Residual de -∇·(σ∇u)=Iδ_A-Iδ_B escrito como ∇·(σ∇u)+Iδ_A-Iδ_B=0."""

    u = u_model(coords)
    sigma = sigma_model(coords)

    grad_u = gradient(u, coords)
    current_density = sigma * grad_u

    div_terms = []
    for axis in range(3):
        comp = current_density[:, axis : axis + 1]
        dcomp_daxis = torch.autograd.grad(
            comp,
            coords,
            grad_outputs=torch.ones_like(comp),
            create_graph=True,
            retain_graph=True,
        )[0][:, axis : axis + 1]
        div_terms.append(dcomp_daxis)

    divergence = div_terms[0] + div_terms[1] + div_terms[2]

    source = current * gaussian_delta_3d(coords, pos_a, epsilon) - current * gaussian_delta_3d(
        coords, pos_b, epsilon
    )
    return divergence + source


def geometric_boundary_weights(coords: Tensor, domain: DomainConfig, p: float = 2.0) -> Dict[str, Tensor]:
    """Pesos geométricos para reforzar fronteras donde se evalúan penalizaciones."""

    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

    wx = torch.minimum(torch.abs(x - domain.x_min), torch.abs(domain.x_max - x))
    wy = torch.minimum(torch.abs(y - domain.y_min), torch.abs(domain.y_max - y))
    wz_top = torch.abs(z - domain.z_min)
    wz_bottom = torch.abs(domain.z_max - z)

    eps = 1e-6
    dirichlet_w = 1.0 / ((wx + wy + wz_bottom + eps) ** p)
    neumann_w = 1.0 / ((wz_top + eps) ** p)
    return {"dirichlet": dirichlet_w.detach(), "neumann": neumann_w.detach()}


def compute_soft_losses(
    u_model: nn.Module,
    sigma_model: nn.Module,
    collocation_coords: Tensor,
    receiver_coords: Tensor,
    observed_potential: Tensor,
    boundary_surface_coords: Tensor,
    boundary_far_coords: Tensor,
    pos_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
    domain: DomainConfig,
    weights: LossWeights,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Calcula la loss compuesta: datos + PDE + Neumann + Dirichlet."""

    residual = poisson_residual(u_model, sigma_model, collocation_coords, pos_a, pos_b)
    loss_pde = torch.mean(residual.pow(2))

    pred_receivers = u_model(receiver_coords)
    loss_data = torch.mean((pred_receivers - observed_potential).pow(2))

    # Neumann: n·(σ∇u)=0 en superficie topográfica (aprox. z=0 con n=(0,0,1)).
    u_surface = u_model(boundary_surface_coords)
    grad_u_surface = gradient(u_surface, boundary_surface_coords)
    sigma_surface = sigma_model(boundary_surface_coords)
    normal_flux = sigma_surface * grad_u_surface[:, 2:3]

    surface_geo = geometric_boundary_weights(boundary_surface_coords, domain)
    loss_neumann = torch.mean(surface_geo["neumann"] * normal_flux.pow(2))

    # Dirichlet en frontera lejana: u -> 0.
    u_far = u_model(boundary_far_coords)
    far_geo = geometric_boundary_weights(boundary_far_coords, domain)
    loss_dirichlet = torch.mean(far_geo["dirichlet"] * u_far.pow(2))

    total = (
        weights.data * loss_data
        + weights.pde * loss_pde
        + weights.neumann * loss_neumann
        + weights.dirichlet * loss_dirichlet
    )

    details = {
        "total": total.detach(),
        "data": loss_data.detach(),
        "pde": loss_pde.detach(),
        "neumann": loss_neumann.detach(),
        "dirichlet": loss_dirichlet.detach(),
    }
    return total, details


def train_step_soft(
    u_model: nn.Module,
    sigma_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    collocation_coords: Tensor,
    receiver_coords: Tensor,
    observed_potential: Tensor,
    boundary_surface_coords: Tensor,
    boundary_far_coords: Tensor,
    pos_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
    domain: DomainConfig,
    weights: LossWeights,
) -> Dict[str, float]:
    """Paso de entrenamiento para PINN con restricciones suaves."""

    optimizer.zero_grad()

    collocation_coords = collocation_coords.requires_grad_(True)
    boundary_surface_coords = boundary_surface_coords.requires_grad_(True)
    boundary_far_coords = boundary_far_coords.requires_grad_(True)

    total, details = compute_soft_losses(
        u_model=u_model,
        sigma_model=sigma_model,
        collocation_coords=collocation_coords,
        receiver_coords=receiver_coords,
        observed_potential=observed_potential,
        boundary_surface_coords=boundary_surface_coords,
        boundary_far_coords=boundary_far_coords,
        pos_a=pos_a,
        pos_b=pos_b,
        domain=domain,
        weights=weights,
    )

    total.backward()
    optimizer.step()

    return {k: float(v.item()) for k, v in details.items()}


if __name__ == "__main__":
    torch.manual_seed(7)

    domain = DomainConfig()
    weights = LossWeights()

    u_net = PotentialNetSoft()
    sigma_net = ConductivityNet()
    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=1e-3)

    # Puntos sintéticos de ejemplo
    collocation = torch.rand(512, 3)
    collocation[:, 0] = collocation[:, 0] * (domain.x_max - domain.x_min) + domain.x_min
    collocation[:, 1] = collocation[:, 1] * (domain.y_max - domain.y_min) + domain.y_min
    collocation[:, 2] = collocation[:, 2] * (domain.z_max - domain.z_min) + domain.z_min

    surface = torch.rand(256, 3)
    surface[:, 0] = surface[:, 0] * (domain.x_max - domain.x_min) + domain.x_min
    surface[:, 1] = surface[:, 1] * (domain.y_max - domain.y_min) + domain.y_min
    surface[:, 2] = domain.z_min

    far = torch.rand(256, 3)
    far[:, 0] = torch.where(torch.rand(256) > 0.5, torch.full((256,), domain.x_max), torch.full((256,), domain.x_min))
    far[:, 1] = far[:, 1] * (domain.y_max - domain.y_min) + domain.y_min
    far[:, 2] = far[:, 2] * (domain.z_max - domain.z_min) + domain.z_min

    receivers = torch.rand(32, 3)
    receivers[:, 0] = receivers[:, 0] * (domain.x_max - domain.x_min) + domain.x_min
    receivers[:, 1] = receivers[:, 1] * (domain.y_max - domain.y_min) + domain.y_min
    receivers[:, 2] = domain.z_min

    observations = torch.zeros(32, 1)
    pos_a, pos_b = (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)

    metrics = train_step_soft(
        u_model=u_net,
        sigma_model=sigma_net,
        optimizer=optimizer,
        collocation_coords=collocation,
        receiver_coords=receivers,
        observed_potential=observations,
        boundary_surface_coords=surface,
        boundary_far_coords=far,
        pos_a=pos_a,
        pos_b=pos_b,
        domain=domain,
        weights=weights,
    )

    print(
        "Soft PINN | "
        f"total={metrics['total']:.4e}, data={metrics['data']:.4e}, "
        f"pde={metrics['pde']:.4e}, neu={metrics['neumann']:.4e}, dir={metrics['dirichlet']:.4e}"
    )
