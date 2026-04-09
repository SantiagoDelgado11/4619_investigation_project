"""PINN con restricciones duras para el problema inverso 3D de ERT.

Ecuación gobernante:
    -∇·(σ∇u)=Iδ(r-r_A)-Iδ(r-r_B)

Las condiciones de frontera se imponen por construcción (hard constraints):
- Dirichlet: u -> 0 al alejarse (envolvente de decaimiento).
- Neumann en topografía (z=0): n·(σ∇u)=0 exacta mediante simetría par en z.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass
class LossWeights:
    data: float = 100.0
    pde: float = 1.0


class FourierFeatures(nn.Module):
    """Embedding sen/cos para aumentar expresividad espacial."""

    def __init__(self, in_features: int = 3, n_frequencies: int = 16, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, n_frequencies) * scale)

    def forward(self, coords: Tensor) -> Tensor:
        proj = 2.0 * math.pi * coords @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 128, hidden_layers: int = 6) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_features, hidden_features), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_features, hidden_features), nn.Tanh()])
        layers.append(nn.Linear(hidden_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConductivityNet(nn.Module):
    """σ(x,y,z)>0 usando Softplus."""

    def __init__(self) -> None:
        super().__init__()
        self.base = MLP(in_features=3)
        self.softplus = nn.Softplus(beta=2.0)

    def forward(self, coords: Tensor) -> Tensor:
        return self.softplus(self.base(coords)) + 1e-6


class PotentialNetHard(nn.Module):
    """u(x,y,z) con fronteras satisfechas exactamente por diseño."""

    def __init__(self, n_frequencies: int = 24, fourier_scale: float = 0.15, decay_scale: float = 20.0) -> None:
        super().__init__()
        self.decay_scale = decay_scale
        self.embed = FourierFeatures(in_features=3, n_frequencies=n_frequencies, scale=fourier_scale)
        # coords_sym = (x, y, z^2) fuerza simetría par respecto de z.
        self.base = MLP(in_features=3 + 2 * n_frequencies)

    def forward(self, coords: Tensor) -> Tensor:
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]

        coords_sym = torch.cat([x, y, z.pow(2)], dim=1)
        fourier = self.embed(coords_sym)
        latent = torch.cat([coords_sym, fourier], dim=1)
        raw = self.base(latent)

        # Envolvente de distancia: garantiza u -> 0 cuando ||r|| -> inf (Dirichlet exacta).
        r2 = x.pow(2) + y.pow(2) + z.pow(2)
        envelope = torch.exp(-r2 / (self.decay_scale**2))

        # Hard constraints: u = envelope * raw(x,y,z^2)
        # Debido a z^2, ∂raw/∂z|_{z=0}=0. Además ∂envelope/∂z|_{z=0}=0.
        # => ∂u/∂z|_{z=0}=0 y por tanto n·(σ∇u)=0 en superficie plana z=0.
        return envelope * raw


def gradient(field: Tensor, coords: Tensor) -> Tensor:
    return torch.autograd.grad(
        field,
        coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]


def gaussian_delta_3d(coords: Tensor, center: Tuple[float, float, float], epsilon: float) -> Tensor:
    c = coords.new_tensor(center).view(1, 3)
    r2 = torch.sum((coords - c) ** 2, dim=1, keepdim=True)
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
    """Residual de la física sin términos de frontera en loss."""

    u = u_model(coords)
    sigma = sigma_model(coords)

    grad_u = gradient(u, coords)
    flux = sigma * grad_u

    div = 0.0
    for axis in range(3):
        comp = flux[:, axis : axis + 1]
        dcomp = torch.autograd.grad(
            comp,
            coords,
            grad_outputs=torch.ones_like(comp),
            create_graph=True,
            retain_graph=True,
        )[0][:, axis : axis + 1]
        div = div + dcomp

    source = current * gaussian_delta_3d(coords, pos_a, epsilon) - current * gaussian_delta_3d(
        coords, pos_b, epsilon
    )
    return div + source


def compute_hard_losses(
    u_model: nn.Module,
    sigma_model: nn.Module,
    collocation_coords: Tensor,
    receiver_coords: Tensor,
    observed_potential: Tensor,
    pos_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
    weights: LossWeights,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Loss simplificada: datos + PDE (sin penalizaciones de frontera)."""

    residual = poisson_residual(u_model, sigma_model, collocation_coords, pos_a, pos_b)
    loss_pde = torch.mean(residual.pow(2))

    pred_receivers = u_model(receiver_coords)
    loss_data = torch.mean((pred_receivers - observed_potential).pow(2))

    total = weights.data * loss_data + weights.pde * loss_pde
    return total, {"total": total.detach(), "data": loss_data.detach(), "pde": loss_pde.detach()}


def train_step_hard(
    u_model: nn.Module,
    sigma_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    collocation_coords: Tensor,
    receiver_coords: Tensor,
    observed_potential: Tensor,
    pos_a: Tuple[float, float, float],
    pos_b: Tuple[float, float, float],
    weights: LossWeights,
) -> Dict[str, float]:
    optimizer.zero_grad()

    collocation_coords = collocation_coords.requires_grad_(True)

    total, losses = compute_hard_losses(
        u_model=u_model,
        sigma_model=sigma_model,
        collocation_coords=collocation_coords,
        receiver_coords=receiver_coords,
        observed_potential=observed_potential,
        pos_a=pos_a,
        pos_b=pos_b,
        weights=weights,
    )

    total.backward()
    optimizer.step()
    return {k: float(v.item()) for k, v in losses.items()}


if __name__ == "__main__":
    torch.manual_seed(7)

    u_net = PotentialNetHard()
    sigma_net = ConductivityNet()
    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=1e-3)

    collocation = torch.rand(512, 3)
    collocation[:, 0] = collocation[:, 0] * 20.0 - 10.0
    collocation[:, 1] = collocation[:, 1] * 20.0 - 10.0
    collocation[:, 2] = collocation[:, 2] * 10.0

    receivers = torch.rand(32, 3)
    receivers[:, 0] = receivers[:, 0] * 20.0 - 10.0
    receivers[:, 1] = receivers[:, 1] * 20.0 - 10.0
    receivers[:, 2] = 0.0

    observed = torch.zeros(32, 1)
    pos_a, pos_b = (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)

    metrics = train_step_hard(
        u_model=u_net,
        sigma_model=sigma_net,
        optimizer=optimizer,
        collocation_coords=collocation,
        receiver_coords=receivers,
        observed_potential=observed,
        pos_a=pos_a,
        pos_b=pos_b,
        weights=LossWeights(),
    )
    print(f"Hard PINN | total={metrics['total']:.4e}, data={metrics['data']:.4e}, pde={metrics['pde']:.4e}")

    # Validación rápida de Neumann exacta en z=0.
    test = torch.tensor([[1.0, -2.0, 0.0]], requires_grad=True)
    uz = gradient(u_net(test), test)[:, 2].item()
    print(f"Chequeo d u/dz en z=0: {uz:.4e} (esperado ~0 por construcción)")
