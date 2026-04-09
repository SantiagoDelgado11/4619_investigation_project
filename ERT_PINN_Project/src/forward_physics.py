"""Forward physics operators for 3D ERT PINNs.

This module is intentionally limited to PDE-related definitions:
- differential operators built with autograd,
- current injection/extraction source terms,
- Poisson residual evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import math

import torch

Tensor = torch.Tensor
FieldNetwork = Callable[[Tensor], Tensor]
SourceFunction = Callable[[Tensor], Tensor]


def gradient_scalar(field: Tensor, coords: Tensor) -> Tensor:
    """Compute grad(field) with respect to spatial coordinates.

    Args:
        field: Scalar field with shape (N, 1), e.g. u(x, y, z).
        coords: Spatial tensor with shape (N, 3), with requires_grad=True.

    Returns:
        Tensor with shape (N, 3) containing [du/dx, du/dy, du/dz].
    """
    if field.ndim != 2 or field.shape[1] != 1:
        raise ValueError("field must have shape (N, 1)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    # autograd tracks how field depends on coords. create_graph=True keeps
    # second-order derivative paths available for divergence computations.
    grad = torch.autograd.grad(
        outputs=field,
        inputs=coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad


def divergence(vector_field: Tensor, coords: Tensor) -> Tensor:
    """Compute div(vector_field) with autograd.

    Args:
        vector_field: Tensor with shape (N, 3), e.g. current flux J.
        coords: Spatial tensor with shape (N, 3), with requires_grad=True.

    Returns:
        Tensor with shape (N, 1) equal to dJx/dx + dJy/dy + dJz/dz.
    """
    if vector_field.ndim != 2 or vector_field.shape[1] != 3:
        raise ValueError("vector_field must have shape (N, 3)")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    div = torch.zeros_like(vector_field[:, :1])
    for axis in range(3):
        component = vector_field[:, axis : axis + 1]
        component_grad = torch.autograd.grad(
            outputs=component,
            inputs=coords,
            grad_outputs=torch.ones_like(component),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        div = div + component_grad[:, axis : axis + 1]
    return div


def gaussian_dirac_delta(coords: Tensor, center: Tuple[float, float, float], epsilon: float) -> Tensor:
    """Smooth approximation of a 3D Dirac delta centered at `center`.

    The approximation converges to delta in distribution as epsilon -> 0.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")

    center_tensor = torch.tensor(center, dtype=coords.dtype, device=coords.device).view(1, 3)
    sq_radius = torch.sum((coords - center_tensor) ** 2, dim=1, keepdim=True)

    norm = 1.0 / ((2.0 * math.pi * epsilon * epsilon) ** 1.5)
    return norm * torch.exp(-sq_radius / (2.0 * epsilon * epsilon))


def dipole_source_term(
    coords: Tensor,
    electrode_a: Tuple[float, float, float],
    electrode_b: Tuple[float, float, float],
    current: float = 1.0,
    epsilon: float = 0.05,
) -> Tensor:
    """Compute I*delta(r-r_A) - I*delta(r-r_B)."""
    delta_a = gaussian_dirac_delta(coords, electrode_a, epsilon)
    delta_b = gaussian_dirac_delta(coords, electrode_b, epsilon)
    return current * (delta_a - delta_b)


@dataclass(frozen=True)
class DipoleCurrentSource:
    """Callable container for injection/extraction source parameters."""

    electrode_a: Tuple[float, float, float]
    electrode_b: Tuple[float, float, float]
    current: float = 1.0
    epsilon: float = 0.05

    def __call__(self, coords: Tensor) -> Tensor:
        return dipole_source_term(
            coords=coords,
            electrode_a=self.electrode_a,
            electrode_b=self.electrode_b,
            current=self.current,
            epsilon=self.epsilon,
        )


def poisson_residual(coords: Tensor, u: Tensor, sigma: Tensor, source: Tensor) -> Tensor:
    """Evaluate residual of -div(sigma * grad(u)) = source.

    Residual is defined as:
        R = -div(sigma * grad(u)) - source
    so an exact solution satisfies R = 0.
    """
    if sigma.ndim != 2 or sigma.shape[1] != 1:
        raise ValueError("sigma must have shape (N, 1)")

    grad_u = gradient_scalar(u, coords)
    flux = sigma * grad_u
    div_flux = divergence(flux, coords)
    return -div_flux - source


def evaluate_poisson_residual(
    u_network: FieldNetwork,
    sigma_network: FieldNetwork,
    coords: Tensor,
    source_fn: SourceFunction,
) -> Tensor:
    """Evaluate Poisson residual directly from network fields and source."""
    u = u_network(coords)
    sigma = sigma_network(coords)
    source = source_fn(coords)
    return poisson_residual(coords=coords, u=u, sigma=sigma, source=source)


def mean_squared_residual(residual: Tensor) -> Tensor:
    """L2 residual objective used in PINN losses."""
    return torch.mean(residual ** 2)


__all__ = [
    "DipoleCurrentSource",
    "dipole_source_term",
    "divergence",
    "evaluate_poisson_residual",
    "gaussian_dirac_delta",
    "gradient_scalar",
    "mean_squared_residual",
    "poisson_residual",
]
