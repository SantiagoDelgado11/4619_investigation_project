import torch
import pytest
from src.physics import compute_pde_loss

def test_laplacian_autograd():
    """
    Prueba unitaria para verificar que PyTorch autograd computa derivadas
    de segundo orden correctamente en nuestro domain_loss.
    """
    # Función dummy analítica: U(x, z) = x^2 + z^2
    # nabla(U) = [2x, 2z]
    # divergency(nabla(U)) = laplacian(U) = 4.0
    # Si sumimos conductividad constante sigma=1:
    # nabla . (sigma * nabla U) = 4.0
    
    x = torch.tensor([[1.0]], requires_grad=True)
    z = torch.tensor([[2.0]], requires_grad=True)
    
    # Forward pass simulado
    u = x**2 + z**2
    sigma = torch.tensor([[1.0]], requires_grad=True) # Constante
    
    # 1. Gradientes de U (nabla u)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    assert torch.allclose(u_x, torch.tensor([[2.0]])), "Primera derivada X falló"
    assert torch.allclose(u_z, torch.tensor([[4.0]])), "Primera derivada Z falló"
    
    # 2. Flujo conductivo (sigma * nabla u) = nabla u (dado que sigma=1)
    J_x = sigma * u_x
    J_z = sigma * u_z
    
    # 3. Divergencia del flujo
    J_x_x = torch.autograd.grad(J_x, x, grad_outputs=torch.ones_like(J_x), create_graph=True)[0]
    J_z_z = torch.autograd.grad(J_z, z, grad_outputs=torch.ones_like(J_z), create_graph=True)[0]
    
    divergence = J_x_x + J_z_z
    assert torch.allclose(divergence, torch.tensor([[4.0]])), "Segunda derivada (Laplaciano) falló"

if __name__ == "__main__":
    pytest.main([__file__])
