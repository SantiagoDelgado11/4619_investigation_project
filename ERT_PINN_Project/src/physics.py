import torch
import math

def compute_pde_loss_dipole(u, sigma, x, z, pos_A, pos_B, I=1.0, epsilon=0.05):
    """
    Calcula el residual para un arreglo de dos electrodos (Dipolo).
    
    Nuevos Parámetros:
      pos_A: tupla (x_a, z_a) - Coordenadas del electrodo de INYECCIÓN (+).
      pos_B: tupla (x_b, z_b) - Coordenadas del electrodo de EXTRACCIÓN (-).
      I: Magnitud de la corriente inyectada (escalar).
      epsilon: Radio de suavizado de la corriente (controla qué tan "puntual" es el electrodo).
    """
    # 1. Gradientes y Divergencia (La física del flujo se mantiene igual)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    J_x = sigma * u_x
    J_z = sigma * u_z
    
    J_x_x = torch.autograd.grad(J_x, x, grad_outputs=torch.ones_like(J_x), create_graph=True)[0]
    J_z_z = torch.autograd.grad(J_z, z, grad_outputs=torch.ones_like(J_z), create_graph=True)[0]
    
    divergence = J_x_x + J_z_z
    
    # 2. NUEVO: Modelar los dos electrodos con una aproximación Gaussiana
    x_a, z_a = pos_A
    x_b, z_b = pos_B
    
    # Electrodo A (Inyección: entra corriente)
    delta_A = (1.0 / (2 * math.pi * epsilon**2)) * torch.exp(
        -((x - x_a)**2 + (z - z_a)**2) / (2 * epsilon**2)
    )
    
    # Electrodo B (Extracción: sale corriente)
    delta_B = (1.0 / (2 * math.pi * epsilon**2)) * torch.exp(
        -((x - x_b)**2 + (z - z_b)**2) / (2 * epsilon**2)
    )
    
    # Término fuente total: Sumamos la inyección y restamos la extracción
    injection_source = (I * delta_A) - (I * delta_B)
    
    # 3. Cálculo del Residual
    pde_residual = divergence - injection_source
    return torch.mean(pde_residual ** 2)