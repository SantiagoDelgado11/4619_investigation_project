import torch

def compute_pde_loss(u, sigma, x, z, injection_source):
    """
    Calcula el residual de la Ecuación de Poisson (ERT):
    nabla . (sigma * nabla u) = I * delta(x)
    
    Donde:
      u: (batch_size, 1) Tensor de potencial.
      sigma: (batch_size, 1) Tensor de conductividad.
      x: (batch_size, 1) Coordenada espacial x.
      z: (batch_size, 1) Coordenada espacial z (profundidad).
      injection_source: (batch_size, 1) Término fuente/sumidero (I * delta)
    """
    # 1. Gradientes de U (nabla u)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # 2. Flujo conductivo (sigma * nabla u)
    J_x = sigma * u_x
    J_z = sigma * u_z
    
    # 3. Divergencia del flujo conductivo nabla . (sigma * nabla u)
    J_x_x = torch.autograd.grad(J_x, x, grad_outputs=torch.ones_like(J_x), create_graph=True)[0]
    J_z_z = torch.autograd.grad(J_z, z, grad_outputs=torch.ones_like(J_z), create_graph=True)[0]
    
    divergence = J_x_x + J_z_z
    
    # 4. Cálculo del Residual (Loss_PDE)
    # expected: divergence = injection_source (para inyección de corriente puntual)
    pde_residual = divergence - injection_source
    return torch.mean(pde_residual ** 2)

def compute_tv_loss(sigma, x, z):
    """
    Regularización Total Variation (TV) en la conductividad.
    Ayuda a recuperar distribuciones de subsuelo por bloques (blocky).
    """
    sigma_x = torch.autograd.grad(sigma, x, grad_outputs=torch.ones_like(sigma), create_graph=True)[0]
    sigma_z = torch.autograd.grad(sigma, z, grad_outputs=torch.ones_like(sigma), create_graph=True)[0]
    
    # Norma L1 aproximada del gradiente espacial
    tv_loss = torch.mean(torch.abs(sigma_x) + torch.abs(sigma_z))
    return tv_loss
