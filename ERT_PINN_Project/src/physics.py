import torch
import math

def compute_primary_potential(x, z, pos_A, pos_B, sigma_0=1.0, I=1.0):
    """
    Calcula el potencial primario analítico u_p asumiendo un semi-espacio (Z>=0).
    u_p = (I / (2 * pi * sigma_0)) * (1/r_A - 1/r_B)
    """
    x_a, z_a = pos_A
    x_b, z_b = pos_B
    
    eps = 1e-6 # Previene división por 0
    r_A = torch.sqrt((x - x_a)**2 + (z - z_a)**2 + eps)
    r_B = torch.sqrt((x - x_b)**2 + (z - z_b)**2 + eps)
    
    u_p = (I / (2 * math.pi * sigma_0)) * ((1.0 / r_A) - (1.0 / r_B))
    return u_p

def compute_pde_loss_anomaly(u_s, sigma, x, z, pos_A, pos_B, sigma_0=1.0, I=1.0):
    """
    Residual de la PDE para el campo Secundario (Anomaly Approach).
    div(sigma * grad(u_s)) = - div((sigma - sigma_0) * grad(u_p))
    """
    u_p = compute_primary_potential(x, z, pos_A, pos_B, sigma_0, I)
    
    # Gradientes del campo Secundario
    u_s_x = torch.autograd.grad(u_s, x, grad_outputs=torch.ones_like(u_s), create_graph=True)[0]
    u_s_z = torch.autograd.grad(u_s, z, grad_outputs=torch.ones_like(u_s), create_graph=True)[0]
    
    J_s_x = sigma * u_s_x
    J_s_z = sigma * u_s_z
    
    J_s_x_x = torch.autograd.grad(J_s_x, x, grad_outputs=torch.ones_like(J_s_x), create_graph=True)[0]
    J_s_z_z = torch.autograd.grad(J_s_z, z, grad_outputs=torch.ones_like(J_s_z), create_graph=True)[0]
    div_J_s = J_s_x_x + J_s_z_z
    
    # Gradientes del campo Primario
    u_p_x = torch.autograd.grad(u_p, x, grad_outputs=torch.ones_like(u_p), create_graph=True)[0]
    u_p_z = torch.autograd.grad(u_p, z, grad_outputs=torch.ones_like(u_p), create_graph=True)[0]
    
    delta_sigma = sigma - sigma_0
    J_p_anom_x = delta_sigma * u_p_x
    J_p_anom_z = delta_sigma * u_p_z
    
    # Divergencia del flujo anómalo
    J_p_anom_x_x = torch.autograd.grad(J_p_anom_x, x, grad_outputs=torch.ones_like(J_p_anom_x), create_graph=True)[0]
    J_p_anom_z_z = torch.autograd.grad(J_p_anom_z, z, grad_outputs=torch.ones_like(J_p_anom_z), create_graph=True)[0]
    div_J_p_anom = J_p_anom_x_x + J_p_anom_z_z
    
    # Residual
    pde_residual = div_J_s + div_J_p_anom
    return torch.mean(pde_residual ** 2)

def compute_bc_loss(model_u, x_bc, z_bc_surf, z_bc_deep, a_x, a_z, b_x, b_z):
    """
    Boundary Conditions para el campo Secundario u_s.
    """
    # Neumann BC en superficie (z=0): du_s/dz = 0
    z_surf = z_bc_surf.clone().requires_grad_(True)
    u_s_surf = model_u(x_bc, z_surf, a_x, a_z, b_x, b_z)
    u_s_z = torch.autograd.grad(u_s_surf, z_surf, grad_outputs=torch.ones_like(u_s_surf), create_graph=True)[0]
    loss_neumann = torch.mean(u_s_z**2)
    
    # Dirichlet BC infinito (profundidad z_max): u_s = 0 
    u_s_deep = model_u(x_bc, z_bc_deep, a_x, a_z, b_x, b_z)
    loss_dirichlet = torch.mean(u_s_deep**2)
    
    return loss_neumann + loss_dirichlet