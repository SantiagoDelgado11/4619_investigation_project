import torch
import torch.nn as nn
import math

class PINN_Sigma_3D(nn.Module):
    """
    Red Neuronal para predecir el campo de Conductividad 3D.
    Entrada: (x, y, z)
    Salida: sigma(x, y, z) (> 0)
    """
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=4, out_features=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Tanh()) 
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_features, out_features))
        layers.append(nn.Softplus()) # Asegurar que sigma > 0
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y, z):
        coords = torch.cat([x, y, z], dim=1)
        return self.net(coords)

class PINN_U_3D_Soft(nn.Module):
    """
    Red Neuronal para predecir el campo de Potencial Eléctrico 3D (Soft Constraints).
    Entrada: (x, y, z)
    Salida: u(x, y, z)
    """
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=4, out_features=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y, z):
        coords = torch.cat([x, y, z], dim=1)
        return self.net(coords)


def compute_pde_loss_3d_soft(u_model, sigma_model, x, y, z, pos_A, pos_B, I=1.0, epsilon=0.1):
    """
    Calcula el residual de la PDE con fuentes puntuales en 3D.
    Ecuación: - div(sigma * grad(u)) = I*delta(r-r_A) - I*delta(r-r_B)
    """
    # Predicciones de redes
    u = u_model(x, y, z)
    sigma = sigma_model(x, y, z)
    
    # 1. Gradiente de u: grad(u)
    # doc: autograd.grad se usa para derivar u respecto a x, y, z, manteniendo el grafo para segundas derivadas.
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Flujo de corriente: J = sigma * grad(u)
    Jx = sigma * u_x
    Jy = sigma * u_y
    Jz = sigma * u_z
    
    # 2. Divergencia: div(J)
    Jx_x = torch.autograd.grad(Jx, x, grad_outputs=torch.ones_like(Jx), create_graph=True)[0]
    Jy_y = torch.autograd.grad(Jy, y, grad_outputs=torch.ones_like(Jy), create_graph=True)[0]
    Jz_z = torch.autograd.grad(Jz, z, grad_outputs=torch.ones_like(Jz), create_graph=True)[0]
    
    divergence = Jx_x + Jy_y + Jz_z
    
    # 3. Término Fuente (Electrodos A y B) usando gaussiana 3D
    xA, yA, zA = pos_A
    xB, yB, zB = pos_B
    
    # Constante de normalización para gaussiana 3D
    norm = 1.0 / ((2 * math.pi * epsilon**2)**1.5)
    
    delta_A = norm * torch.exp(-((x - xA)**2 + (y - yA)**2 + (z - zA)**2) / (2 * epsilon**2))
    delta_B = norm * torch.exp(-((x - xB)**2 + (y - yB)**2 + (z - zB)**2) / (2 * epsilon**2))
    
    source = I * delta_A - I * delta_B
    
    # Residual de la PDE: div(J) + source = 0
    pde_residual = divergence + source
    
    return torch.mean(pde_residual**2)


def compute_boundary_loss_soft(u_model, boundary_points_surface, boundary_points_far):
    """
    Calcula la pérdida por condiciones de frontera (Soft Constraints).
    """
    # 1. Condición de Neumann en la superficie (z=0): d_u/d_z = 0
    x_s, y_s, z_s = boundary_points_surface
    u_s = u_model(x_s, y_s, z_s)
    u_s_z = torch.autograd.grad(u_s, z_s, grad_outputs=torch.ones_like(u_s), create_graph=True)[0]
    loss_neumann = torch.mean(u_s_z**2)
    
    # 2. Condición de Dirichlet en el infinito (lejos): u = 0
    x_f, y_f, z_f = boundary_points_far
    u_f = u_model(x_f, y_f, z_f)
    loss_dirichlet = torch.mean(u_f**2)
    
    return loss_neumann, loss_dirichlet


def train_step(u_model, sigma_model, optimizer, domain_pts, surface_pts, far_pts, data_pts, vals_obs, pos_A, pos_B):
    optimizer.zero_grad()
    
    x, y, z = domain_pts
    # Habilitar gradientes para coordenadas espaciales
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    # Loss PDE
    loss_pde = compute_pde_loss_3d_soft(u_model, sigma_model, x, y, z, pos_A, pos_B)
    
    # Loss Boundary (Soft Constraints)
    xs, ys, zs = surface_pts
    xs.requires_grad_(True)
    ys.requires_grad_(True)
    zs.requires_grad_(True)
    loss_neu, loss_dir = compute_boundary_loss_soft(u_model, (xs, ys, zs), far_pts)
    
    # Loss Data (observaciones en electrodos)
    xd, yd, zd = data_pts
    u_pred = u_model(xd, yd, zd)
    loss_data = torch.mean((u_pred - vals_obs)**2)
    
    # Loss total con penalizaciones (Soft Constraints)
    lambda_pde = 1.0
    lambda_neu = 10.0
    lambda_dir = 10.0
    lambda_data = 100.0
    
    total_loss = (lambda_pde * loss_pde) + (lambda_neu * loss_neu) + (lambda_dir * loss_dir) + (lambda_data * loss_data)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_pde.item(), loss_neu.item(), loss_dir.item(), loss_data.item()

if __name__ == "__main__":
    # Script listo para ser ejecutado (Demo simplificada)
    print("Iniciando entrenamiento PINN 3D con Soft Constraints...")
    u_net = PINN_U_3D_Soft()
    sigma_net = PINN_Sigma_3D()
    
    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=1e-3)
    
    # Puntos de colocación dummy
    domain_x = torch.rand(100, 1) * 10 - 5
    domain_y = torch.rand(100, 1) * 10 - 5
    domain_z = torch.rand(100, 1) * 10  # z >= 0 (profundidad)
    
    surface_x = torch.rand(50, 1) * 10 - 5
    surface_y = torch.rand(50, 1) * 10 - 5
    surface_z = torch.zeros(50, 1) # Superficie en z=0
    
    far_x = torch.ones(50, 1) * 100
    far_y = torch.ones(50, 1) * 100
    far_z = torch.ones(50, 1) * 100
    
    data_x = torch.zeros(5, 1)
    data_y = torch.zeros(5, 1)
    data_z = torch.zeros(5, 1)
    vals_obs = torch.ones(5, 1) * 0.5
    
    pos_A = (-2.0, 0.0, 0.0)
    pos_B = (2.0, 0.0, 0.0)
    
    loss_val = train_step(u_net, sigma_net, optimizer, 
                          (domain_x, domain_y, domain_z), 
                          (surface_x, surface_y, surface_z), 
                          (far_x, far_y, far_z), 
                          (data_x, data_y, data_z), vals_obs, pos_A, pos_B)
                          
    print(f"Iter 1 | Total Loss: {loss_val[0]:.4f} | PDE: {loss_val[1]:.4f}")
