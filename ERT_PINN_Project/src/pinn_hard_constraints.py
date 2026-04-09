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


class PINN_U_3D_Hard(nn.Module):
    """
    Red Neuronal para predecir el Potencial Eléctrico 3D (Hard Constraints).
    
    Implementa un Ansatz que fuerza matemáticamente el cumplimiento de 
    las condiciones de frontera sin importar los pesos de la red:
    - Dirichlet (v -> 0 cuando r -> inf)
    - Neumann (dv/dz = 0 en z=0)
    """
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=4, out_features=1, L=100.0):
        super().__init__()
        self.L = L # Parámetro espacial para escalado del decaimiento
        
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y, z):
        # 1. Alimentamos a la red con z^2 en vez de z para asegurar simetría respecto a z=0
        z_squared = z**2
        coords = torch.cat([x, y, z_squared], dim=1)
        raw_u = self.net(coords)
        
        # 2. Ansatz para imponer Hard Constraints:
        # Multiplicamos por una función envolvente gaussiana.
        # r^2 = x^2 + y^2 + z^2
        r2 = x**2 + y**2 + z**2
        decay_envelope = torch.exp(-r2 / (self.L**2))
        
        # Demostración del Ansatz:
        # A) En r -> infinito, decay_envelope -> 0. Luego u -> 0. (Cumple Dirichlet)
        # B) Para la derivada en la superficie (z=0):
        # d_u/d_z = decay_envelope * (-2z/L^2) * raw_u + decay_envelope * d_raw_u/d_z
        # Por la regla de la cadena: d_raw_u/d_z = (d_raw_u/d_z^2) * 2z
        # En consecuencia ambos sumandos tienen 'z'. Al evaluar en z=0, d_u/d_z = 0. (Cumple Neumann Estricto).
        
        u_constrained = decay_envelope * raw_u
        return u_constrained


def compute_pde_loss_3d_hard(u_model, sigma_model, x, y, z, pos_A, pos_B, I=1.0, epsilon=0.1):
    """
    Calcula el residual de la PDE con fuentes puntuales en 3D.
    Ya no se calculan las condiciones de frontera aquí, solo la física del dominio.
    """
    u = u_model(x, y, z)
    sigma = sigma_model(x, y, z)
    
    # 1. Gradiente de u
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    Jx = sigma * u_x
    Jy = sigma * u_y
    Jz = sigma * u_z
    
    # 2. Divergencia
    Jx_x = torch.autograd.grad(Jx, x, grad_outputs=torch.ones_like(Jx), create_graph=True)[0]
    Jy_y = torch.autograd.grad(Jy, y, grad_outputs=torch.ones_like(Jy), create_graph=True)[0]
    Jz_z = torch.autograd.grad(Jz, z, grad_outputs=torch.ones_like(Jz), create_graph=True)[0]
    
    divergence = Jx_x + Jy_y + Jz_z
    
    # 3. Término sumidero-fuente (Electrodos A y B) en 3D
    xA, yA, zA = pos_A
    xB, yB, zB = pos_B
    norm = 1.0 / ((2 * math.pi * epsilon**2)**1.5)
    delta_A = norm * torch.exp(-((x - xA)**2 + (y - yA)**2 + (z - zA)**2) / (2 * epsilon**2))
    delta_B = norm * torch.exp(-((x - xB)**2 + (y - yB)**2 + (z - zB)**2) / (2 * epsilon**2))
    
    source = I * delta_A - I * delta_B
    
    # Residual: div(sigma*grad(u)) = fuente-sumidero <=> divergence + source = 0
    pde_residual = divergence + source
    return torch.mean(pde_residual**2)


def train_step_hard(u_model, sigma_model, optimizer, domain_pts, data_pts, vals_obs, pos_A, pos_B):
    """
    Paso de entrenamiento simplificado debido a las Hard Constraints.
    """
    optimizer.zero_grad()
    
    x, y, z = domain_pts
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    
    # Loss PDE (Término Físico central)
    loss_pde = compute_pde_loss_3d_hard(u_model, sigma_model, x, y, z, pos_A, pos_B)
    
    # Loss Data (Fidelidad de datos)
    xd, yd, zd = data_pts
    u_pred = u_model(xd, yd, zd)
    loss_data = torch.mean((u_pred - vals_obs)**2)
    
    # Loss Total Simplificada: NO hay lambdas de frontera (lambda_neu, lambda_dir)
    lambda_pde = 1.0
    lambda_data = 100.0
    
    total_loss = (lambda_pde * loss_pde) + (lambda_data * loss_data)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_pde.item(), loss_data.item()

if __name__ == "__main__":
    # Test local de construcción y gradientes
    print("Iniciando entrenamiento PINN 3D con Hard Constraints...")
    u_net = PINN_U_3D_Hard(L=50.0)
    sigma_net = PINN_Sigma_3D()
    
    optimizer = torch.optim.Adam(list(u_net.parameters()) + list(sigma_net.parameters()), lr=1e-3)
    
    # Puntos de dominio para PDE
    domain_x = torch.rand(100, 1) * 10 - 5
    domain_y = torch.rand(100, 1) * 10 - 5
    domain_z = torch.rand(100, 1) * 10  # z >= 0
    
    # Datos sintéticos para pérdida Data
    data_x = torch.zeros(5, 1)
    data_y = torch.zeros(5, 1)
    data_z = torch.zeros(5, 1)
    vals_obs = torch.ones(5, 1) * 0.5
    
    pos_A = (-2.0, 0.0, 0.0)
    pos_B = (2.0, 0.0, 0.0)
    
    loss_val = train_step_hard(u_net, sigma_net, optimizer, 
                               (domain_x, domain_y, domain_z), 
                               (data_x, data_y, data_z), 
                               vals_obs, pos_A, pos_B)
                          
    print(f"Iter 1 | Total Loss: {loss_val[0]:.4f} | PDE: {loss_val[1]:.4f} | Data: {loss_val[2]:.4f}")
    
    # Comprobación analítica rápida de Neumann: dh/dz en z=0
    test_x = torch.tensor([[1.0]], requires_grad=True)
    test_y = torch.tensor([[1.0]], requires_grad=True)
    test_z = torch.tensor([[0.0]], requires_grad=True) # Superficie
    
    u_test = u_net(test_x, test_y, test_z)
    u_z_test = torch.autograd.grad(u_test, test_z, grad_outputs=torch.ones_like(u_test))[0]
    print(f"Comprobación Analítica de Neumann (d_u/d_z en z=0): {u_z_test.item()} (Debería ser ~0)")
