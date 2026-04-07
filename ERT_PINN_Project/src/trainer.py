import torch
import torch.optim as optim
import wandb
from src.physics import compute_bc_loss, compute_pde_loss_anomaly, compute_primary_potential

class ERTTrainer:
    """
    Clase que orquesta el entrenamiento de las redes PINN_Sigma y PINN_U,
    usando formulación Anomaly (Secondary Potential) y Redes Condicionadas.
    """
    def __init__(self, model_sigma, model_u, dataloader, config):
        self.model_sigma = model_sigma
        self.model_u = model_u
        self.dataloader = dataloader
        self.config = config
        
        self.optimizer = optim.Adam([
            {'params': self.model_sigma.parameters(), 'lr': config['lr_sigma']},
            {'params': self.model_u.parameters(), 'lr': config['lr_u']}
        ])
        
        self.lambda_data = config.get('lambda_data', 1.0)
        self.lambda_pde = config.get('lambda_pde', 0.1)
        self.lambda_bc = 0.05 # Nuevo peso para boundary conditions
        
    def train_epoch(self, epoch):
        self.model_sigma.train()
        self.model_u.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            batch_size = batch['M_x'].shape[0]
            
            # --- 1. DATA LOSS (V_pred = V_p + V_s) ---
            m_x = batch['M_x'].unsqueeze(1)
            n_x = batch['N_x'].unsqueeze(1)
            a_x = batch['A_x'].unsqueeze(1)
            b_x = batch['B_x'].unsqueeze(1)
            z_surf = torch.zeros_like(m_x)
            
            # V_p Analítico
            u_p_M = compute_primary_potential(m_x, z_surf, pos_A=(a_x, z_surf), pos_B=(b_x, z_surf))
            u_p_N = compute_primary_potential(n_x, z_surf, pos_A=(a_x, z_surf), pos_B=(b_x, z_surf))
            V_p = u_p_M - u_p_N
            
            # V_s Red Neuronal Condicionada
            u_s_M = self.model_u(m_x, z_surf, a_x, z_surf, b_x, z_surf)
            u_s_N = self.model_u(n_x, z_surf, a_x, z_surf, b_x, z_surf)
            V_s = u_s_M - u_s_N
            
            V_pred = V_p + V_s
            V_meas = batch['V_meas'].unsqueeze(1)
            data_loss = torch.mean((V_pred - V_meas)**2)
            
            # --- 2. PDE LOSS (Collocation Points en Malla Dominio + Inyecciones Mixtas) ---
            # Dominio x:[0, 100], z:[0, 50]
            x_col = (torch.rand(batch_size, 1, requires_grad=True) * 100.0)
            z_col = (torch.rand(batch_size, 1, requires_grad=True) * 50.0)
            
            a_x_col = torch.rand(batch_size, 1) * 100.0
            b_x_col = torch.rand(batch_size, 1) * 100.0
            z_surf_col = torch.zeros_like(a_x_col)
            
            u_s_pred = self.model_u(x_col, z_col, a_x_col, z_surf_col, b_x_col, z_surf_col)
            sigma_pred = self.model_sigma(x_col, z_col)
            
            pde_loss = compute_pde_loss_anomaly(
                u_s_pred, sigma_pred, x_col, z_col, 
                pos_A=(a_x_col, z_surf_col), pos_B=(b_x_col, z_surf_col)
            )
            
            # --- 3. BOUNDARY CONDITION LOSS ---
            x_bc = torch.rand(batch_size, 1) * 100.0
            z_bc_surf = torch.zeros_like(x_bc)
            z_bc_deep = torch.ones_like(x_bc) * 50.0 
            
            bc_loss = compute_bc_loss(
                self.model_u, x_bc, z_bc_surf, z_bc_deep, 
                a_x_col, z_surf_col, b_x_col, z_surf_col
            )
            
            # --- COMPOSITE LOSS ---
            loss = (self.lambda_data * data_loss) + (self.lambda_pde * pde_loss) + (self.lambda_bc * bc_loss)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.dataloader)
        if self.config.get('use_wandb', False):
            wandb.log({"loss": avg_loss, "epoch": epoch})
            
        return avg_loss

    def train(self):
        epochs = self.config['epochs']
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.6f}")
