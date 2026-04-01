import torch
import torch.optim as optim
import wandb
from src.physics import compute_pde_loss, compute_tv_loss

class ERTTrainer:
    """
    Clase que orquesta el entrenamiento de las redes PINN_Sigma y PINN_U,
    balanceando el loss de las pruebas físicas, matemáticas y regularizaciones.
    """
    def __init__(self, model_sigma, model_u, dataloader, config):
        self.model_sigma = model_sigma
        self.model_u = model_u
        self.dataloader = dataloader
        self.config = config
        
        # Optimizador Adam
        self.optimizer = optim.Adam([
            {'params': self.model_sigma.parameters(), 'lr': config['lr_sigma']},
            {'params': self.model_u.parameters(), 'lr': config['lr_u']}
        ])
        
        # Loss multipliers
        self.lambda_data = config.get('lambda_data', 1.0)
        self.lambda_pde = config.get('lambda_pde', 0.1)
        self.lambda_tv = config.get('lambda_tv', 0.01)
        
    def train_epoch(self, epoch):
        self.model_sigma.train()
        self.model_u.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            
            # Forward real measures: [Implement measurement loss matching V_pred con V_meas]
            # data_loss = calculate_voltage_loss(self.model_u, batch)
            data_loss = torch.tensor(0.0, requires_grad=True) # Placeholder
            
            # PDE Loss (Collocation points over domain)
            # Sample random points in the 2D domain (x, z)
            # x_col = torch.rand(batch_size, 1, requires_grad=True)
            # z_col = torch.rand(batch_size, 1, requires_grad=True)
            # u_pred = self.model_u(x_col, z_col)
            # sigma_pred = self.model_sigma(x_col, z_col)
            # pde_loss = compute_pde_loss(u_pred, sigma_pred, x_col, z_col, injection=0.0)
            pde_loss = torch.tensor(0.0, requires_grad=True) # Placeholder
            
            # TV Loss on conductivity map
            # tv_loss = compute_tv_loss(sigma_pred, x_col, z_col)
            tv_loss = torch.tensor(0.0, requires_grad=True) # Placeholder
            
            # Composite Loss
            loss = (self.lambda_data * data_loss) + (self.lambda_pde * pde_loss) + (self.lambda_tv * tv_loss)
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
