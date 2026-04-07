import torch
import torch.nn as nn

class PINN_Sigma(nn.Module):
    """
    Red Neuronal para predecir el campo de Conductividad (o Resistividad).
    Entrada: (x, z)
    Salida: sigma(x, z) (> 0)
    """
    def __init__(self, in_features=2, hidden_features=64, hidden_layers=4, out_features=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Softplus())  # Activación suave para cálculo de derivadas continuas
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Softplus())
            
        layers.append(nn.Linear(hidden_features, out_features))
        layers.append(nn.Softplus()) # Asegurar que la conductividad predicha sea estrictamente positiva
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, z):
        # x, z shape: (batch_size, 1)
        coords = torch.cat([x, z], dim=1)
        sigma = self.net(coords)
        return sigma

class PINN_U(nn.Module):
    """
    Red Neuronal condicionada para predecir el campo de Potencial Secundario.
    Entrada: (x, z, a_x, a_z, b_x, b_z)
    Salida: u_s(x, z)
    """
    def __init__(self, in_features=6, hidden_features=64, hidden_layers=4, out_features=1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Softplus())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Softplus())
            
        layers.append(nn.Linear(hidden_features, out_features))
        # Sin activación en la última capa
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, z, a_x, a_z, b_x, b_z):
        coords = torch.cat([x, z, a_x, a_z, b_x, b_z], dim=1)
        u_s = self.net(coords)
        return u_s
