import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(model_sigma, model_u, optimizer, epoch, path):
    """Guarda los pesos del modelo en un path específico."""
    torch.save({
        'epoch': epoch,
        'model_sigma_state_dict': model_sigma.state_dict(),
        'model_u_state_dict': model_u.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model_sigma, model_u, optimizer, path):
    """Carga los pesos de un archivo .pt."""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
    checkpoint = torch.load(path)
    model_sigma.load_state_dict(checkpoint['model_sigma_state_dict'])
    model_u.load_state_dict(checkpoint['model_u_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint['epoch']

def plot_resistivity_profile(sigma_net, x_min=0, x_max=100, z_min=0, z_max=50, resolution=100):
    """Genera un mapa 2D del perfil de resistividad conductividad predicho por PINN_Sigma."""
    x = np.linspace(x_min, x_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    XX, ZZ = np.meshgrid(x, z)
    
    # Flatten y a Tensor
    x_tensor = torch.tensor(XX.flatten(), dtype=torch.float32).unsqueeze(1)
    z_tensor = torch.tensor(ZZ.flatten(), dtype=torch.float32).unsqueeze(1)
    
    sigma_net.eval()
    with torch.no_grad():
        pred_sigma = sigma_net(x_tensor, z_tensor).numpy()
        
    pred_sigma = pred_sigma.reshape(resolution, resolution)
    
    # (Opcional) Convertir r = 1/sigma para resistividad aparente real
    res_app = 1.0 / (pred_sigma + 1e-8)
    
    plt.figure(figsize=(10, 5))
    plt.contourf(XX, ZZ, res_app, levels=50, cmap='jet')
    plt.colorbar(label='Resistivity (Ohm.m)')
    plt.title("Predicted Resistivity Tomogram")
    plt.xlabel('X (m)')
    plt.ylabel('Depth Z (m)')
    plt.gca().invert_yaxis() # Z grows downwards
    plt.show()
