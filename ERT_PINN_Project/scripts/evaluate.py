import argparse
import torch
import yaml
from src.models import PINN_Sigma, PINN_U
from src.utils import load_checkpoint, plot_resistivity_profile

def evaluate():
    parser = argparse.ArgumentParser(description="Scripts de inferencia y plot para ERT PINN")
    parser.add_argument('--config', type=str, default='configs/exp_01_synthetic.yaml')
    parser.add_argument('--checkpoint', type=str, default='pinn_ert_final.pt')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Recrear arquitecturas
    net_cfg = config['network']
    model_sigma = PINN_Sigma(hidden_features=net_cfg['sigma_hidden_features'], hidden_layers=net_cfg['sigma_hidden_layers'])
    model_u = PINN_U(hidden_features=net_cfg['u_hidden_features'], hidden_layers=net_cfg['u_hidden_layers'])
    
    # Cargar pesos aprendidos de sigma
    load_checkpoint(model_sigma, model_u, None, args.checkpoint)
    
    # Inferir sobre dominio 2D continuo y plotear
    print("Borrando gráfico de resistividad invertida...")
    plot_resistivity_profile(model_sigma, x_min=0, x_max=50, z_min=0, z_max=20, resolution=150)

if __name__ == "__main__":
    evaluate()
