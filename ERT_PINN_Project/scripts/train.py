import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import PINN_Sigma, PINN_U
from src.data_loader import ERTDataset
from src.trainer import ERTTrainer
from src.utils import save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Tren de Entrenamiento PINN para ERT")
    parser.add_argument('--config', type=str, default='configs/exp_01_synthetic.yaml', help='Path to YAML config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    net_cfg = config_dict['network']
    train_cfg = config_dict['training']
    loss_cfg = config_dict['loss_weights']
    
    # 1. Instanciar Redes
    model_sigma = PINN_Sigma(hidden_features=net_cfg['sigma_hidden_features'], hidden_layers=net_cfg['sigma_hidden_layers'])
    model_u = PINN_U(hidden_features=net_cfg['u_hidden_features'], hidden_layers=net_cfg['u_hidden_layers'])
    
    # 2. Cargar Dataloader
    dataset = ERTDataset(data_path=config_dict['data']['train_data_path'])
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    
    # 3. Empacar config integral para el trainer
    trainer_cfg = {
        'lr_sigma': train_cfg['lr_sigma'],
        'lr_u': train_cfg['lr_u'],
        'epochs': train_cfg['epochs'],
        'lambda_data': loss_cfg['lambda_data'],
        'lambda_pde': loss_cfg['lambda_pde'],
        'lambda_tv': loss_cfg['lambda_tv'],
        'use_wandb': config_dict['logging']['use_wandb']
    }
    
    if trainer_cfg['use_wandb']:
        import wandb
        wandb.init(project=config_dict['logging']['project_name'], config=config_dict)
        
    # 4. Lanzar Entrenamiento
    print("Iniciando Entrenamiento ERT PINN...")
    trainer = ERTTrainer(model_sigma, model_u, dataloader, trainer_cfg)
    trainer.train()
    
    # 5. Guardar Pesos Finales
    save_checkpoint(model_sigma, model_u, trainer.optimizer, trainer_cfg['epochs'], 'pinn_ert_final.pt')

if __name__ == '__main__':
    main()
