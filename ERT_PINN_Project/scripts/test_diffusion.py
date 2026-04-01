import sys
import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt

from algos.dps import DPS
from algos.ddnm import DDNM
from algos.diffpir import DiffPIR
from guided_diffusion.script_util import create_model
from src.diffusion_physics import ERT_Forward_Operator

def test_diffusion_sampling():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp_04_diffusion.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando Muestreo de Difusión DPS en {device} para ERT")
    
    ############################
    # 1. CARGAR U-NET PRIOR #
    ############################
    # (Por ahora creamos uno dummy si el path falla, pero debería cargar tus pesos)
    net_cfg = cfg['diffusion_model']
    net = create_model(image_size=net_cfg['image_size'],
                       num_channels=net_cfg['num_channels_model'],
                       num_res_blocks=3,
                       input_channels=1).to(device)  # ERT es conductividad 1 canal, no RGB
                       
    try:
        ckpt = torch.load(net_cfg['weights_path'], map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model_state"])
        print("Pesos STSIVA Originales cargados correctamente.")
    except Exception as e:
        print(f"Warning: No se pudieron cargar los pesos from {net_cfg['weights_path']}. \nAsegúrate que el UNet entrenado en STSIVA tenga 1 canal, o haz transfer learning.")
    
    net.eval()
    
    #######################################
    # 2. CARGAR FORWARD ERT Y MEDICIONES  #
    #######################################
    inv_model = ERT_Forward_Operator(im_size=net_cfg['image_size'], num_electrodes=cfg['ert']['num_electrodes'])
    
    # Ground Truth de Conductividad (Test Síntesis)
    # Rango de difusión: [-1, 1], mapeado luego a [sigma_min, sigma_max] en ERT
    GT_sigma = torch.randn(1, 1, net_cfg['image_size'], net_cfg['image_size']).clamp(-1, 1).to(device)
    
    # Voltaje limpio simulado + Ruido experimental
    y_meas = inv_model.forward_pass(GT_sigma)
    y_meas = y_meas + 0.05 * torch.randn_like(y_meas)
    
    #######################################
    # 3. LANZAR EL DIFUSION INFERENCE     #
    #######################################
    algo_name = cfg['sampling']['algo']
    print(f"Ejecutando {algo_name}...")
    
    if algo_name == "DPS":
        diff = DPS(device=device,
                   img_size=net_cfg['image_size'],
                   noise_steps=cfg['sampling']['sampling_steps'],
                   schedule_name="cosine",
                   channels=1,
                   scale=cfg['sampling']['dps_scale'],
                   clip_denoised=False)
                   
        reconstruction = diff.sample(model=net,
                                     y=y_meas,
                                     forward_pass=inv_model.forward_pass)
                                     
    elif algo_name == "DDNM":
        diff = DDNM(device=device,
                    img_size=net_cfg['image_size'],
                    noise_steps=cfg['sampling']['sampling_steps'],
                    schedule_name="cosine",
                    channels=1,
                    eta=cfg['sampling']['ddnm_eta'])
                    
        reconstruction = diff.sample(model=net, y=y_meas,
                                     forward_pass=inv_model.forward_pass,
                                     pseudo_inverse=inv_model.pseudo_inverse,
                                     ground_truth=GT_sigma, track_metrics=False)
                                     
    elif algo_name == "DiffPIR":
        diff = DiffPIR(device=device,
                       img_size=net_cfg['image_size'],
                       noise_steps=cfg['sampling']['sampling_steps'],
                       schedule_name="cosine",
                       channels=1,
                       cg_iters=cfg['sampling']['CG_iters_diffpir'],
                       noise_level_img=cfg['sampling']['noise_level_img'],
                       iter_num=1000, eta=0, zeta=1)
                       
        reconstruction = diff.sample(model=net, y=y_meas,
                                     forward_pass=inv_model.forward_pass,
                                     transpose_pass=inv_model.transpose_pass)
                                     
    else:
        raise ValueError(f"Algoritmo {algo_name} no soportado.")
        
    #######################################
    # 4. PLOT Y EVALUACION                #
    #######################################
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(GT_sigma[0,0].cpu().numpy(), cmap='turbo')
    ax[0].set_title("Conductividad Subsuelo (Ground Truth)")
    
    ax[1].imshow(reconstruction[0,0].cpu().numpy(), cmap='turbo')
    ax[1].set_title(f"Reconstrucción Inversa ({algo_name} Prior)\nBasado en Volts ERT")
    plt.suptitle("Tomografía ERT guiada por Modelos de Difusión de STSIVA_2026")
    plt.show()

if __name__ == "__main__":
    test_diffusion_sampling()
