import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_random_subsurface(im_size=32):
    """
    Genera un subsuelo virtual 2D con capas, fallas y anomalías (rocas/agua).
    Todo esto es matemático y sintético, ¡NO requiere datos reales previos!
    Retorna: tensor de conductividad sigma de tamaño (1, im_size, im_size)
    """
    # Fondo base (ej. conductividad típica de la tierra = 0.1)
    sigma = torch.ones((im_size, im_size)) * 0.1
    
    # Añadir 1 a 3 cajas (bloques de distinta resistividad)
    num_blocks = np.random.randint(1, 4)
    for _ in range(num_blocks):
        x0 = np.random.randint(0, im_size - 5)
        w = np.random.randint(5, im_size // 2)
        z0 = np.random.randint(0, im_size - 5)
        h = np.random.randint(5, im_size // 2)
        
        # Valor de conductividad (ej. 1.0 agua salada, 0.01 lecho rocoso)
        val = np.random.uniform(0.01, 1.0)
        sigma[z0:min(z0+h, im_size), x0:min(x0+w, im_size)] = val

    # Añadir suavizado de la naturaleza (filtro gaussiano simple simulado)
    # sigma = gaussian_filter(sigma) # Idealmente difuminar bordes
    return sigma.unsqueeze(0)

def master_generator(num_samples=1000, output_dir="data/synthetic/geology_priors"):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    print(f"Generando {num_samples} mapas de subsuelo falso matemáticamente...")
    for i in range(num_samples):
        geology = generate_random_subsurface(im_size=32)
        images.append(geology)
        
        # Guardar algunos PNGs para ver qué se entrenará
        if i < 5:
            plt.imshow(geology[0].numpy(), cmap='turbo')
            plt.title(f"Subsuelo Falso {i} para entrenar UNet")
            plt.colorbar()
            plt.savefig(f"{output_dir}/sample_{i}.png")
            plt.close()
            
    # Guardamos todo el dataset puro de entrenamiento de la UNet en un solo Tensor de PyTorch!
    dataset_tensor = torch.stack(images)
    torch.save(dataset_tensor, f"{output_dir}/geology_dataset.pt")
    print(f"¡Dataset creado con éxito! ({dataset_tensor.shape}) Guardado en {output_dir}/geology_dataset.pt")

if __name__ == "__main__":
    master_generator()
