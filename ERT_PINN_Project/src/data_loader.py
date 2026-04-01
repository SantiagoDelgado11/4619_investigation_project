import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ERTDataset(Dataset):
    """
    Dataset para cargar mediciones crudas de ERT.
    Normalmente, incluyen posiciones de electrodos de corriente (A, B),
    electrodos de potencial (M, N) y un valor medido de voltaje V (o resistividad aparente).
    """
    def __init__(self, data_path, is_synthetic=False):
        super().__init__()
        self.is_synthetic = is_synthetic
        
        # Ejemplo: Cargar CSV de datos ERT
        # Si usas Res2DInv/DAT, necesitarías un parseador ad-hoc, aquí asumimos CSV preprocesado.
        try:
            df = pd.read_csv(data_path)
            self.a_x = df['A_x'].values
            self.b_x = df['B_x'].values
            self.m_x = df['M_x'].values
            self.n_x = df['N_x'].values
            self.v_meas = df['V_meas'].values
        except:
            # Placeholder con datos aleatorios si el archivo no existe aún
            print(f"Warning: No valid data found at {data_path}. Using dummy data.")
            num_points = 100
            self.a_x = np.random.rand(num_points)
            self.b_x = np.random.rand(num_points)
            self.m_x = np.random.rand(num_points)
            self.n_x = np.random.rand(num_points)
            self.v_meas = np.random.rand(num_points)
            
    def __len__(self):
        return len(self.v_meas)
        
    def __getitem__(self, idx):
        # Convertir a tensores
        sample = {
            'A_x': torch.tensor(self.a_x[idx], dtype=torch.float32),
            'B_x': torch.tensor(self.b_x[idx], dtype=torch.float32),
            'M_x': torch.tensor(self.m_x[idx], dtype=torch.float32),
            'N_x': torch.tensor(self.n_x[idx], dtype=torch.float32),
            'V_meas': torch.tensor(self.v_meas[idx], dtype=torch.float32)
        }
        return sample
