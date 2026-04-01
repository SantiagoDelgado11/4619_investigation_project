import numpy as np
import pandas as pd
import os

def create_synthetic_data(num_points=500, output_path="data/synthetic/ideal_data.csv"):
    """
    Script intermedio. Aquí podrías llamar al solver FEM 
    (ej: pygimli, fipy) para generar medidas ideales teóricas V
    de ciertas inyecciones. Por ahora lo llenaremos de dummy data.
    """
    print("Generando posiciones de arreglos ERT sintéticos...")
    data = {
        'A_x': np.random.uniform(0, 100, num_points),
        'B_x': np.random.uniform(0, 100, num_points),
        'M_x': np.random.uniform(0, 100, num_points),
        'N_x': np.random.uniform(0, 100, num_points),
        'V_meas': np.random.uniform(0.1, 5.0, num_points)
    }
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Set de datos sintéticos guardado en {output_path}")

if __name__ == "__main__":
    create_synthetic_data()
