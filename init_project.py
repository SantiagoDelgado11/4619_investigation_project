import os

project_dir = "ERT_PINN_Project"

structure = {
    "data/raw": None,
    "data/processed": None,
    "data/synthetic": None,
    "configs/exp_01_synthetic.yaml": "# Configuración para probar con datos ideales\n",
    "configs/exp_02_noise.yaml": "# Configuración para prueba de estrés (ruido)\n",
    "configs/exp_03_real.yaml": "# Configuración para el archivo crudo de campo\n",
    "src/__init__.py": "",
    "src/data_loader.py": '"""Ingesta, limpieza y formato de los 4 electrodos."""\n',
    "src/models.py": '"""Definición de las redes neuronales (MLPs, Softplus)."""\n',
    "src/physics.py": '"""Motor matemático (Ecuación de Poisson, autograd, TV)."""\n',
    "src/trainer.py": '"""Ciclo de entrenamiento, cálculo del Loss y Scheduler."""\n',
    "src/utils.py": '"""Visualización (Matplotlib) y guardado de modelos."""\n',
    "scripts/generate_data.py": '"""Forward model para crear datos ideales (calibración)."""\n',
    "scripts/train.py": '"""Script principal para lanzar la inversión tomográfica."""\n',
    "scripts/evaluate.py": '"""Carga pesos entrenados para inferencia y análisis post-mortem."""\n',
    "tests/__init__.py": "",
    "tests/test_physics.py": '"""Pruebas unitarias para tus derivadas espaciales."""\n',
    ".gitignore": "data/\n__pycache__/\n*.pyc\nlogs/\nwandb/\n",
    "requirements.txt": "torch\nnumpy\npandas\nmatplotlib\nwandb\n",
    "README.md": "# ERT_PINN_Project\n\nDocumentación rigurosa de tu metodología y comandos.\n"
}

for path, content in structure.items():
    full_path = os.path.join(project_dir, path)
    if content is None:
        os.makedirs(full_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

print("Project structure created successfully!")
