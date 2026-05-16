import os
from pathlib import Path

# Base directory for configuration files
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_DIR = os.path.join(BASE_DIR, 'users') 

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(USERS_DIR, exist_ok=True)

# Memory Configuration
MAX_VECTOR_RESULTS = 3
MEMORY_CATEGORIES =[
    "personal",
    "profecional",
    "preferencias",
    "hechos_importantes"
]

CATEGORY_DESCRIPTION = "Categoria: personal, professional, preferencias, hechos_importantes"
CONTENT_DESCRIPTION = "Contenido de la memoria"
IMPORTANCE_DESCRIPTION = "Importancia del 1 al 5"