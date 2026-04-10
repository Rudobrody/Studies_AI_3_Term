from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
DATABASE_FILE_PATH = PROCESSED_DATA_DIR / "embeddings.pkl"

MODELS_DIR = PROJ_ROOT / "models"
FACE_MODEL_PATH = MODELS_DIR / "mobilefacenet.onnx"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Global parameters
SIMILARITY_THRESHOLD = 0.7