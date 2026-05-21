import os
from dotenv import load_dotenv

load_dotenv()

# Path ke file model .keras
MODEL_PATH: str = os.getenv("MODEL_PATH", "model/model_terbaik.keras")

# Port FastAPI berjalan
PORT: int = int(os.getenv("PORT", 8001))

# Host
HOST: str = os.getenv("HOST", "0.0.0.0")

# Tipe file yang diizinkan
ALLOWED_CONTENT_TYPES: set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}

# Ukuran maksimal file upload (5MB)
MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024