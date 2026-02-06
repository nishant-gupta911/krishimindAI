import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# AWS Config
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET = os.getenv("S3_BUCKET", "krishimind-data")

# Model Config
MODELS = {
    "yield": "yield_model.joblib",
    "price": "price_model.joblib",
    "risk": "risk_model.joblib"
}

# Crop Config
SEASONS = ["Kharif", "Rabi", "Zaid"]
MAJOR_CROPS = ["Rice", "Wheat", "Cotton", "Sugarcane", "Maize", "Pulses", "Soybean"]

# API Config
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
