"""AWS Lambda handler for serverless deployment"""
import json
from mangum import Mangum
from src.api.main import app

# Wrap FastAPI app for Lambda
handler = Mangum(app)
