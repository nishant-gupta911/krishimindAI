# KrishiMind AI

AI Crop Planning & Resource Optimization Engine for Indian Agriculture

## Overview

KrishiMind AI is a decision engine that provides district-level crop recommendations using multi-model machine learning. It predicts:
- Crop yield
- Crop price
- Expected revenue
- Climate risk scores

## Features

- **Multi-Model Predictions**: Combines weather, soil, mandi prices, and crop calendars
- **District-Level Recommendations**: Top-3 ranked crop plans per district and season
- **Climate Scenario Simulation**: Rainfall drop and temperature rise impact analysis
- **Revenue Optimization**: Expected revenue per hectare calculations
- **API & Dashboard**: RESTful API with web interface

## Tech Stack

- Python 3.9+
- FastAPI
- Scikit-learn, XGBoost, LightGBM
- Pandas, NumPy
- AWS (S3, SageMaker, Lambda, API Gateway)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m src.api.main

# Train models
python -m src.models.train

# Generate recommendations
python -m src.engine.recommender --district "Pune" --season "Kharif"
```

## Project Structure

```
krishimind-ai/
├── src/
│   ├── data/           # Data ingestion and preprocessing
│   ├── models/         # ML models (yield, price, risk)
│   ├── engine/         # Recommendation and ranking engine
│   ├── api/            # FastAPI endpoints
│   └── utils/          # Helpers and config
├── data/               # Raw and processed data
├── models/             # Trained model artifacts
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit tests
└── deployment/         # AWS deployment configs
```

## API Endpoints

- `POST /api/v1/recommend` - Get crop recommendations
- `POST /api/v1/predict/yield` - Predict crop yield
- `POST /api/v1/predict/price` - Predict crop price
- `POST /api/v1/simulate` - Run climate scenario simulation

## License

MIT
