# KrishiMind AI

**AI Crop Planning & Resource Optimization Engine for Indian Agriculture**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

KrishiMind AI is a decision engine that provides district-level crop recommendations for Indian farmers using multi-model machine learning. Unlike simple yield predictors, it integrates:

- **Crop Yield Prediction** (Ensemble: Random Forest + Gradient Boosting + XGBoost)
- **Mandi Price Forecasting** (LightGBM)
- **Climate Risk Scoring** (Random Forest)
- **Revenue Optimization** (Multi-factor ranking)
- **Climate Scenario Simulation** (Rainfall deficit & temperature rise)

## Key Features

✅ **Top-3 Crop Recommendations** per district and season (Kharif/Rabi/Zaid)  
✅ **Expected Revenue per Hectare** with risk-adjusted scoring  
✅ **Climate Shock Simulation** for resilience planning  
✅ **REST API** with FastAPI and Pydantic validation  
✅ **AWS Deployable** (S3, Lambda, API Gateway, CloudWatch)  
✅ **Local-First Runnable** for development and testing

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│                   (FastAPI + Pydantic)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Decision Engine                           │
│              (Crop Ranking & Scoring)                       │
└────┬───────────────┬───────────────┬────────────────────────┘
     │               │               │
┌────▼─────┐  ┌─────▼──────┐  ┌────▼──────┐
│  Yield   │  │   Price    │  │   Risk    │
│  Model   │  │   Model    │  │   Model   │
│(Ensemble)│  │ (LightGBM) │  │    (RF)   │
└──────────┘  └────────────┘  └───────────┘
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/nishant-gupta911/krishimindAI.git
cd krishimindAI

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running Locally

```bash
# Start the API server
python -m src.api.main

# API will be available at:
# http://localhost:8000

# Interactive API docs:
# http://localhost:8000/docs
```

## API Usage

### Get Crop Recommendations

```bash
curl -X POST "http://localhost:8000/api/v1/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Pune",
    "season": "Kharif",
    "year": 2024,
    "top_n": 3
  }'
```

**Response:**
```json
{
  "district": "Pune",
  "season": "Kharif",
  "year": 2024,
  "recommendations": [
    {
      "crop": "Sugarcane",
      "expected_yield_qtl_per_ha": 800.5,
      "expected_price_per_qtl": 350.0,
      "expected_revenue_per_ha": 280175.0,
      "risk_score": 18.5,
      "composite_score": 228342.6
    },
    {
      "crop": "Cotton",
      "expected_yield_qtl_per_ha": 25.3,
      "expected_price_per_qtl": 6500.0,
      "expected_revenue_per_ha": 164450.0,
      "risk_score": 32.0,
      "composite_score": 111826.0
    }
  ]
}
```

### Run Climate Scenario Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "district": "Pune",
    "season": "Kharif",
    "year": 2024,
    "scenario_type": "rainfall_drop",
    "magnitude": 20.0
  }'
```

## Project Structure

```
krishimindAI/
├── src/
│   ├── api/              # FastAPI application
│   │   └── main.py       # API endpoints
│   ├── models/           # ML models
│   │   ├── yield_model.py
│   │   ├── price_model.py
│   │   ├── risk_model.py
│   │   └── train.py      # Training pipeline
│   ├── engine/           # Decision engine
│   │   ├── recommender.py
│   │   └── simulator.py
│   ├── data/             # Data ingestion
│   │   └── ingestion.py
│   └── config.py         # Configuration
├── deployment/           # AWS deployment
│   ├── lambda_handler.py
│   ├── cloudformation.yaml
│   └── requirements-lambda.txt
├── data/                 # Data storage
│   └── raw/
├── models/               # Trained model artifacts
├── requirements.md       # Requirements document
├── design.md            # Design document
└── README.md            # This file
```

## Data Sources

The system integrates multiple data sources:

- **Weather Data**: IMD (India Meteorological Department) rainfall and temperature datasets
- **Soil Data**: District-level soil nutrient profiles (N, P, K, pH)
- **Yield History**: Historical crop yield datasets from state agriculture departments
- **Mandi Prices**: Historical market price data from AGMARKNET
- **Crop Calendars**: Sowing and harvesting window datasets

**Note**: All predictions are at district-level aggregation. Field-level precision is not supported.

## ML Models

### Yield Prediction Model
- **Architecture**: Ensemble (Random Forest + Gradient Boosting + XGBoost)
- **Weights**: RF (0.3) + GBM (0.3) + XGBoost (0.4)
- **Features**: Climate, soil, temporal, historical yield

### Price Prediction Model
- **Architecture**: LightGBM Regressor
- **Features**: Historical prices, seasonal patterns, market trends

### Risk Scoring Model
- **Architecture**: Random Forest Classifier
- **Output**: Risk score (0-100 scale)
- **Factors**: Rainfall variability, temperature extremes, drought history

## Crop Scoring Formula

```
Expected Revenue = Predicted Yield × Predicted Price
Composite Score = Expected Revenue × (1 - Risk Score/100)
```

Crops are ranked by composite score to balance revenue potential with risk.

## AWS Deployment

### Architecture

- **S3**: Model artifacts and data storage
- **Lambda**: Serverless inference function
- **API Gateway**: HTTP API endpoint
- **CloudWatch**: Logging and monitoring

### Deploy to AWS

```bash
# Upload models to S3
aws s3 cp models/ s3://krishimind-data/models/ --recursive

# Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file deployment/cloudformation.yaml \
  --stack-name krishimind-ai \
  --capabilities CAPABILITY_IAM

# Package and deploy Lambda
pip install -r deployment/requirements-lambda.txt -t package/
cp -r src/ package/
cd package && zip -r ../lambda_function.zip .
aws lambda update-function-code \
  --function-name krishimind-inference \
  --zip-file fileb://lambda_function.zip
```

## Limitations

**Geographic Scope**:
- District-level predictions only (not field-level)
- Limited to districts with sufficient training data

**Model Constraints**:
- Based on historical patterns (may not capture unprecedented events)
- Price predictions subject to market volatility
- Scenario simulations use simplified adjustment factors

**Data Quality**:
- Depends on accuracy of input datasets
- Sparse mandi coverage for some crops
- Static soil profiles (no real-time updates)

See [requirements.md](requirements.md) and [design.md](design.md) for detailed documentation.

## Documentation

- **[Requirements Document](requirements.md)**: Comprehensive requirements specification
- **[Design Document](design.md)**: Technical architecture and design details
- **API Documentation**: Available at `/docs` endpoint when running locally

## Development

### Training Models

```bash
# Train all models (yield, price, risk)
python -m src.models.train
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

## Contributing

This is a hackathon project. Contributions, issues, and feature requests are welcome!

## License

MIT License - see LICENSE file for details

## Authors

- **Nishant Gupta** - [GitHub](https://github.com/nishant-gupta911)

## Acknowledgments

- India Meteorological Department (IMD) for weather data
- AGMARKNET for mandi price data
- State agriculture departments for yield datasets

---

**Built for**: Hackathon Submission  
**Tech Stack**: Python, FastAPI, Scikit-learn, XGBoost, LightGBM, AWS  
**Status**: Production-ready inference API with pre-trained models
