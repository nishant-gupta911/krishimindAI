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

