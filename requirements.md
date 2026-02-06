# KrishiMind AI - Requirements Document

## 1. Problem Statement

Indian farmers face significant challenges in crop planning due to:
- Unpredictable weather patterns and climate variability
- Limited access to data-driven decision support systems
- Uncertainty in crop yield and market price forecasting
- Inadequate tools for evaluating climate risk scenarios
- Difficulty in optimizing revenue per hectare across seasons

Current solutions focus primarily on yield prediction in isolation, without integrating price forecasting, soil suitability, and climate risk into a unified decision framework.

## 2. Solution Overview

KrishiMind AI is an AI-powered crop planning and resource optimization engine that provides district-level crop recommendations for Indian agriculture. The system integrates multiple data sources and machine learning models to rank crops based on predicted yield, expected market prices, revenue potential, soil suitability, and climate risk.

**Core Capability**: Decision engine that produces top-3 ranked crop recommendations per district and season, with scenario simulation for climate shock analysis.

**Scope**: District-level aggregation only. Not field-level precision agriculture.

## 3. Objectives

### Primary Objectives
- Provide actionable crop recommendations ranked by expected revenue and risk
- Predict crop yield at district level using climate and soil features
- Forecast mandi prices based on historical patterns
- Calculate expected revenue per hectare for crop-district-season combinations
- Assess climate risk scores for each recommendation
- Simulate impact of climate shocks (rainfall deficit, temperature rise)

### Secondary Objectives
- Expose predictions via REST API for integration with external systems
- Support AWS-deployable architecture for scalability
- Maintain inference latency under 2 seconds per request
- Provide explainable outputs with component scores

## 4. Functional Requirements

### FR-1: Crop Recommendation Engine
- **FR-1.1**: Accept inputs: district name, season (Kharif/Rabi/Zaid), year, area (optional)
- **FR-1.2**: Return top-3 ranked crops with scores
- **FR-1.3**: Each recommendation must include:
  - Crop name
  - Expected yield (quintals per hectare)
  - Expected price (INR per quintal)
  - Expected revenue (INR per hectare)
  - Risk score (0-100 scale)
  - Composite ranking score
- **FR-1.4**: Ranking based on weighted composite score: revenue adjusted by risk factor

### FR-2: Yield Prediction
- **FR-2.1**: Predict crop yield for given district-crop-season-year combination
- **FR-2.2**: Use ensemble model (Random Forest + Gradient Boosting + XGBoost)
- **FR-2.3**: Input features: rainfall, temperature, soil nutrients, historical yield, season
- **FR-2.4**: Output: yield in quintals per hectare

### FR-3: Price Prediction
- **FR-3.1**: Predict expected mandi price for crop at harvest time
- **FR-3.2**: Use historical mandi price patterns and seasonal trends
- **FR-3.3**: Model: LightGBM regression
- **FR-3.4**: Output: price in INR per quintal

### FR-4: Risk Scoring
- **FR-4.1**: Calculate climate risk score (0-100) for crop-district combination
- **FR-4.2**: Risk factors: rainfall variability, temperature extremes, drought history
- **FR-4.3**: Model: Random Forest classifier
- **FR-4.4**: Higher score indicates higher risk

### FR-5: Scenario Simulation
- **FR-5.1**: Simulate rainfall deficit scenario (percentage drop)
- **FR-5.2**: Simulate temperature rise scenario (degrees Celsius increase)
- **FR-5.3**: Return baseline vs shocked recommendations
- **FR-5.4**: Show impact on yield, revenue, and risk scores
- **FR-5.5**: Simulation is model-based adjustment, not physical crop simulation

### FR-6: API Interface
- **FR-6.1**: REST API with JSON request/response
- **FR-6.2**: Endpoint: `POST /api/v1/recommend` - get crop recommendations
- **FR-6.3**: Endpoint: `POST /api/v1/simulate` - run climate scenario
- **FR-6.4**: Request validation using Pydantic schemas
- **FR-6.5**: Error responses with appropriate HTTP status codes

## 5. Non-Functional Requirements

### NFR-1: Performance
- **NFR-1.1**: API response time < 2 seconds for recommendation requests
- **NFR-1.2**: API response time < 3 seconds for simulation requests
- **NFR-1.3**: Support concurrent requests (target: 10 concurrent users)

### NFR-2: Scalability
- **NFR-2.1**: Stateless inference design for horizontal scaling
- **NFR-2.2**: AWS Lambda compatible for serverless deployment
- **NFR-2.3**: Model artifacts stored in S3 for distributed access

### NFR-3: Reliability
- **NFR-3.1**: API availability target: 95% uptime (hackathon scope)
- **NFR-3.2**: Graceful error handling with informative messages
- **NFR-3.3**: Input validation to prevent invalid predictions

### NFR-4: Maintainability
- **NFR-4.1**: Modular architecture with separation of concerns
- **NFR-4.2**: Configuration externalized via environment variables
- **NFR-4.3**: Logging for debugging and monitoring

### NFR-5: Deployability
- **NFR-5.1**: Local-first runnable (development and testing)
- **NFR-5.2**: AWS deployable architecture (S3, Lambda, API Gateway, CloudWatch)
- **NFR-5.3**: Containerization support (Docker) for consistent environments

## 6. Data Requirements

### DR-1: Weather Data
- **Source**: IMD (India Meteorological Department) rainfall and temperature datasets
- **Granularity**: District-level monthly aggregates
- **Features**: Total rainfall (mm), average temperature (°C), min/max temperature
- **Temporal Coverage**: Historical data for training (minimum 5 years recommended)
- **Limitation**: District-level aggregation only; no sub-district precision

### DR-2: Soil Data
- **Source**: Soil nutrient datasets (district-level)
- **Features**: Nitrogen (N), Phosphorus (P), Potassium (K), pH, organic carbon
- **Granularity**: District-level averages
- **Limitation**: Static soil profiles; does not account for field-level variation

### DR-3: Crop Yield Data
- **Source**: Historical crop yield datasets (state agriculture departments, DACNET)
- **Features**: Crop type, district, season, year, yield (quintals/hectare)
- **Temporal Coverage**: Multi-year historical records
- **Limitation**: Reported yields may have data quality issues

### DR-4: Mandi Price Data
- **Source**: Mandi price datasets (AGMARKNET or equivalent)
- **Features**: Crop, market location, date, modal price (INR/quintal)
- **Temporal Coverage**: Historical price time series
- **Limitation**: Sparse coverage for some crops/districts; price volatility

### DR-5: Crop Calendar Data
- **Source**: Crop calendar datasets (sowing and harvesting windows)
- **Features**: Crop, region, season, sowing period, harvesting period
- **Usage**: Validate season-crop compatibility

### DR-6: Data Quality Constraints
- **DR-6.1**: Missing data handling: imputation or exclusion based on threshold
- **DR-6.2**: Outlier detection and treatment in training data
- **DR-6.3**: Data versioning for reproducibility
- **DR-6.4**: No real-time data ingestion in hackathon scope

## 7. ML Requirements

### MLR-1: Model Training (Pre-Production)
- **MLR-1.1**: Yield model: Ensemble (Random Forest + Gradient Boosting + XGBoost)
- **MLR-1.2**: Price model: LightGBM regression
- **MLR-1.3**: Risk model: Random Forest classifier
- **MLR-1.4**: Training pipeline separate from inference API
- **MLR-1.5**: Model artifacts serialized (joblib format)
- **MLR-1.6**: Cross-validation for model evaluation
- **MLR-1.7**: Feature importance analysis for interpretability

### MLR-2: Model Inference (Production)
- **MLR-2.1**: Load pre-trained models at API startup
- **MLR-2.2**: Stateless inference (no model updates during runtime)
- **MLR-2.3**: Feature preprocessing consistent with training pipeline
- **MLR-2.4**: Ensemble prediction: weighted average of base models

### MLR-3: Model Limitations
- **MLR-3.1**: Models trained on historical data; may not capture unprecedented events
- **MLR-3.2**: District-level predictions; not applicable to individual farms
- **MLR-3.3**: Price predictions subject to market volatility and external shocks
- **MLR-3.4**: Risk scores are relative indicators, not absolute probabilities
- **MLR-3.5**: Scenario simulations use simplified adjustment factors, not crop growth models

### MLR-4: Model Evaluation Metrics
- **MLR-4.1**: Yield model: RMSE, MAE, R² score
- **MLR-4.2**: Price model: RMSE, MAPE
- **MLR-4.3**: Risk model: Accuracy, Precision, Recall, F1-score
- **MLR-4.4**: Recommendation quality: Revenue ranking correlation with actual outcomes (if validation data available)

## 8. API Requirements

### API-1: Request Schema
```json
{
  "district": "string (required)",
  "season": "string (required, enum: Kharif/Rabi/Zaid)",
  "year": "integer (required)",
  "top_n": "integer (optional, default: 3)"
}
```

### API-2: Response Schema
```json
{
  "district": "string",
  "season": "string",
  "year": "integer",
  "recommendations": [
    {
      "crop": "string",
      "expected_yield_qtl_per_ha": "float",
      "expected_price_per_qtl": "float",
      "expected_revenue_per_ha": "float",
      "risk_score": "float",
      "composite_score": "float"
    }
  ]
}
```

### API-3: Simulation Request Schema
```json
{
  "district": "string (required)",
  "season": "string (required)",
  "year": "integer (required)",
  "scenario_type": "string (required, enum: rainfall_drop/temperature_rise)",
  "magnitude": "float (required)"
}
```

### API-4: Error Handling
- **API-4.1**: 400 Bad Request: Invalid input parameters
- **API-4.2**: 404 Not Found: District or crop not in training data
- **API-4.3**: 500 Internal Server Error: Model inference failure
- **API-4.4**: Error response includes descriptive message

## 9. Risk & Limitation Disclosure

### Limitations
1. **Geographic Granularity**: District-level only; not suitable for field-level decisions
2. **Data Dependency**: Predictions quality depends on historical data availability and accuracy
3. **Price Volatility**: Mandi price predictions may not capture sudden market shocks
4. **Climate Extremes**: Models trained on historical data may underperform during unprecedented climate events
5. **Soil Variability**: District-level soil data does not capture within-district heterogeneity
6. **Scenario Simplification**: Climate shock simulations use adjustment factors, not mechanistic crop models
7. **No Real-Time Updates**: Inference uses pre-trained models; no online learning

### Risks
1. **Model Drift**: Performance degradation if climate patterns shift significantly
2. **Data Quality**: Inaccurate input data leads to unreliable predictions
3. **Overfitting**: Models may overfit to historical patterns not representative of future
4. **Bias**: Training data biases (e.g., under-representation of certain crops/districts) propagate to predictions

### Mitigation Strategies
1. Regular model retraining with updated data (post-hackathon)
2. Input validation and sanity checks
3. Confidence intervals or uncertainty quantification (future enhancement)
4. User education on appropriate use cases and limitations

## 10. Success Metrics

### Hackathon Evaluation Criteria
1. **Functionality**: All API endpoints operational and return valid predictions
2. **Technical Soundness**: ML pipeline follows best practices; architecture is scalable
3. **Documentation**: Clear requirements, design, and deployment instructions
4. **Deployability**: Successfully deployable on AWS or runnable locally
5. **Innovation**: Multi-factor decision engine beyond simple yield prediction

### Performance Metrics
1. **Model Accuracy**: Yield RMSE < 20% of mean yield (validation set)
2. **API Latency**: 95th percentile response time < 2 seconds
3. **Recommendation Quality**: Top-3 crops include at least one high-revenue option in 80% of test cases

### User Value Metrics (Post-Hackathon)
1. Adoption rate by agricultural extension services
2. User satisfaction with recommendation quality
3. Revenue improvement for farmers following recommendations (long-term study)

## 11. Hackathon Scope Boundaries

### In Scope
- Pre-trained ML models for yield, price, and risk
- Inference API with recommendation and simulation endpoints
- District-level predictions for major crops (Rice, Wheat, Cotton, Sugarcane, Maize, Pulses, Soybean)
- AWS deployment architecture design
- Local development and testing environment
- Documentation (requirements, design, API reference)

### Out of Scope
- Real-time data ingestion pipelines
- Field-level precision agriculture
- Mobile application or web dashboard UI
- Model retraining infrastructure
- Multi-language support (API in English)
- Integration with government databases
- Payment or subscription system
- Farmer feedback loop and recommendation refinement
- Crop insurance integration
- Pest and disease prediction

### Future Enhancements
- Expand to more crops and districts
- Incorporate satellite imagery for field-level analysis
- Add pest and disease risk models
- Integrate with IoT sensors for real-time soil monitoring
- Develop farmer-facing mobile application
- Implement online learning for model updates
- Add explainability features (SHAP values, feature contributions)
- Multi-objective optimization (revenue, sustainability, water usage)

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Hackathon Submission
