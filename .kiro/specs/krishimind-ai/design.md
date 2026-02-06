# KrishiMind AI - Design Document

## 1. System Architecture

### 1.1 High-Level Architecture

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
└────┬─────┘  └─────┬──────┘  └────┬──────┘
     │               │               │
┌────▼───────────────▼───────────────▼────────────────────────┐
│              Feature Engineering Layer                      │
│         (Data Loading + Preprocessing)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Data Sources                              │
│  Weather │ Soil │ Yield History │ Mandi Prices │ Calendar  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Overview

**API Layer**: FastAPI application exposing REST endpoints for recommendations and simulations

**Decision Engine**: Multi-factor crop ranking system combining yield, price, and risk predictions

**ML Model Layer**: Three independent models for yield, price, and risk prediction

**Feature Engineering**: Data loading, preprocessing, and feature construction

**Data Sources**: Historical datasets (weather, soil, yield, prices, crop calendars)


## 2. Data Pipeline

### 2.1 Data Ingestion

**DataIngestion Class** (`src/data/ingestion.py`)

Responsibilities:
- Load weather data (rainfall, temperature) for district and year
- Load soil nutrient data (N, P, K, pH) for district
- Load historical mandi prices for crop and year
- Load crop calendar (sowing/harvesting windows)
- Combine data sources into feature matrix

**Data Loading Strategy**:
- Local file system for development (CSV/Parquet files in `data/raw/`)
- S3 bucket for production deployment
- Lazy loading: data loaded on-demand per request
- No caching in hackathon scope (stateless design)

### 2.2 Data Schema

**Weather Features**:
- `district`: string
- `year`: integer
- `month`: integer
- `rainfall_mm`: float
- `avg_temp_c`: float
- `min_temp_c`: float
- `max_temp_c`: float

**Soil Features**:
- `district`: string
- `nitrogen_kg_per_ha`: float
- `phosphorus_kg_per_ha`: float
- `potassium_kg_per_ha`: float
- `ph`: float
- `organic_carbon_percent`: float

**Yield History**:
- `district`: string
- `crop`: string
- `season`: string (Kharif/Rabi/Zaid)
- `year`: integer
- `yield_qtl_per_ha`: float

**Mandi Prices**:
- `crop`: string
- `market`: string
- `date`: date
- `modal_price_per_qtl`: float

### 2.3 Data Quality Handling

- **Missing Values**: Impute with district-level median or exclude if >30% missing
- **Outliers**: Cap at 1st and 99th percentiles
- **Validation**: Schema validation using Pydantic models
- **Logging**: Log data loading errors for debugging


## 3. Feature Engineering Layer

### 3.1 Feature Construction

**Input Features for ML Models**:

1. **Climate Features**:
   - Total seasonal rainfall (sum of monthly rainfall during crop season)
   - Average seasonal temperature
   - Temperature variability (std dev)
   - Rainfall variability (coefficient of variation)
   - Growing degree days (GDD) - accumulated temperature above base threshold

2. **Soil Features**:
   - N, P, K levels (normalized)
   - pH (normalized)
   - Organic carbon percentage
   - Soil suitability index (crop-specific weighted combination)

3. **Temporal Features**:
   - Year (for trend capture)
   - Season (one-hot encoded: Kharif, Rabi, Zaid)
   - Month of sowing (derived from crop calendar)

4. **Historical Features**:
   - Previous year yield (lag feature)
   - 3-year moving average yield
   - Price trend (6-month moving average)

5. **Categorical Features**:
   - District (label encoded or one-hot encoded)
   - Crop (label encoded or one-hot encoded)

### 3.2 Feature Preprocessing

- **Normalization**: StandardScaler for continuous features
- **Encoding**: LabelEncoder for categorical features (district, crop)
- **Feature Selection**: Top features based on importance from training phase
- **Feature Store**: Preprocessed features cached per request (in-memory, request scope)

### 3.3 Feature Engineering Pipeline

```python
def prepare_features(district, crop, season, year):
    # Load raw data
    weather = load_weather_data(district, year)
    soil = load_soil_data(district)
    yield_history = load_yield_history(district, crop, year-1)
    prices = load_mandi_prices(crop, year)
    
    # Aggregate seasonal features
    seasonal_rainfall = aggregate_seasonal_rainfall(weather, season)
    seasonal_temp = aggregate_seasonal_temperature(weather, season)
    
    # Construct feature vector
    features = {
        'district': district,
        'crop': crop,
        'season': season,
        'year': year,
        'seasonal_rainfall': seasonal_rainfall,
        'avg_temp': seasonal_temp,
        'nitrogen': soil['nitrogen'],
        'phosphorus': soil['phosphorus'],
        'potassium': soil['potassium'],
        'ph': soil['ph'],
        'prev_yield': yield_history['yield'],
        'price_trend': calculate_price_trend(prices)
    }
    
    return pd.DataFrame([features])
```


## 4. ML Model Layer

### 4.1 Yield Prediction Model

**Architecture**: Ensemble of three models
- Random Forest Regressor (100 trees)
- Gradient Boosting Regressor (100 estimators)
- XGBoost Regressor (100 estimators)

**Ensemble Strategy**: Weighted average
- RF weight: 0.3
- GBM weight: 0.3
- XGBoost weight: 0.4

**Input Features**: Climate + Soil + Temporal + Historical (15-20 features)

**Output**: Yield in quintals per hectare

**Training**:
- Loss function: Mean Squared Error (MSE)
- Cross-validation: 5-fold
- Hyperparameter tuning: Grid search (pre-training phase)

**Inference**:
```python
def predict_yield(features):
    pred_rf = rf_model.predict(features) * 0.3
    pred_gbm = gbm_model.predict(features) * 0.3
    pred_xgb = xgb_model.predict(features) * 0.4
    return pred_rf + pred_gbm + pred_xgb
```

### 4.2 Price Prediction Model

**Architecture**: LightGBM Regressor

**Input Features**: 
- Crop type
- Historical price trends (6-month, 12-month moving averages)
- Seasonal patterns (month, season)
- Year (for inflation/trend)
- District (market proximity proxy)

**Output**: Expected mandi price in INR per quintal

**Training**:
- Loss function: Mean Absolute Percentage Error (MAPE)
- Cross-validation: Time-series split (no future leakage)
- Hyperparameters: learning_rate=0.05, num_leaves=31, max_depth=7

**Inference**:
```python
def predict_price(features):
    return lgbm_model.predict(features)
```

**Limitations**:
- Price predictions assume normal market conditions
- Cannot predict sudden shocks (policy changes, export bans, etc.)
- Sparse mandi coverage for some crops may reduce accuracy

### 4.3 Risk Scoring Model

**Architecture**: Random Forest Classifier

**Input Features**:
- Rainfall variability (coefficient of variation)
- Temperature extremes (days above/below thresholds)
- Drought history (binary: drought in past 3 years)
- Flood history (binary: flood in past 3 years)
- Crop-district historical failure rate

**Output**: Risk probability (0-1), converted to risk score (0-100)

**Training**:
- Binary classification: High risk (1) vs Low risk (0)
- Threshold: Yield < 70% of district average = High risk
- Class balancing: SMOTE or class weights

**Inference**:
```python
def predict_risk_score(features):
    risk_proba = rf_classifier.predict_proba(features)[:, 1]
    return risk_proba * 100  # Convert to 0-100 scale
```

**Interpretation**:
- 0-30: Low risk
- 31-60: Moderate risk
- 61-100: High risk

### 4.4 Model Persistence

**Serialization**: joblib format
- `models/yield_model.joblib` (contains all three ensemble models)
- `models/price_model.joblib`
- `models/risk_model.joblib`

**Loading Strategy**:
- Models loaded once at API startup (singleton pattern)
- Stored in memory for fast inference
- No model updates during runtime (stateless)

**Model Versioning**:
- Model files tagged with training date
- Version metadata stored in `models/metadata.json`


## 5. Decision Engine Logic

### 5.1 Crop Ranking System

**CropRecommender Class** (`src/engine/recommender.py`)

**Workflow**:
1. For each candidate crop in `MAJOR_CROPS`:
   - Load and prepare features for (district, crop, season, year)
   - Predict yield using YieldPredictor
   - Predict price using PricePredictor
   - Predict risk score using RiskScorer
   - Calculate expected revenue: `revenue = yield × price`
   - Calculate composite score: `score = revenue × (1 - risk_score/100)`
2. Sort crops by composite score (descending)
3. Return top-N crops

**Candidate Crops**: Rice, Wheat, Cotton, Sugarcane, Maize, Pulses, Soybean

### 5.2 Crop Scoring Formula

**Revenue Calculation**:
```
Expected Revenue (INR/ha) = Predicted Yield (qtl/ha) × Predicted Price (INR/qtl)
```

**Composite Score**:
```
Composite Score = Expected Revenue × Risk Adjustment Factor

where:
Risk Adjustment Factor = 1 - (Risk Score / 100)
```

**Example**:
- Crop: Rice
- Predicted Yield: 50 qtl/ha
- Predicted Price: 2000 INR/qtl
- Risk Score: 25
- Expected Revenue: 50 × 2000 = 100,000 INR/ha
- Risk Adjustment: 1 - 0.25 = 0.75
- Composite Score: 100,000 × 0.75 = 75,000

**Rationale**:
- Higher revenue increases score
- Higher risk decreases score
- Linear risk penalty (can be adjusted to non-linear in future)

### 5.3 Recommendation Output

**Response Structure**:
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
    },
    {
      "crop": "Soybean",
      "expected_yield_qtl_per_ha": 18.7,
      "expected_price_per_qtl": 4200.0,
      "expected_revenue_per_ha": 78540.0,
      "risk_score": 28.5,
      "composite_score": 56156.1
    }
  ]
}
```

### 5.4 Filtering and Validation

**Pre-Recommendation Filters**:
- Crop-season compatibility check (using crop calendar)
- District-crop historical presence (exclude crops never grown in district)
- Minimum data availability threshold

**Post-Prediction Validation**:
- Yield sanity check: 0 < yield < 3× district max historical yield
- Price sanity check: 0 < price < 5× historical max price
- Risk score bounds: 0 ≤ risk ≤ 100


## 6. Scenario Simulator Design

### 6.1 Climate Shock Simulation

**ClimateSimulator Class** (`src/engine/simulator.py`)

**Purpose**: Evaluate impact of climate shocks on crop recommendations

**Supported Scenarios**:
1. Rainfall deficit (percentage drop)
2. Temperature rise (degrees Celsius increase)

### 6.2 Rainfall Deficit Simulation

**Input**: 
- District, season, year
- Rainfall drop percentage (e.g., 20% deficit)

**Simulation Logic**:
1. Generate baseline recommendations (normal conditions)
2. Adjust rainfall features: `rainfall_shocked = rainfall_baseline × (1 - drop_percent/100)`
3. Re-run yield prediction with adjusted features
4. Adjust risk score: `risk_shocked = risk_baseline + (drop_percent / 2)`
5. Recalculate revenue and composite scores
6. Return baseline vs shocked recommendations

**Adjustment Rationale**:
- Yield impact: Empirical relationship (1% rainfall drop ≈ 0.5-1% yield drop)
- Risk impact: Drought risk increases proportionally

**Example**:
```
Scenario: 20% rainfall deficit
Baseline Yield: 50 qtl/ha → Shocked Yield: 40 qtl/ha (20% drop)
Baseline Risk: 25 → Shocked Risk: 35 (10 point increase)
```

### 6.3 Temperature Rise Simulation

**Input**:
- District, season, year
- Temperature rise (degrees Celsius, e.g., +2°C)

**Simulation Logic**:
1. Generate baseline recommendations
2. Adjust temperature features: `temp_shocked = temp_baseline + rise_celsius`
3. Re-run yield prediction with adjusted features
4. Adjust risk score: `risk_shocked = risk_baseline + (rise_celsius × 5)`
5. Recalculate revenue and composite scores
6. Return baseline vs shocked recommendations

**Adjustment Rationale**:
- Yield impact: Heat stress reduces yield (crop-dependent, simplified to 5% per °C)
- Risk impact: Heat extremes increase crop failure risk

**Example**:
```
Scenario: +2°C temperature rise
Baseline Yield: 50 qtl/ha → Shocked Yield: 45 qtl/ha (10% drop)
Baseline Risk: 25 → Shocked Risk: 35 (10 point increase)
```

### 6.4 Simulation Response

**Response Structure**:
```json
{
  "scenario": "Rainfall drop 20%",
  "baseline": [
    {"crop": "Rice", "expected_yield_qtl_per_ha": 50.0, "risk_score": 25.0, ...},
    {"crop": "Wheat", "expected_yield_qtl_per_ha": 35.0, "risk_score": 20.0, ...}
  ],
  "shocked": [
    {"crop": "Rice", "expected_yield_qtl_per_ha": 40.0, "risk_score": 35.0, ...},
    {"crop": "Wheat", "expected_yield_qtl_per_ha": 28.0, "risk_score": 30.0, ...}
  ]
}
```

### 6.5 Simulation Limitations

**Important Disclaimers**:
- Simulations use simplified adjustment factors, not mechanistic crop growth models
- Linear assumptions may not hold for extreme shocks (>30% rainfall deficit, >3°C rise)
- Does not account for farmer adaptation strategies
- Does not model pest/disease changes under climate stress
- Intended for comparative analysis, not absolute predictions


## 7. API Layer Design

### 7.1 Technology Stack

- **Framework**: FastAPI (async support, automatic OpenAPI docs)
- **Validation**: Pydantic models for request/response schemas
- **Serialization**: JSON
- **Server**: Uvicorn (ASGI server)

### 7.2 API Endpoints

**Endpoint 1: Health Check**
```
GET /
Response: {"message": "KrishiMind AI API", "status": "active"}
```

**Endpoint 2: Crop Recommendations**
```
POST /api/v1/recommend

Request Body:
{
  "district": "Pune",
  "season": "Kharif",
  "year": 2024,
  "top_n": 3
}

Response:
{
  "district": "Pune",
  "season": "Kharif",
  "year": 2024,
  "recommendations": [...]
}
```

**Endpoint 3: Climate Scenario Simulation**
```
POST /api/v1/simulate

Request Body:
{
  "district": "Pune",
  "season": "Kharif",
  "year": 2024,
  "scenario_type": "rainfall_drop",
  "magnitude": 20.0
}

Response:
{
  "scenario": "Rainfall drop 20%",
  "baseline": [...],
  "shocked": [...]
}
```

### 7.3 Request Validation

**Pydantic Models**:

```python
class RecommendationRequest(BaseModel):
    district: str
    season: str  # Enum: Kharif, Rabi, Zaid
    year: int
    top_n: int = 3
    
    @validator('season')
    def validate_season(cls, v):
        if v not in ['Kharif', 'Rabi', 'Zaid']:
            raise ValueError('Invalid season')
        return v
    
    @validator('year')
    def validate_year(cls, v):
        if v < 2020 or v > 2030:
            raise ValueError('Year out of range')
        return v

class SimulationRequest(BaseModel):
    district: str
    season: str
    year: int
    scenario_type: str  # Enum: rainfall_drop, temperature_rise
    magnitude: float
    
    @validator('scenario_type')
    def validate_scenario(cls, v):
        if v not in ['rainfall_drop', 'temperature_rise']:
            raise ValueError('Invalid scenario type')
        return v
    
    @validator('magnitude')
    def validate_magnitude(cls, v, values):
        scenario = values.get('scenario_type')
        if scenario == 'rainfall_drop' and (v < 0 or v > 100):
            raise ValueError('Rainfall drop must be 0-100%')
        if scenario == 'temperature_rise' and (v < 0 or v > 5):
            raise ValueError('Temperature rise must be 0-5°C')
        return v
```

### 7.4 Error Handling

**Error Response Format**:
```json
{
  "detail": "Error message describing the issue"
}
```

**HTTP Status Codes**:
- 200: Success
- 400: Bad Request (invalid input)
- 404: Not Found (district/crop not in training data)
- 500: Internal Server Error (model inference failure)

**Exception Handling**:
```python
@app.post("/api/v1/recommend")
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommender.rank_crops(...)
        return {"district": ..., "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Data not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")
```

### 7.5 API Documentation

- **Auto-generated**: FastAPI provides interactive docs at `/docs` (Swagger UI)
- **Schema**: OpenAPI 3.0 specification available at `/openapi.json`
- **Examples**: Request/response examples included in Pydantic models


## 8. AWS Deployment Architecture

### 8.1 Deployment Overview

**Architecture Type**: AWS deployable, local-first runnable

**AWS Services**:
- **S3**: Model artifacts and data storage
- **Lambda**: Serverless inference function
- **API Gateway**: HTTP API endpoint
- **CloudWatch**: Logging and monitoring
- **SageMaker**: (Optional) Model training and hosting

### 8.2 Deployment Diagram

```
┌──────────────┐
│   Client     │
│ (Dashboard/  │
│   Mobile)    │
└──────┬───────┘
       │ HTTPS
       ▼
┌──────────────────────┐
│   API Gateway        │
│  (HTTP API)          │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Lambda Function    │
│  (FastAPI + Mangum)  │
│                      │
│  - Load models       │
│  - Run inference     │
│  - Return results    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   S3 Bucket          │
│                      │
│  - Model artifacts   │
│  - Training data     │
│  - Feature data      │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│   CloudWatch Logs    │
│  - API requests      │
│  - Errors            │
│  - Latency metrics   │
└──────────────────────┘
```

### 8.3 Lambda Function Design

**Handler**: `deployment/lambda_handler.py`

```python
from mangum import Mangum
from src.api.main import app

# Wrap FastAPI app for Lambda
handler = Mangum(app)
```

**Configuration**:
- Runtime: Python 3.9
- Memory: 1024 MB (adjust based on model size)
- Timeout: 30 seconds
- Environment Variables:
  - `S3_BUCKET`: Model storage bucket
  - `AWS_REGION`: Deployment region (ap-south-1)
  - `MODEL_PATH`: S3 path to model artifacts

**Cold Start Optimization**:
- Keep models in `/tmp` directory (512 MB available)
- Use Lambda layers for dependencies (scikit-learn, xgboost, etc.)
- Provisioned concurrency for production (optional)

### 8.4 S3 Bucket Structure

```
s3://krishimind-data/
├── models/
│   ├── yield_model.joblib
│   ├── price_model.joblib
│   ├── risk_model.joblib
│   └── metadata.json
├── data/
│   ├── weather/
│   ├── soil/
│   ├── yield_history/
│   └── mandi_prices/
└── logs/
    └── inference_logs/
```

### 8.5 CloudFormation Template

**Infrastructure as Code**: `deployment/cloudformation.yaml`

**Resources Defined**:
- S3 bucket with versioning
- Lambda function with execution role
- API Gateway HTTP API
- CloudWatch log group
- IAM roles and policies

**Deployment Command**:
```bash
aws cloudformation deploy \
  --template-file deployment/cloudformation.yaml \
  --stack-name krishimind-ai \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides S3BucketName=krishimind-data
```

### 8.6 Local Development Setup

**Run Locally**:
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export S3_BUCKET=krishimind-data
export AWS_REGION=ap-south-1

# Run API server
python -m src.api.main
```

**Access**:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### 8.7 Deployment Workflow

1. **Train Models** (one-time or periodic):
   ```bash
   python -m src.models.train
   ```

2. **Upload Models to S3**:
   ```bash
   aws s3 cp models/ s3://krishimind-data/models/ --recursive
   ```

3. **Package Lambda Function**:
   ```bash
   pip install -r deployment/requirements-lambda.txt -t package/
   cp -r src/ package/
   cd package && zip -r ../lambda_function.zip .
   ```

4. **Deploy CloudFormation Stack**:
   ```bash
   aws cloudformation deploy --template-file deployment/cloudformation.yaml ...
   ```

5. **Update Lambda Code**:
   ```bash
   aws lambda update-function-code \
     --function-name krishimind-inference \
     --zip-file fileb://lambda_function.zip
   ```

### 8.8 Monitoring and Logging

**CloudWatch Metrics**:
- Invocation count
- Error rate
- Duration (latency)
- Throttles

**Custom Metrics** (optional):
- Prediction count by district
- Average composite score
- Model inference time

**Logging**:
- Request/response payloads (sanitized)
- Error stack traces
- Model loading time
- Feature engineering time


## 9. Model Inference Flow

### 9.1 Recommendation Request Flow

```
1. Client Request
   ↓
2. API Gateway → Lambda
   ↓
3. FastAPI Endpoint (/api/v1/recommend)
   ↓
4. Request Validation (Pydantic)
   ↓
5. CropRecommender.rank_crops()
   ↓
6. For each candidate crop:
   ├─→ DataIngestion.prepare_features()
   │   ├─→ Load weather data
   │   ├─→ Load soil data
   │   ├─→ Load yield history
   │   ├─→ Load mandi prices
   │   └─→ Construct feature vector
   │
   ├─→ YieldPredictor.predict()
   │   └─→ Ensemble prediction (RF + GBM + XGBoost)
   │
   ├─→ PricePredictor.predict()
   │   └─→ LightGBM prediction
   │
   ├─→ RiskScorer.predict_risk_score()
   │   └─→ Random Forest classification
   │
   └─→ Calculate revenue and composite score
   ↓
7. Sort crops by composite score
   ↓
8. Return top-N recommendations
   ↓
9. JSON Response → Client
```

### 9.2 Simulation Request Flow

```
1. Client Request (scenario parameters)
   ↓
2. API Gateway → Lambda
   ↓
3. FastAPI Endpoint (/api/v1/simulate)
   ↓
4. Request Validation (Pydantic)
   ↓
5. ClimateSimulator.simulate_*()
   ↓
6. Generate Baseline Recommendations
   ├─→ CropRecommender.rank_crops() [normal conditions]
   ↓
7. Adjust Features for Shock
   ├─→ Modify rainfall or temperature
   ↓
8. Generate Shocked Recommendations
   ├─→ Re-run predictions with adjusted features
   ├─→ Adjust risk scores
   ├─→ Recalculate composite scores
   ↓
9. Return {baseline, shocked} comparison
   ↓
10. JSON Response → Client
```

### 9.3 Model Loading Strategy

**Singleton Pattern**:
```python
class ModelLoader:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_models(self):
        if not self._models_loaded:
            self.yield_model = joblib.load('models/yield_model.joblib')
            self.price_model = joblib.load('models/price_model.joblib')
            self.risk_model = joblib.load('models/risk_model.joblib')
            self._models_loaded = True
```

**Loading Timing**:
- **Local Development**: Load at API startup (once)
- **Lambda**: Load on cold start, reuse across warm invocations
- **Optimization**: Store models in Lambda `/tmp` directory

### 9.4 Performance Optimization

**Caching**:
- No request-level caching in hackathon scope (stateless design)
- Future: Redis cache for frequently requested district-season combinations

**Batch Prediction**:
- Current: Sequential prediction for each crop
- Future: Batch predict all crops simultaneously (vectorized operations)

**Feature Precomputation**:
- Current: Compute features on-demand
- Future: Precompute seasonal aggregates, store in feature store

**Model Optimization**:
- Use quantized models for faster inference (future)
- ONNX runtime for cross-platform optimization (future)


## 10. Risk Controls & Validation

### 10.1 Input Validation

**District Validation**:
- Check against whitelist of supported districts
- Return 404 if district not in training data
- Fuzzy matching for minor spelling variations (future)

**Season Validation**:
- Enum validation: Kharif, Rabi, Zaid only
- Reject invalid season names

**Year Validation**:
- Range check: 2020 ≤ year ≤ 2030
- Warn if year is far from training data range

**Magnitude Validation** (simulations):
- Rainfall drop: 0-100%
- Temperature rise: 0-5°C
- Reject extreme values outside realistic ranges

### 10.2 Output Validation

**Yield Sanity Checks**:
- Minimum: yield > 0
- Maximum: yield < 3× historical max for crop-district
- Flag predictions outside 2 standard deviations from mean

**Price Sanity Checks**:
- Minimum: price > 0
- Maximum: price < 5× historical max for crop
- Warn if price prediction deviates >50% from recent average

**Risk Score Bounds**:
- Enforce: 0 ≤ risk_score ≤ 100
- Clip values outside range

**Revenue Consistency**:
- Verify: revenue = yield × price (within rounding tolerance)

### 10.3 Model Monitoring

**Prediction Logging**:
- Log all predictions with timestamps
- Store: district, crop, season, year, predicted values
- Enable post-hoc analysis and model drift detection

**Error Tracking**:
- Log all API errors with stack traces
- Track error rate by endpoint
- Alert if error rate exceeds threshold (future)

**Latency Monitoring**:
- Measure inference time per request
- Track p50, p95, p99 latencies
- Alert if latency exceeds SLA (future)

### 10.4 Data Quality Checks

**Missing Data Handling**:
- If >30% of features missing for a district-crop: return 404
- Impute missing values with district median (for <30% missing)
- Log data quality issues for investigation

**Outlier Detection**:
- Flag extreme feature values (>3 std dev from mean)
- Log outliers but proceed with prediction
- Consider excluding outliers in future model retraining

**Data Freshness**:
- Check timestamp of data sources
- Warn if data is >2 years old (stale data risk)

### 10.5 Security Considerations

**Input Sanitization**:
- Validate all string inputs (district, crop names)
- Prevent SQL injection (not applicable, no SQL database)
- Prevent path traversal attacks in file loading

**Rate Limiting**:
- Implement rate limiting to prevent abuse (future)
- Current: Rely on API Gateway throttling

**Authentication**:
- No authentication in hackathon scope (public API)
- Future: API key authentication for production

**Data Privacy**:
- No personally identifiable information (PII) collected
- Aggregate district-level data only


## 11. Limitations

### 11.1 Geographic Limitations

**District-Level Only**:
- Predictions aggregated at district level
- Cannot provide field-specific recommendations
- Within-district variability not captured
- Not suitable for precision agriculture applications

**Coverage Limitations**:
- Limited to districts with sufficient training data
- New districts require model retraining
- Border districts may have data quality issues

### 11.2 Model Limitations

**Historical Data Dependency**:
- Models trained on past patterns
- May not generalize to unprecedented climate events
- Performance degrades if climate shifts significantly from training period

**Simplified Assumptions**:
- Linear relationships assumed in some features
- Crop-soil interactions simplified
- Pest and disease impacts not modeled
- Farmer management practices not considered

**Price Prediction Uncertainty**:
- Cannot predict policy shocks (export bans, MSP changes)
- Market volatility not fully captured
- Sparse mandi coverage for some crops
- Global commodity price impacts not modeled

**Risk Scoring Limitations**:
- Risk score is relative, not absolute probability
- Based on historical failure patterns
- Does not account for emerging risks (new pests, diseases)
- Climate change impacts may not be fully reflected

### 11.3 Scenario Simulation Limitations

**Simplified Shock Models**:
- Linear adjustment factors, not mechanistic crop models
- Does not simulate crop phenology changes
- Ignores farmer adaptation strategies
- Pest/disease dynamics under stress not modeled

**Extreme Event Handling**:
- Accuracy decreases for extreme shocks (>30% rainfall deficit, >3°C rise)
- Compound shocks (drought + heat) not explicitly modeled
- Threshold effects (crop failure) not captured

### 11.4 Data Limitations

**Data Quality**:
- Reported yield data may have errors or biases
- Mandi price data has gaps for some markets
- Weather data interpolated for some districts
- Soil data may be outdated (static profiles)

**Temporal Coverage**:
- Limited historical data for some crops/districts
- Recent years may be under-represented
- Seasonal patterns may shift over time

**Feature Completeness**:
- Irrigation availability not modeled
- Fertilizer application not captured
- Crop variety differences not considered
- Farm size and mechanization not included

### 11.5 Operational Limitations

**No Real-Time Updates**:
- Models are static (no online learning)
- Predictions based on pre-trained models
- Requires periodic retraining to stay current

**Inference Only**:
- No model retraining in production API
- Training pipeline separate from inference
- Model updates require redeployment

**Scalability Constraints**:
- Sequential crop evaluation (not parallelized)
- Lambda cold start latency (2-5 seconds)
- Model size limits Lambda deployment options


## 12. Future Extensions

### 12.1 Model Enhancements

**Advanced ML Techniques**:
- Deep learning models (LSTM for time series, CNN for spatial patterns)
- Transfer learning from global crop models
- Uncertainty quantification (prediction intervals)
- Multi-task learning (joint yield-price-risk prediction)

**Feature Expansion**:
- Satellite imagery integration (NDVI, soil moisture)
- IoT sensor data (real-time soil conditions)
- Irrigation availability and water stress indices
- Fertilizer application and soil amendments
- Crop variety characteristics

**Mechanistic Models**:
- Integrate crop growth simulation models (DSSAT, APSIM)
- Physics-based climate impact modeling
- Pest and disease risk models
- Water balance and irrigation optimization

### 12.2 System Enhancements

**Real-Time Data Integration**:
- Live weather data feeds (IMD API)
- Real-time mandi price updates
- Satellite imagery processing pipeline
- IoT sensor data ingestion

**Explainability**:
- SHAP values for feature importance
- Counterfactual explanations ("what-if" analysis)
- Confidence scores for predictions
- Visualization of decision factors

**Optimization**:
- Multi-objective optimization (revenue, sustainability, water usage)
- Crop rotation planning (multi-season optimization)
- Resource allocation (land, water, labor)
- Risk-return frontier analysis

### 12.3 Geographic Expansion

**Finer Granularity**:
- Tehsil-level predictions
- Village-level recommendations (with sufficient data)
- Field-level precision agriculture (with satellite/IoT data)

**Broader Coverage**:
- Expand to all Indian districts
- Include more crops (vegetables, fruits, spices)
- Regional crop varieties and local cultivars

### 12.4 User-Facing Features

**Dashboard Interface**:
- Web-based visualization of recommendations
- Interactive maps (district-level heatmaps)
- Historical trend analysis
- Scenario comparison tools

**Mobile Application**:
- Farmer-facing mobile app (Android/iOS)
- Vernacular language support (Hindi, regional languages)
- Voice-based input and output
- Offline mode for low-connectivity areas

**Feedback Loop**:
- Farmer feedback on recommendation quality
- Actual yield reporting for model validation
- Crowdsourced data collection (pest sightings, crop health)
- Continuous model improvement with user feedback

### 12.5 Integration and Ecosystem

**Government Systems**:
- Integration with state agriculture departments
- Link to crop insurance schemes
- Connection to subsidy and loan programs
- Alignment with national agricultural policies

**Market Linkages**:
- Connect farmers to buyers (contract farming)
- Price negotiation support
- Supply chain optimization
- Market demand forecasting

**Advisory Services**:
- Personalized agronomic advice
- Pest and disease alerts
- Weather-based advisories
- Best practice recommendations

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Hackathon Submission  
**Architecture Review**: Pending
