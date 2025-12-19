# Technical Documentation: Player Churn Prediction System

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Model Training](#model-training)
6. [Deployment](#deployment)
7. [Monitoring](#monitoring)
8. [API Reference](#api-reference)

## System Overview

The Player Churn Prediction System is an end-to-end machine learning pipeline designed to identify players at risk of churning and enable proactive retention interventions.

### Key Components
- **Data Layer**: PostgreSQL + Redis for structured and real-time data
- **ETL Pipeline**: Apache Airflow for orchestration
- **ML Pipeline**: Ensemble of XGBoost, Random Forest, and LightGBM
- **API Layer**: FastAPI for real-time predictions
- **Dashboard**: Streamlit for visualization and analysis
- **A/B Testing**: Framework for measuring intervention effectiveness

### Technology Stack
```
Data Storage:    PostgreSQL, Redis
Processing:      Python, Pandas, NumPy
ML Frameworks:   XGBoost, LightGBM, scikit-learn
Orchestration:   Apache Airflow
API:             FastAPI, Uvicorn
Dashboard:       Streamlit, Plotly
Containerization: Docker, Docker Compose
```

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Sources                          │
│           (Steam API / Synthetic Data Generator)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                       │
│              (Python Scripts + Airflow DAG)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     ETL Pipeline                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │ Extract  │───▶│Transform │───▶│   Load   │            │
│   └──────────┘    └──────────┘    └──────────┘            │
│                                                              │
│   Feature Engineering + Data Quality Checks                 │
└──────────────┬──────────────────────┬────────────────────────┘
               │                      │
               ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐
    │   PostgreSQL     │   │      Redis       │
    │ (Historical Data)│   │ (Real-time Cache)│
    └────────┬─────────┘   └─────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML Training Pipeline                      │
│                                                              │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│   │ XGBoost  │  │ Random   │  │ LightGBM │                │
│   │          │  │ Forest   │  │          │                │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│        └────────────┬─────────────┘                        │
│                     ▼                                        │
│             ┌──────────────┐                                │
│             │   Ensemble   │                                │
│             │    Model     │                                │
│             └──────┬───────┘                                │
└────────────────────┼────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Serving (FastAPI)                    │
│                                                              │
│   POST /predict       GET /health       GET /models         │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐
    │   Streamlit      │   │  External        │
    │   Dashboard      │   │  Applications    │
    └──────────────────┘   └──────────────────┘
```

### Component Interactions

```python
# Daily workflow
1. Airflow triggers data ingestion (2 AM daily)
2. ETL pipeline processes new player data
3. Features stored in PostgreSQL + Redis
4. Batch predictions generated for all players
5. Dashboard updates with latest risk scores
6. Monitoring checks for model drift
```

## Data Pipeline

### 1. Data Ingestion

#### Synthetic Data Generation
```python
from data.synthetic_data_generator import PlayerDataGenerator

generator = PlayerDataGenerator(n_players=100000, n_months=6)
df = generator.generate_complete_dataset()
generator.save_dataset(df, 'data/raw/player_data.csv')
```

#### Steam API Integration (Optional)
```python
from src.data_ingestion.steam_api_connector import SteamAPIConnector

connector = SteamAPIConnector(api_key='YOUR_KEY')
df = connector.batch_extract_features(steam_ids, app_id=730)
```

### 2. ETL Pipeline

#### Configuration
```python
config = {
    'data_source': 'synthetic',
    'use_postgres': True,
    'use_redis': True,
    'postgres_host': 'localhost',
    'postgres_port': 5432,
    'postgres_db': 'churn_db',
    'redis_host': 'localhost',
    'redis_port': 6379,
    'output_csv': 'data/processed/player_features.csv'
}
```

#### Running ETL
```python
from src.etl.etl_pipeline import ETLPipeline

pipeline = ETLPipeline(config)
df = pipeline.run_pipeline(file_path='data/raw/player_data.csv')
```

### 3. Data Schema

#### Player Features Table
```sql
CREATE TABLE player_features (
    player_id VARCHAR(50),
    date TIMESTAMP,
    -- Behavioral features
    sessions_per_week INTEGER,
    avg_session_length_mins REAL,
    days_since_last_login INTEGER,
    session_decay_rate REAL,
    activity_consistency REAL,
    engagement_score REAL,
    -- Performance features
    win_rate REAL,
    kd_ratio REAL,
    current_rank INTEGER,
    rank_change INTEGER,
    performance_score REAL,
    -- Social features
    friends_online INTEGER,
    party_play_percentage REAL,
    social_engagement REAL,
    -- Monetization features
    monthly_spent REAL,
    days_since_last_purchase INTEGER,
    purchase_frequency INTEGER,
    total_spent REAL,
    spending_velocity REAL,
    -- Engagement features
    achievements_unlocked INTEGER,
    content_completion_pct REAL,
    feature_adoption_pct REAL,
    content_engagement REAL,
    -- Risk scores
    activity_risk REAL,
    engagement_risk REAL,
    monetization_risk REAL,
    overall_risk_score REAL,
    -- Target
    churned INTEGER,
    churn_probability REAL,
    
    PRIMARY KEY (player_id, date)
);
```

## Feature Engineering

### Feature Categories

#### 1. Time-Based Features
```python
- account_age_days: Days since registration
- day_of_week: 0-6 (Monday to Sunday)
- is_weekend: Binary flag
- month: 1-12
```

#### 2. Behavioral Aggregations
```python
- sessions_per_week_rolling_3m: 3-month rolling average
- sessions_trend: Current vs. historical ratio
- total_playtime_hours: Estimated total playtime
```

#### 3. Risk Indicators
```python
- activity_risk: Composite score (0-1)
- engagement_risk: Content engagement risk
- monetization_risk: Spending behavior risk
- overall_risk_score: Weighted combination
```

### Feature Engineering Pipeline
```python
from src.feature_engineering.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
df_transformed = engineer.engineer_all_features(df_raw)
df_model_ready, feature_cols = engineer.prepare_model_data(df_transformed)
```

## Model Training

### Training Pipeline

```python
from src.ml_pipeline.train import ChurnModelTrainer

# Initialize trainer
trainer = ChurnModelTrainer(random_state=42)

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_data(
    df, feature_cols, target_col='churned'
)

# Scale features
X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)

# Train models
trainer.train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
trainer.train_random_forest(X_train_scaled, y_train)
trainer.train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test)

# Create ensemble
trainer.create_ensemble(X_test_scaled, y_test)

# Evaluate
for model_name in trainer.models.keys():
    metrics = trainer.evaluate_model(model_name, X_test_scaled, y_test)

# Save models
trainer.save_models('models/')
```

### Hyperparameter Configuration

#### XGBoost
```python
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': 3  # Handle class imbalance
}
```

#### Random Forest
```python
params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4
}
```

## Deployment

### API Deployment

#### Start FastAPI Server
```bash
# Local development
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production (with Docker)
docker-compose up api
```

#### API Endpoints

**Health Check**
```bash
GET /health
```

**Predict Churn**
```bash
POST /predict
Content-Type: application/json

{
  "players": [
    {
      "player_id": "PLAYER_00001",
      "sessions_per_week": 3,
      "avg_session_length_mins": 45,
      "days_since_last_login": 7,
      ...
    }
  ],
  "model_name": "xgboost"
}
```

**Response**
```json
{
  "predictions": [
    {
      "player_id": "PLAYER_00001",
      "churn_probability": 0.6542,
      "risk_category": "Medium",
      "top_risk_factors": [
        {
          "factor": "Low Session Frequency",
          "impact": 0.3,
          "value": 3
        }
      ]
    }
  ],
  "model_used": "xgboost",
  "prediction_time": "2024-12-19T10:30:00"
}
```

### Dashboard Deployment

```bash
# Local
streamlit run streamlit/dashboard.py

# Docker
docker-compose up streamlit
```

Access at: http://localhost:8501

## Monitoring

### Model Performance Monitoring

#### Metrics to Track
1. **Prediction Distribution**
   - Track risk score distribution over time
   - Alert if distribution shifts significantly

2. **Feature Drift**
   - Monitor mean/std of key features
   - Detect data quality issues

3. **Business Metrics**
   - Actual churn rate vs. predicted
   - Intervention effectiveness
   - ROI of retention campaigns

#### Monitoring Dashboard (Grafana)

```yaml
# grafana/datasources/postgres.yml
apiVersion: 1
datasources:
  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: churn_db
    user: postgres
    secureJsonData:
      password: postgres
```

### Alerting Rules

```python
# Check prediction distribution
if high_risk_percentage > 40%:
    alert("Unusually high churn risk detected")

# Check feature drift
if abs(current_mean - historical_mean) > 2 * std:
    alert("Feature drift detected")

# Check model performance
if actual_churn_rate > predicted_rate * 1.5:
    alert("Model underperforming, retraining needed")
```

## API Reference

### Complete API Documentation

See interactive API docs at: `http://localhost:8000/docs` (Swagger UI)

### Python Client Example

```python
import requests

# Prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "players": [player_data],
        "model_name": "ensemble"
    }
)

predictions = response.json()
```

## Business Impact

### Expected Outcomes

**Baseline Metrics**
- Current churn rate: 25%
- Average player LTV: $300
- Players: 100,000

**With Churn Prediction**
- Identified high-risk players: 15,000
- Intervention cost: $5 per player
- Expected churn reduction: 15%
- Players saved: 2,250
- Revenue retained: $675,000
- Total cost: $75,000
- **Net benefit: $600,000**
- **ROI: 800%**

### Success Metrics

1. **Model Accuracy**: ROC-AUC > 0.90
2. **Intervention Effectiveness**: >10% churn reduction
3. **Business Impact**: >5x ROI on retention campaigns
4. **Operational Efficiency**: <100ms prediction latency

## Troubleshooting

### Common Issues

**Issue: Models not loading**
```python
# Solution: Check model files exist
import os
assert os.path.exists('models/xgboost_model.pkl')
```

**Issue: Redis connection failed**
```python
# Solution: Verify Redis is running
docker ps | grep redis
```

**Issue: Airflow DAG not running**
```bash
# Solution: Check Airflow scheduler
docker logs churn_airflow_scheduler
```

## Future Enhancements

1. **Real-time Streaming**: Kafka integration for live predictions
2. **Deep Learning**: LSTM for time-series patterns
3. **Causal Inference**: Uplift modeling for interventions
4. **Multi-game Support**: Cross-game churn prediction
5. **AutoML**: Automated hyperparameter tuning

## Contact & Support

- **Technical Issues**: ml-team@company.com
- **Dashboard Access**: ops-team@company.com
- **Documentation**: https://docs.company.com/churn-prediction

## Changelog

### v1.0.0 (December 2024)
- Initial release
- Ensemble model implementation
- Streamlit dashboard
- A/B testing framework
- Complete ETL pipeline
