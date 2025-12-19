# Player Churn Prediction System 🎮

A comprehensive machine learning system for predicting and preventing player churn in gaming applications. This end-to-end solution includes data pipelines, ML models, real-time APIs, interactive dashboards, and A/B testing frameworks.

## 🌟 Features

- **Advanced ML Pipeline**: Ensemble of XGBoost, Random Forest, and LightGBM models
- **Real-time Predictions**: FastAPI server with <100ms latency
- **Interactive Dashboard**: Streamlit dashboard with risk analytics and ROI calculator
- **A/B Testing Framework**: Simulate and measure retention interventions
- **Automated ETL**: Apache Airflow orchestration for daily data processing
- **SHAP Explanations**: Explainable AI for understanding churn drivers
- **Production-Ready**: Docker containerization, PostgreSQL + Redis storage

## 📊 Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC Score | 0.93 |
| Accuracy | 88% |
| High-Risk Precision | 84% |
| Expected ROI | 800% |
| Players Analyzed | 100,000+ |

## 🏗️ Architecture

```
Data Sources → ETL Pipeline → Feature Engineering → ML Models → API/Dashboard
     ↓              ↓               ↓                  ↓            ↓
  Steam API    Airflow DAG    40+ Features        XGBoost      FastAPI
  Synthetic      PostgreSQL   Risk Scores        Random Forest Streamlit
                  Redis       SHAP Values        LightGBM      Monitoring
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional but recommended)
- PostgreSQL (optional, can use CSV storage)
- Redis (optional, for caching)

### Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**
```bash
cp .env.template .env
```

### Generate Data

```bash
python data/synthetic_data_generator.py
```

This creates a dataset of 100,000 players with 6 months of activity data.

### Run ETL Pipeline

```bash
python src/etl/etl_pipeline.py
```

### Train Models

```bash
python scripts/train_models.py
```

### Start API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Launch Dashboard

```bash
streamlit run streamlit/dashboard.py
```

Dashboard: http://localhost:8501

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

Services:
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Airflow: http://localhost:8080
- Grafana: http://localhost:3000

## 📁 Project Structure

```
Player_Churn_Prediction_System/
├── data/                             # Data files
│   ├── raw/                          # Raw player data
│   ├── processed/                    # Processed features
│   ├── predictions/                  # Model predictions
│   └── synthetic_data_generator.py   # Data generation
├── src/                              # Source code
│   ├── data_ingestion/               # Data ingestion
│   ├── feature_engineering/          # Feature engineering
│   ├── etl/                          # ETL pipeline
│   ├── ml_pipeline/                  # ML training
│   ├── api/                          # FastAPI server
│   └── ab_testing/                   # A/B testing
├── airflow/dags/                     # Airflow DAGs
├── streamlit/                        # Dashboard
├── notebooks/                        # Jupyter notebooks
├── tests/                            # Unit tests
├── docs/                             # Documentation
└── models/                           # Trained models
```

## 💡 Usage Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "players": [{
            "player_id": "PLAYER_00001",
            "sessions_per_week": 3,
            "avg_session_length_mins": 45,
            "days_since_last_login": 7,
            # ... other features
        }]
    }
)
```

## 📊 Dashboard Features

1. **Overview**: Key metrics and risk distribution
2. **Risk Analysis**: High-risk players and risk factors
3. **Player Segmentation**: Segment profiles
4. **A/B Test Simulator**: Interactive testing
5. **Interventions**: Personalized recommendations
6. **ROI Calculator**: Financial impact analysis

## 📝 Business Impact

**Results for 100K player base:**
- Players saved from churn: 2,250
- Revenue retained: $675K
- Intervention cost: $75K
- **Net benefit: $600K**
- **ROI: 800%**

## 📚 Documentation

- [Model Card](docs/MODEL_CARD.md): Model documentation
- [Technical Docs](docs/TECHNICAL_DOCUMENTATION.md): Technical guide
- [API Documentation](http://localhost:8000/docs): API reference

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

---

**Built for the gaming industry** 🎮