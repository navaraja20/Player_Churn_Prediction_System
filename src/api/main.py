"""
FastAPI Model Serving
Real-time churn prediction API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import pickle
import json
import redis
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Player Churn Prediction API",
    description="Real-time churn prediction and risk scoring for players",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and config
models = {}
scalers = {}
feature_names = []
redis_client = None


# Pydantic models for API
class PlayerFeatures(BaseModel):
    """Player features for prediction"""
    player_id: str
    sessions_per_week: float
    avg_session_length_mins: float
    days_since_last_login: int
    session_decay_rate: float = 0.0
    win_rate: float = 0.5
    kd_ratio: float = 1.0
    current_rank: int = 50
    rank_change: int = 0
    friends_online: int = 0
    party_play_percentage: float = 0.0
    toxicity_reports_received: int = 0
    monthly_spent: float = 0.0
    days_since_last_purchase: int = 180
    purchase_frequency: int = 0
    achievements_unlocked: int = 0
    content_completion_pct: float = 0.0
    feature_adoption_pct: float = 0.0


class PredictionRequest(BaseModel):
    """Request for churn prediction"""
    players: List[PlayerFeatures]
    model_name: str = Field(default="xgboost", description="Model to use for prediction")


class PredictionResponse(BaseModel):
    """Response with churn predictions"""
    player_id: str
    churn_probability: float
    risk_category: str
    top_risk_factors: List[Dict[str, float]]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    model_used: str
    prediction_time: str


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global models, scalers, feature_names, redis_client
    
    logger.info("Loading models...")
    
    model_dir = 'models'
    
    # Load models
    for model_file in ['xgboost_model.pkl', 'random_forest_model.pkl', 'lightgbm_model.pkl']:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace('_model.pkl', '')
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
            logger.info(f"Loaded {model_name} model")
    
    # Load scalers
    scaler_path = os.path.join(model_dir, 'standard_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers['standard'] = pickle.load(f)
        logger.info("Loaded standard scaler")
    
    # Load feature names
    features_path = os.path.join(model_dir, 'feature_names.json')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        logger.info(f"Loaded {len(feature_names)} feature names")
    
    # Connect to Redis (optional)
    try:
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        redis_client = None
    
    logger.info("Startup complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Player Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": list(models.keys())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": len(models),
        "redis_connected": redis_client is not None
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(models.keys()),
        "features_count": len(feature_names)
    }


def engineer_features(player_data: Dict) -> Dict:
    """Engineer features from raw player data"""
    
    # Create derived features
    features = player_data.copy()
    
    # Behavioral features
    features['total_playtime_hours'] = (
        features['sessions_per_week'] * features['avg_session_length_mins'] * 4
    ) / 60
    features['activity_consistency'] = 1 - min(features.get('session_decay_rate', 0), 1)
    features['engagement_score'] = min(1, max(0,
        (features['sessions_per_week'] / 20) * 0.3 +
        (features['avg_session_length_mins'] / 120) * 0.3 +
        (1 - features['days_since_last_login'] / 30) * 0.4
    ))
    
    # Performance features
    features['performance_score'] = (
        features['win_rate'] * 0.5 +
        min(1, features['kd_ratio'] / 3) * 0.3 +
        (features['current_rank'] / 100) * 0.2
    )
    features['skill_improving'] = 1 if features['rank_change'] > 0 else 0
    features['skill_declining'] = 1 if features['rank_change'] < -5 else 0
    
    # Social features
    features['social_engagement'] = (
        min(1, features['friends_online'] / 15) * 0.6 +
        features['party_play_percentage'] * 0.4
    )
    features['low_social_engagement'] = 1 if features['friends_online'] < 2 else 0
    features['solo_player'] = 1 if features['party_play_percentage'] < 0.2 else 0
    features['has_toxicity_issues'] = 1 if features['toxicity_reports_received'] > 0 else 0
    
    # Monetization features
    features['is_spender'] = 1 if features['monthly_spent'] > 0 else 0
    features['recent_spender'] = 1 if features['days_since_last_purchase'] < 30 else 0
    features['spending_velocity'] = features['monthly_spent'] * features['purchase_frequency']
    features['avg_transaction_value'] = features['monthly_spent'] / max(1, features['purchase_frequency'])
    
    # Engagement features
    features['content_engagement'] = (
        features['content_completion_pct'] * 0.4 +
        features['feature_adoption_pct'] * 0.3 +
        min(1, features['achievements_unlocked'] / 100) * 0.3
    )
    
    # Risk indicators
    features['activity_risk'] = (
        (1 if features['sessions_per_week'] < 2 else 0) * 3 +
        (1 if features['days_since_last_login'] > 14 else 0) * 3 +
        (1 if features.get('session_decay_rate', 0) > 0.3 else 0) * 2
    ) / 8
    
    features['engagement_risk'] = (
        (1 if features['engagement_score'] < 0.3 else 0) * 2 +
        features['low_social_engagement'] * 2
    ) / 4
    
    features['monetization_risk'] = (
        (1 if features['days_since_last_purchase'] > 90 else 0) * 2 +
        (1 if not features['is_spender'] else 0)
    ) / 3
    
    features['overall_risk_score'] = (
        features['activity_risk'] * 0.4 +
        features['engagement_risk'] * 0.3 +
        features['monetization_risk'] * 0.3
    )
    
    return features


def identify_risk_factors(features: Dict, shap_values: Optional[np.ndarray] = None) -> List[Dict]:
    """Identify top risk factors for a player"""
    
    risk_factors = []
    
    # Rule-based risk factors
    if features['sessions_per_week'] < 2:
        risk_factors.append({
            'factor': 'Low Session Frequency',
            'impact': 0.3,
            'value': features['sessions_per_week']
        })
    
    if features['days_since_last_login'] > 14:
        risk_factors.append({
            'factor': 'Inactive Player',
            'impact': 0.25,
            'value': features['days_since_last_login']
        })
    
    if features['session_decay_rate'] > 0.3:
        risk_factors.append({
            'factor': 'Declining Engagement',
            'impact': 0.2,
            'value': features['session_decay_rate']
        })
    
    if features['friends_online'] < 2:
        risk_factors.append({
            'factor': 'Low Social Connection',
            'impact': 0.15,
            'value': features['friends_online']
        })
    
    if features['days_since_last_purchase'] > 90:
        risk_factors.append({
            'factor': 'No Recent Purchases',
            'impact': 0.1,
            'value': features['days_since_last_purchase']
        })
    
    # Sort by impact and return top 5
    risk_factors.sort(key=lambda x: x['impact'], reverse=True)
    
    return risk_factors[:5]


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Predict churn for batch of players"""
    
    if request.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model_name} not found. Available: {list(models.keys())}"
        )
    
    model = models[request.model_name]
    
    predictions = []
    
    for player in request.players:
        # Convert to dict and engineer features
        player_dict = player.dict()
        engineered_features = engineer_features(player_dict)
        
        # Prepare features for model
        X = []
        for feat in feature_names:
            X.append(engineered_features.get(feat, 0))
        
        X = np.array(X).reshape(1, -1)
        
        # Handle missing/inf values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Predict
        churn_prob = model.predict_proba(X)[0, 1]
        
        # Determine risk category
        if churn_prob < 0.3:
            risk_cat = "Low"
        elif churn_prob < 0.7:
            risk_cat = "Medium"
        else:
            risk_cat = "High"
        
        # Identify risk factors
        risk_factors = identify_risk_factors(engineered_features)
        
        # Cache in Redis if available
        if redis_client:
            try:
                cache_key = f"player:{player.player_id}:prediction"
                cache_data = {
                    'churn_probability': float(churn_prob),
                    'risk_category': risk_cat,
                    'timestamp': datetime.now().isoformat()
                }
                redis_client.setex(cache_key, 3600, json.dumps(cache_data))
            except Exception as e:
                logger.warning(f"Redis cache failed: {e}")
        
        predictions.append(PredictionResponse(
            player_id=player.player_id,
            churn_probability=round(float(churn_prob), 4),
            risk_category=risk_cat,
            top_risk_factors=risk_factors
        ))
    
    return BatchPredictionResponse(
        predictions=predictions,
        model_used=request.model_name,
        prediction_time=datetime.now().isoformat()
    )


@app.get("/predict/{player_id}")
async def get_cached_prediction(player_id: str):
    """Get cached prediction for a player"""
    
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis cache not available")
    
    try:
        cache_key = f"player:{player_id}:prediction"
        cached = redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        else:
            raise HTTPException(status_code=404, detail="No cached prediction found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict-file")
async def batch_predict_from_file(background_tasks: BackgroundTasks):
    """Process batch predictions from file"""
    
    # This would process a large file in the background
    background_tasks.add_task(process_batch_file, "data/batch_input.csv")
    
    return {"message": "Batch prediction started", "status": "processing"}


def process_batch_file(file_path: str):
    """Background task to process batch predictions"""
    logger.info(f"Processing batch file: {file_path}")
    # Implementation would go here
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
