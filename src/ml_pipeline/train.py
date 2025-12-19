"""
ML Training Pipeline
Trains churn prediction models
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, List
import logging

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
from lightgbm import LGBMClassifier
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """Train and evaluate churn prediction models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.metrics = {}
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'churned',
        test_size: float = 0.2
    ) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target variable column name
            test_size: Test set size
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # Check for missing target
        df = df[df[target_col].notna()].copy()
        
        # Split features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scaler_name: str = 'standard'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features
        
        Args:
            X_train: Training features
            X_test: Test features
            scaler_name: Name for storing the scaler
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[scaler_name] = scaler
        logger.info(f"Features scaled using {scaler.__class__.__name__}")
        
        return X_train_scaled, X_test_scaled
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        params: Dict = None
    ):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'scale_pos_weight': 1
        }
        
        if params:
            default_params.update(params)
        
        model = xgb.XGBClassifier(**default_params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        logger.info("XGBoost model trained")
        
        return model
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Dict = None
    ):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        logger.info("Random Forest model trained")
        
        return model
    
    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        params: Dict = None
    ):
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        model = LGBMClassifier(**default_params)
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    # early_stopping(stopping_rounds=20)
                ]
            )
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        logger.info("LightGBM model trained")
        
        return model
    
    def create_ensemble(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        weights: Dict[str, float] = None
    ):
        """
        Create ensemble model from trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            weights: Model weights (if None, uses equal weights)
        """
        logger.info("Creating ensemble model...")
        
        if not self.models:
            raise ValueError("No models trained yet")
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions[name] = pred_proba
        
        # Weighted average
        ensemble_proba = sum(
            predictions[name] * weights.get(name, 0)
            for name in predictions.keys()
        )
        
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Store ensemble as a "model"
        self.models['ensemble'] = {
            'type': 'ensemble',
            'weights': weights,
            'predictions': ensemble_proba
        }
        
        logger.info(f"Ensemble created with weights: {weights}")
        
        return ensemble_pred, ensemble_proba
    
    def evaluate_model(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name} model...")
        
        model = self.models[model_name]
        
        # Get predictions
        if model_name == 'ensemble':
            y_pred_proba = model['predictions']
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        self.metrics[model_name] = metrics
        
        logger.info(f"{model_name} Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_feature_importance(
        self,
        model_name: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance for a model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} does not have feature importances")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def explain_with_shap(
        self,
        model_name: str,
        X_sample: np.ndarray,
        max_samples: int = 100
    ):
        """
        Generate SHAP explanations
        
        Args:
            model_name: Name of the model
            X_sample: Sample data for SHAP
            max_samples: Maximum samples to use for SHAP
        """
        logger.info(f"Generating SHAP explanations for {model_name}...")
        
        model = self.models[model_name]
        
        # Limit samples for performance
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        logger.info("SHAP values computed")
        
        return explainer, shap_values
    
    def save_models(self, output_dir: str = 'models'):
        """Save trained models and scalers"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if name != 'ensemble':
                model_path = os.path.join(output_dir, f'{name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} model to {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(output_dir, f'{name}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved {name} scaler to {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(output_dir, 'feature_names.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"All models and artifacts saved to {output_dir}")
    
    def load_models(self, model_dir: str = 'models'):
        """Load trained models and scalers"""
        logger.info(f"Loading models from {model_dir}...")
        
        # Load models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('_model.pkl'):
                model_name = model_file.replace('_model.pkl', '')
                model_path = os.path.join(model_dir, model_file)
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name} model")
        
        # Load scalers
        for scaler_file in os.listdir(model_dir):
            if scaler_file.endswith('_scaler.pkl'):
                scaler_name = scaler_file.replace('_scaler.pkl', '')
                scaler_path = os.path.join(model_dir, scaler_file)
                with open(scaler_path, 'rb') as f:
                    self.scalers[scaler_name] = pickle.load(f)
                logger.info(f"Loaded {scaler_name} scaler")
        
        # Load feature names
        features_path = os.path.join(model_dir, 'feature_names.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
        
        logger.info("Models loaded successfully")


if __name__ == "__main__":
    print("ML Training Pipeline initialized")
