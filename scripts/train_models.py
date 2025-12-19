"""
Training Script
Train churn prediction models
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering.feature_engineer import FeatureEngineer
from src.ml_pipeline.train import ChurnModelTrainer

def main():
    print("="*60)
    print("PLAYER CHURN PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load processed data
    print("\n1. Loading data...")
    df = pd.read_csv('data/processed/player_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Loaded {len(df)} records for {df['player_id'].nunique()} players")
    
    # Prepare data for modeling
    print("\n2. Preparing model data...")
    engineer = FeatureEngineer()
    df_model, feature_cols = engineer.prepare_model_data(df)
    print(f"   Model-ready data: {len(df_model)} samples, {len(feature_cols)} features")
    
    # Initialize trainer
    print("\n3. Initializing trainer...")
    trainer = ChurnModelTrainer(random_state=42)
    
    # Prepare train/test split
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df_model,
        feature_cols,
        target_col='churned',
        test_size=0.2
    )
    
    # Scale features
    print("\n5. Scaling features...")
    X_train_scaled, X_test_scaled = trainer.scale_features(
        X_train, X_test, scaler_name='standard'
    )
    
    # Train models
    print("\n6. Training models...")
    print("\n   a) Training XGBoost...")
    trainer.train_xgboost(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        params={'n_estimators': 200, 'max_depth': 6}
    )
    
    print("\n   b) Training Random Forest...")
    trainer.train_random_forest(
        X_train_scaled, y_train,
        params={'n_estimators': 200, 'max_depth': 10}
    )
    
    print("\n   c) Training LightGBM...")
    trainer.train_lightgbm(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        params={'n_estimators': 200, 'max_depth': 6}
    )
    
    # Create ensemble
    print("\n7. Creating ensemble model...")
    trainer.create_ensemble(X_test_scaled, y_test)
    
    # Evaluate all models
    print("\n8. Evaluating models...")
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    for model_name in ['xgboost', 'random_forest', 'lightgbm', 'ensemble']:
        metrics = trainer.evaluate_model(model_name, X_test_scaled, y_test)
        print(f"\n{model_name.upper()}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Feature importance
    print("\n9. Analyzing feature importance...")
    importance_df = trainer.get_feature_importance('xgboost', top_n=20)
    print("\nTop 20 Most Important Features:")
    print(importance_df.to_string(index=False))
    
    # Save models
    print("\n10. Saving models...")
    trainer.save_models('models/')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nModels saved to: models/")
    print("  - xgboost_model.pkl")
    print("  - random_forest_model.pkl")
    print("  - lightgbm_model.pkl")
    print("  - standard_scaler.pkl")
    print("  - feature_names.json")
    print("  - model_metrics.json")
    
    print("\nNext steps:")
    print("  1. Start API: uvicorn src.api.main:app --reload")
    print("  2. Launch dashboard: streamlit run streamlit/dashboard.py")
    print("  3. Make predictions: See README for examples")


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run if processed data exists
    if not os.path.exists('data/processed/player_features.csv'):
        print("ERROR: Processed data not found!")
        print("Please run: python src/etl/etl_pipeline.py")
        sys.exit(1)
    
    main()
