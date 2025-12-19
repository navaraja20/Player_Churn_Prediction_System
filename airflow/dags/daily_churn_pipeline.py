"""
Airflow DAG for Daily Churn Prediction Pipeline
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def run_data_ingestion(**context):
    """Task to ingest player data"""
    from src.data_ingestion import DataIngestor
    
    ingestor = DataIngestor(data_source='synthetic')
    df = ingestor.ingest_data(file_path='data/raw/player_data.csv')
    
    # Save to staging
    df.to_csv('data/staging/ingested_data.csv', index=False)
    
    return f"Ingested {len(df)} records"


def run_feature_engineering(**context):
    """Task to engineer features"""
    import pandas as pd
    from src.feature_engineering.feature_engineer import FeatureEngineer
    
    # Load staged data
    df = pd.read_csv('data/staging/ingested_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.engineer_all_features(df)
    
    # Save to staging
    df_features.to_csv('data/staging/features.csv', index=False)
    
    return f"Engineered {df_features.shape[1]} features"


def run_etl_load(**context):
    """Task to load data to destinations"""
    import pandas as pd
    from src.etl.etl_pipeline import ETLPipeline
    
    # Load features
    df = pd.read_csv('data/staging/features.csv')
    
    # Configure pipeline (for loading only)
    config = {
        'data_source': 'synthetic',
        'use_postgres': False,  # Set to True when PostgreSQL is available
        'use_redis': False,     # Set to True when Redis is available
        'output_csv': 'data/processed/player_features.csv'
    }
    
    pipeline = ETLPipeline(config)
    
    # Load to destinations
    pipeline.load(df)
    
    return f"Loaded {len(df)} records"


def run_model_predictions(**context):
    """Task to generate predictions for all players"""
    import pandas as pd
    import pickle
    import json
    import os
    
    # Load data
    df = pd.read_csv('data/processed/player_features.csv')
    
    # Load model
    model_path = 'models/xgboost_model.pkl'
    if not os.path.exists(model_path):
        print("No trained model found, skipping predictions")
        return "Skipped - no model"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Get latest data per player
    df_latest = df.sort_values('date').groupby('player_id').last().reset_index()
    
    # Prepare features
    X = df_latest[feature_names].fillna(0).replace([float('inf'), -float('inf')], 0)
    
    # Predict
    predictions = model.predict_proba(X)[:, 1]
    
    # Add predictions to dataframe
    df_latest['churn_risk_score'] = predictions
    df_latest['risk_category'] = pd.cut(
        predictions,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Save predictions
    df_latest[['player_id', 'churn_risk_score', 'risk_category']].to_csv(
        'data/predictions/daily_predictions.csv',
        index=False
    )
    
    return f"Generated predictions for {len(df_latest)} players"


def run_model_monitoring(**context):
    """Task to monitor model performance"""
    import pandas as pd
    import json
    from datetime import datetime
    
    # Load predictions
    df_pred = pd.read_csv('data/predictions/daily_predictions.csv')
    
    # Calculate distribution metrics
    metrics = {
        'date': datetime.now().isoformat(),
        'total_players': len(df_pred),
        'high_risk_count': (df_pred['risk_category'] == 'High').sum(),
        'medium_risk_count': (df_pred['risk_category'] == 'Medium').sum(),
        'low_risk_count': (df_pred['risk_category'] == 'Low').sum(),
        'avg_risk_score': df_pred['churn_risk_score'].mean(),
        'high_risk_percentage': (df_pred['risk_category'] == 'High').mean() * 100
    }
    
    # Save monitoring metrics
    os.makedirs('data/monitoring', exist_ok=True)
    
    # Append to history
    history_path = 'data/monitoring/metrics_history.jsonl'
    with open(history_path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    print(f"High Risk Players: {metrics['high_risk_count']} ({metrics['high_risk_percentage']:.1f}%)")
    
    return metrics


# Define the DAG
dag = DAG(
    'daily_churn_prediction_pipeline',
    default_args=default_args,
    description='Daily ETL and prediction pipeline for player churn',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    catchup=False,
    tags=['churn', 'ml', 'production']
)


# Define tasks
task_ingest = PythonOperator(
    task_id='ingest_data',
    python_callable=run_data_ingestion,
    dag=dag,
)

task_feature_engineering = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag,
)

task_load = PythonOperator(
    task_id='load_data',
    python_callable=run_etl_load,
    dag=dag,
)

task_predict = PythonOperator(
    task_id='generate_predictions',
    python_callable=run_model_predictions,
    dag=dag,
)

task_monitor = PythonOperator(
    task_id='monitor_performance',
    python_callable=run_model_monitoring,
    dag=dag,
)

# Create staging and predictions directories
task_create_dirs = BashOperator(
    task_id='create_directories',
    bash_command='mkdir -p data/staging data/predictions data/monitoring',
    dag=dag,
)

# Set task dependencies
task_create_dirs >> task_ingest >> task_feature_engineering >> task_load >> task_predict >> task_monitor
