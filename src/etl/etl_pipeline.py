"""
ETL Pipeline
Orchestrates Extract, Transform, Load operations
"""

import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import redis
import json
from datetime import datetime
import logging
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion import DataIngestor
from feature_engineering.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETLPipeline:
    """ETL Pipeline for player churn data"""
    
    def __init__(self, config: Dict):
        """
        Initialize ETL pipeline
        
        Args:
            config: Configuration dictionary with database and redis settings
        """
        self.config = config
        self.ingestor = DataIngestor(data_source=config.get('data_source', 'synthetic'))
        self.feature_engineer = FeatureEngineer()
        
        # Database connection
        self.db_conn = None
        if config.get('use_postgres', False):
            self.db_conn = self._connect_to_postgres()
        
        # Redis connection
        self.redis_client = None
        if config.get('use_redis', False):
            self.redis_client = self._connect_to_redis()
    
    def _connect_to_postgres(self):
        """Connect to PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host=self.config.get('postgres_host', 'localhost'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'churn_db'),
                user=self.config.get('postgres_user', 'postgres'),
                password=self.config.get('postgres_password', 'password')
            )
            logger.info("Connected to PostgreSQL")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return None
    
    def _connect_to_redis(self):
        """Connect to Redis"""
        try:
            client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            client.ping()
            logger.info("Connected to Redis")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None
    
    def extract(self, **kwargs) -> pd.DataFrame:
        """
        Extract data from source
        
        Returns:
            Raw DataFrame
        """
        logger.info("=== EXTRACT PHASE ===")
        df = self.ingestor.ingest_data(**kwargs)
        logger.info(f"Extracted {len(df)} records")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with feature engineering
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Transformed DataFrame with engineered features
        """
        logger.info("=== TRANSFORM PHASE ===")
        df_transformed = self.feature_engineer.engineer_all_features(df)
        logger.info(f"Transformation complete. Shape: {df_transformed.shape}")
        return df_transformed
    
    def load_to_postgres(self, df: pd.DataFrame, table_name: str = 'player_features'):
        """
        Load data to PostgreSQL
        
        Args:
            df: DataFrame to load
            table_name: Target table name
        """
        if not self.db_conn:
            logger.warning("PostgreSQL not connected, skipping load")
            return
        
        logger.info(f"Loading {len(df)} records to PostgreSQL table: {table_name}")
        
        try:
            # Create table if not exists
            self._create_postgres_table(df, table_name)
            
            # Insert data
            cursor = self.db_conn.cursor()
            
            # Prepare data
            columns = list(df.columns)
            values = [tuple(row) for row in df.values]
            
            # Build insert query
            insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(columns))})
                ON CONFLICT (player_id, date) DO UPDATE SET
                {', '.join([f'{col} = EXCLUDED.{col}' for col in columns if col not in ['player_id', 'date']])}
            """
            
            execute_batch(cursor, insert_query, values, page_size=1000)
            self.db_conn.commit()
            
            logger.info(f"Successfully loaded {len(df)} records to {table_name}")
            
        except Exception as e:
            logger.error(f"Error loading to PostgreSQL: {e}")
            self.db_conn.rollback()
    
    def _create_postgres_table(self, df: pd.DataFrame, table_name: str):
        """Create PostgreSQL table if not exists"""
        
        # Map pandas dtypes to PostgreSQL types
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            pg_type = type_mapping.get(dtype, 'TEXT')
            columns.append(f"{col} {pg_type}")
        
        # Add primary key
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)},
                PRIMARY KEY (player_id, date)
            )
        """
        
        cursor = self.db_conn.cursor()
        cursor.execute(create_table_sql)
        self.db_conn.commit()
        logger.info(f"Table {table_name} ready")
    
    def load_to_redis(self, df: pd.DataFrame, key_prefix: str = 'player'):
        """
        Load latest player features to Redis for real-time serving
        
        Args:
            df: DataFrame with player features
            key_prefix: Redis key prefix
        """
        if not self.redis_client:
            logger.warning("Redis not connected, skipping load")
            return
        
        logger.info(f"Loading {len(df)} player features to Redis")
        
        try:
            # Get latest data for each player
            df_latest = df.sort_values('date').groupby('player_id').last().reset_index()
            
            # Store each player's features
            pipe = self.redis_client.pipeline()
            
            for _, row in df_latest.iterrows():
                player_id = row['player_id']
                
                # Convert row to dict
                features = row.to_dict()
                
                # Convert datetime to string
                for key, value in features.items():
                    if pd.isna(value):
                        features[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        features[key] = value.isoformat()
                    elif isinstance(value, (np.integer, np.floating)):
                        features[key] = float(value)
                
                # Store as JSON
                redis_key = f"{key_prefix}:{player_id}"
                pipe.setex(
                    redis_key,
                    86400 * 7,  # 7 day TTL
                    json.dumps(features)
                )
            
            pipe.execute()
            logger.info(f"Successfully loaded {len(df_latest)} players to Redis")
            
        except Exception as e:
            logger.error(f"Error loading to Redis: {e}")
    
    def load_to_csv(self, df: pd.DataFrame, output_path: str):
        """
        Load data to CSV file
        
        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        logger.info(f"Saving data to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    
    def load(self, df: pd.DataFrame):
        """
        Load data to all configured destinations
        
        Args:
            df: Transformed DataFrame
        """
        logger.info("=== LOAD PHASE ===")
        
        # Load to PostgreSQL
        if self.config.get('use_postgres', False):
            self.load_to_postgres(df, table_name='player_features')
        
        # Load to Redis
        if self.config.get('use_redis', False):
            self.load_to_redis(df)
        
        # Always save to CSV as backup
        output_path = self.config.get('output_csv', 'data/processed/player_features.csv')
        self.load_to_csv(df, output_path)
        
        logger.info("Load phase complete")
    
    def run_pipeline(self, **extract_kwargs):
        """
        Run complete ETL pipeline
        
        Args:
            **extract_kwargs: Arguments for extract phase
        """
        logger.info("Starting ETL Pipeline...")
        start_time = datetime.now()
        
        try:
            # Extract
            df_raw = self.extract(**extract_kwargs)
            
            # Transform
            df_transformed = self.transform(df_raw)
            
            # Load
            self.load(df_transformed)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"ETL Pipeline completed successfully in {duration:.2f} seconds")
            
            return df_transformed
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            raise
        
        finally:
            self.close_connections()
    
    def close_connections(self):
        """Close database connections"""
        if self.db_conn:
            self.db_conn.close()
            logger.info("Closed PostgreSQL connection")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("Closed Redis connection")


if __name__ == "__main__":
    # Example configuration
    config = {
        'data_source': 'synthetic',
        'use_postgres': False,  # Set to True if PostgreSQL is available
        'use_redis': False,     # Set to True if Redis is available
        'output_csv': 'data/processed/player_features.csv'
    }
    
    # Run pipeline
    pipeline = ETLPipeline(config)
    df = pipeline.run_pipeline(file_path='data/raw/player_data.csv')
    
    print(f"\nPipeline complete! Processed {len(df)} records")
    print(f"Features: {df.shape[1]} columns")
