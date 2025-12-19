"""
Data Ingestion Module
Handles data collection from various sources
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.steam_api_connector import SteamAPIConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestor:
    """Main data ingestion orchestrator"""
    
    def __init__(self, data_source: str = 'synthetic', steam_api_key: str = None):
        """
        Initialize data ingestor
        
        Args:
            data_source: 'synthetic' or 'steam_api'
            steam_api_key: Steam API key (required if using steam_api)
        """
        self.data_source = data_source
        self.steam_api_key = steam_api_key
        
        if data_source == 'steam_api' and not steam_api_key:
            raise ValueError("Steam API key required for steam_api data source")
        
        if data_source == 'steam_api':
            self.steam_connector = SteamAPIConnector(steam_api_key)
    
    def load_synthetic_data(self, file_path: str) -> pd.DataFrame:
        """Load synthetic data from CSV"""
        logger.info(f"Loading synthetic data from {file_path}")
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def load_steam_data(self, steam_ids: list, app_id: int) -> pd.DataFrame:
        """Load data from Steam API"""
        logger.info(f"Fetching data for {len(steam_ids)} players from Steam API")
        df = self.steam_connector.batch_extract_features(steam_ids, app_id)
        logger.info(f"Fetched data for {len(df)} players")
        return df
    
    def ingest_data(self, **kwargs) -> pd.DataFrame:
        """
        Ingest data based on configured source
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame with player data
        """
        if self.data_source == 'synthetic':
            file_path = kwargs.get('file_path', 'data/raw/player_data.csv')
            return self.load_synthetic_data(file_path)
        
        elif self.data_source == 'steam_api':
            steam_ids = kwargs.get('steam_ids', [])
            app_id = kwargs.get('app_id')
            
            if not steam_ids or not app_id:
                raise ValueError("steam_ids and app_id required for Steam API ingestion")
            
            return self.load_steam_data(steam_ids, app_id)
        
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def save_raw_data(self, df: pd.DataFrame, output_path: str):
        """Save raw ingested data"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved raw data to {output_path}")


if __name__ == "__main__":
    # Example usage
    ingestor = DataIngestor(data_source='synthetic')
    df = ingestor.ingest_data(file_path='data/raw/player_data.csv')
    print(f"Ingested {len(df)} records")
    print(df.head())
