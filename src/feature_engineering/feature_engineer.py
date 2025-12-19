"""
Feature Engineering Module
Transforms raw data into ML-ready features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for churn prediction"""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating time-based features...")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Account age in days
        if 'registration_date' in df.columns:
            df['registration_date'] = pd.to_datetime(df['registration_date'])
            df['account_age_days'] = (df['date'] - df['registration_date']).dt.days
        
        # Day of week, month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced behavioral features"""
        logger.info("Creating behavioral features...")
        
        df = df.copy()
        
        # Total playtime estimate
        df['total_playtime_hours'] = (df['sessions_per_week'] * df['avg_session_length_mins'] * 4) / 60
        
        # Activity consistency score (inverse of session decay)
        df['activity_consistency'] = 1 - df['session_decay_rate'].clip(0, 1)
        
        # Engagement score (composite)
        df['engagement_score'] = (
            (df['sessions_per_week'] / 20) * 0.3 +  # Normalized sessions
            (df['avg_session_length_mins'] / 120) * 0.3 +  # Normalized duration
            (1 - df['days_since_last_login'] / 30) * 0.4  # Recency
        ).clip(0, 1)
        
        # Create rolling features (per player)
        df = df.sort_values(['player_id', 'date'])
        
        # Rolling averages (3-month window)
        for col in ['sessions_per_week', 'avg_session_length_mins']:
            df[f'{col}_rolling_3m'] = df.groupby('player_id')[col].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
        
        # Trend detection (comparing recent vs. historical)
        df['sessions_trend'] = df['sessions_per_week'] / (df['sessions_per_week_rolling_3m'] + 0.1)
        
        return df
    
    def create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance-based features"""
        logger.info("Creating performance features...")
        
        df = df.copy()
        
        # Performance score
        df['performance_score'] = (
            df['win_rate'] * 0.5 +
            (df['kd_ratio'] / 3).clip(0, 1) * 0.3 +  # Normalize K/D
            (df['current_rank'] / 100) * 0.2
        )
        
        # Skill progression indicator
        df['skill_improving'] = (df['rank_change'] > 0).astype(int)
        df['skill_declining'] = (df['rank_change'] < -5).astype(int)
        
        # Performance consistency (rolling std)
        df['win_rate_std'] = df.groupby('player_id')['win_rate'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
        
        return df
    
    def create_social_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create social interaction features"""
        logger.info("Creating social features...")
        
        df = df.copy()
        
        # Social engagement score
        df['social_engagement'] = (
            (df['friends_online'] / 15).clip(0, 1) * 0.6 +
            df['party_play_percentage'] * 0.4
        )
        
        # Social risk indicators
        df['low_social_engagement'] = (df['friends_online'] < 2).astype(int)
        df['solo_player'] = (df['party_play_percentage'] < 0.2).astype(int)
        df['has_toxicity_issues'] = (df['toxicity_reports_received'] > 0).astype(int)
        
        return df
    
    def create_monetization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monetization-based features"""
        logger.info("Creating monetization features...")
        
        df = df.copy()
        
        # Cumulative spending per player
        df = df.sort_values(['player_id', 'date'])
        df['total_spent'] = df.groupby('player_id')['monthly_spent'].cumsum()
        
        # Spending behavior
        df['is_spender'] = (df['total_spent'] > 0).astype(int)
        df['is_whale'] = (df['total_spent'] > 500).astype(int)
        df['recent_spender'] = (df['days_since_last_purchase'] < 30).astype(int)
        
        # Spending velocity
        df['spending_velocity'] = df['monthly_spent'] * df['purchase_frequency']
        
        # Average transaction value
        df['avg_transaction_value'] = df['monthly_spent'] / (df['purchase_frequency'] + 1)
        
        # Spending trend
        df['spending_rolling_3m'] = df.groupby('player_id')['monthly_spent'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['spending_trend'] = df['monthly_spent'] / (df['spending_rolling_3m'] + 0.1)
        
        return df
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create content engagement features"""
        logger.info("Creating engagement features...")
        
        df = df.copy()
        
        # Content engagement score
        df['content_engagement'] = (
            (df['content_completion_pct']) * 0.4 +
            (df['feature_adoption_pct']) * 0.3 +
            (df['achievements_unlocked'] / 100).clip(0, 1) * 0.3
        )
        
        # Achievement velocity
        df['achievement_velocity'] = df.groupby('player_id')['achievements_unlocked'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Content exhaustion indicator
        df['content_exhausted'] = (
            (df['content_completion_pct'] > 0.9) & 
            (df['achievement_velocity'] < 1)
        ).astype(int)
        
        return df
    
    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create churn risk indicators"""
        logger.info("Creating risk indicators...")
        
        df = df.copy()
        
        # Activity risk
        df['activity_risk'] = (
            (df['sessions_per_week'] < 2).astype(int) * 3 +
            (df['days_since_last_login'] > 14).astype(int) * 3 +
            (df['session_decay_rate'] > 0.3).astype(int) * 2
        ) / 8
        
        # Engagement risk
        df['engagement_risk'] = (
            (df['engagement_score'] < 0.3).astype(int) * 2 +
            df['low_social_engagement'] * 2 +
            df['content_exhausted'] * 1
        ) / 5
        
        # Monetization risk
        df['monetization_risk'] = (
            (df['days_since_last_purchase'] > 90).astype(int) * 2 +
            (df['spending_trend'] < 0.5).astype(int) * 1 +
            (~df['is_spender'].astype(bool)).astype(int) * 1
        ) / 4
        
        # Overall risk score
        df['overall_risk_score'] = (
            df['activity_risk'] * 0.4 +
            df['engagement_risk'] * 0.3 +
            df['monetization_risk'] * 0.3
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Performance x Engagement
        df['performance_engagement'] = df['performance_score'] * df['engagement_score']
        
        # Social x Spending
        df['social_spending'] = df['social_engagement'] * df['spending_velocity']
        
        # Activity x Monetization
        df['activity_monetization'] = df['sessions_per_week'] * df['monthly_spent']
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting feature engineering pipeline...")
        
        df = self.create_time_based_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_performance_features(df)
        df = self.create_social_features(df)
        df = self.create_monetization_features(df)
        df = self.create_engagement_features(df)
        df = self.create_risk_indicators(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for modeling"""
        
        # Exclude non-feature columns
        exclude_cols = [
            'player_id', 'date', 'registration_date', 'archetype',
            'churned', 'churn_probability', 'country', 'platform', 'month'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for modeling
        
        Returns:
            Tuple of (processed_df, feature_columns)
        """
        # Get most recent data for each player
        df_latest = df.sort_values('date').groupby('player_id').last().reset_index()
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df_latest)
        
        # Handle missing values
        df_latest[feature_cols] = df_latest[feature_cols].fillna(0)
        
        # Handle infinite values
        df_latest[feature_cols] = df_latest[feature_cols].replace([np.inf, -np.inf], 0)
        
        logger.info(f"Prepared {len(df_latest)} samples with {len(feature_cols)} features")
        
        return df_latest, feature_cols


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering module initialized")
