"""
Unit tests for Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample player data for testing"""
    np.random.seed(42)
    
    dates = [datetime.now() - timedelta(days=i*30) for i in range(3)]
    
    data = []
    for player_id in range(10):
        for date in dates:
            data.append({
                'player_id': f'PLAYER_{player_id:04d}',
                'date': date,
                'registration_date': date - timedelta(days=180),
                'archetype': np.random.choice(['hardcore', 'casual', 'whale']),
                'sessions_per_week': np.random.randint(1, 20),
                'avg_session_length_mins': np.random.uniform(30, 120),
                'days_since_last_login': np.random.randint(0, 30),
                'session_decay_rate': np.random.uniform(0, 0.5),
                'win_rate': np.random.uniform(0.3, 0.7),
                'kd_ratio': np.random.uniform(0.5, 2.0),
                'current_rank': np.random.randint(20, 100),
                'rank_change': np.random.randint(-10, 10),
                'friends_online': np.random.randint(0, 15),
                'party_play_percentage': np.random.uniform(0, 1),
                'toxicity_reports_received': np.random.randint(0, 3),
                'monthly_spent': np.random.uniform(0, 100),
                'days_since_last_purchase': np.random.randint(0, 180),
                'purchase_frequency': np.random.randint(0, 5),
                'achievements_unlocked': np.random.randint(0, 50),
                'content_completion_pct': np.random.uniform(0, 1),
                'feature_adoption_pct': np.random.uniform(0, 1),
            })
    
    return pd.DataFrame(data)


class TestFeatureEngineer:
    """Test suite for FeatureEngineer"""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert engineer.feature_columns == []
    
    def test_create_time_based_features(self, sample_data):
        """Test time-based feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_time_based_features(sample_data)
        
        assert 'account_age_days' in df.columns
        assert 'day_of_week' in df.columns
        assert 'month' in df.columns
        assert 'is_weekend' in df.columns
        
        # Check values are reasonable
        assert (df['account_age_days'] >= 0).all()
        assert (df['day_of_week'] >= 0).all() and (df['day_of_week'] <= 6).all()
        assert df['is_weekend'].isin([0, 1]).all()
    
    def test_create_behavioral_features(self, sample_data):
        """Test behavioral feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_behavioral_features(sample_data)
        
        assert 'total_playtime_hours' in df.columns
        assert 'activity_consistency' in df.columns
        assert 'engagement_score' in df.columns
        
        # Check ranges
        assert (df['activity_consistency'] >= 0).all()
        assert (df['activity_consistency'] <= 1).all()
        assert (df['engagement_score'] >= 0).all()
        assert (df['engagement_score'] <= 1).all()
    
    def test_create_performance_features(self, sample_data):
        """Test performance feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_performance_features(sample_data)
        
        assert 'performance_score' in df.columns
        assert 'skill_improving' in df.columns
        assert 'skill_declining' in df.columns
        
        # Check binary flags
        assert df['skill_improving'].isin([0, 1]).all()
        assert df['skill_declining'].isin([0, 1]).all()
    
    def test_create_social_features(self, sample_data):
        """Test social feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_social_features(sample_data)
        
        assert 'social_engagement' in df.columns
        assert 'low_social_engagement' in df.columns
        assert 'solo_player' in df.columns
        assert 'has_toxicity_issues' in df.columns
    
    def test_create_monetization_features(self, sample_data):
        """Test monetization feature creation"""
        engineer = FeatureEngineer()
        df = engineer.create_monetization_features(sample_data)
        
        assert 'total_spent' in df.columns
        assert 'is_spender' in df.columns
        assert 'is_whale' in df.columns
        assert 'spending_velocity' in df.columns
        
        # Check cumulative spending is monotonic per player
        for player_id in df['player_id'].unique():
            player_df = df[df['player_id'] == player_id].sort_values('date')
            total_spent = player_df['total_spent'].values
            assert (np.diff(total_spent) >= -0.01).all()  # Allow small float errors
    
    def test_create_risk_indicators(self, sample_data):
        """Test risk indicator creation"""
        engineer = FeatureEngineer()
        
        # First engineer features needed for risk indicators
        df = engineer.create_behavioral_features(sample_data)
        df = engineer.create_monetization_features(df)
        df = engineer.create_social_features(df)
        df = engineer.create_risk_indicators(df)
        
        assert 'activity_risk' in df.columns
        assert 'engagement_risk' in df.columns
        assert 'monetization_risk' in df.columns
        assert 'overall_risk_score' in df.columns
        
        # Check ranges
        assert (df['overall_risk_score'] >= 0).all()
        assert (df['overall_risk_score'] <= 1).all()
    
    def test_engineer_all_features(self, sample_data):
        """Test complete feature engineering pipeline"""
        engineer = FeatureEngineer()
        df = engineer.engineer_all_features(sample_data)
        
        # Check that new features were added
        assert len(df.columns) > len(sample_data.columns)
        
        # Check for NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_counts = df[numeric_cols].isna().sum()
        
        # Some NaN is okay for rolling features at the start
        assert nan_counts.max() < len(df) * 0.5
    
    def test_prepare_model_data(self, sample_data):
        """Test model data preparation"""
        engineer = FeatureEngineer()
        df = engineer.engineer_all_features(sample_data)
        
        df_prepared, feature_cols = engineer.prepare_model_data(df)
        
        # Should have one row per player (latest data)
        assert len(df_prepared) == df['player_id'].nunique()
        
        # Should have feature columns
        assert len(feature_cols) > 0
        
        # No NaN or Inf values
        assert not df_prepared[feature_cols].isna().any().any()
        assert not np.isinf(df_prepared[feature_cols]).any().any()
    
    def test_get_feature_columns(self, sample_data):
        """Test feature column extraction"""
        engineer = FeatureEngineer()
        df = engineer.engineer_all_features(sample_data)
        
        feature_cols = engineer.get_feature_columns(df)
        
        # Should not include ID or target columns
        excluded = ['player_id', 'date', 'archetype', 'churned']
        for col in excluded:
            if col in df.columns:
                assert col not in feature_cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
