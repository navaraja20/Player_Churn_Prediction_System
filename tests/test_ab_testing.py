"""
Unit tests for A/B Testing Framework
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ab_testing.ab_test_framework import ABTestSimulator


@pytest.fixture
def sample_players():
    """Create sample player data for testing"""
    np.random.seed(42)
    
    data = []
    for i in range(1000):
        data.append({
            'player_id': f'PLAYER_{i:04d}',
            'churn_probability': np.random.uniform(0.1, 0.8),
            'sessions_per_week': np.random.randint(1, 20),
            'monthly_spent': np.random.uniform(0, 100)
        })
    
    return pd.DataFrame(data)


class TestABTestSimulator:
    """Test suite for ABTestSimulator"""
    
    def test_initialization(self):
        """Test ABTestSimulator initialization"""
        simulator = ABTestSimulator(random_state=42)
        assert simulator is not None
        assert simulator.random_state == 42
    
    def test_assign_test_groups(self, sample_players):
        """Test group assignment"""
        simulator = ABTestSimulator(random_state=42)
        df = simulator.assign_test_groups(
            sample_players,
            test_name='test1',
            control_size=0.5
        )
        
        assert 'test1_group' in df.columns
        
        # Check group distribution
        control_count = (df['test1_group'] == 'control').sum()
        treatment_count = (df['test1_group'] == 'treatment').sum()
        
        assert control_count > 0
        assert treatment_count > 0
        assert control_count + treatment_count == len(df)
        
        # Should be roughly 50/50
        control_ratio = control_count / len(df)
        assert 0.45 < control_ratio < 0.55
    
    def test_simulate_intervention_effect(self, sample_players):
        """Test intervention effect simulation"""
        simulator = ABTestSimulator(random_state=42)
        df = simulator.assign_test_groups(sample_players, 'test1')
        df = simulator.simulate_intervention_effect(
            df,
            test_name='test1',
            intervention_type='discount_offer',
            effect_size=0.15
        )
        
        assert 'simulated_churn_prob' in df.columns
        assert 'simulated_churned' in df.columns
        
        # Treatment group should have lower churn probability
        control = df[df['test1_group'] == 'control']
        treatment = df[df['test1_group'] == 'treatment']
        
        control_churn_rate = control['simulated_churned'].mean()
        treatment_churn_rate = treatment['simulated_churned'].mean()
        
        # Treatment should generally have lower churn
        assert treatment_churn_rate <= control_churn_rate * 1.1  # Allow some variance
    
    def test_calculate_retention_metrics(self, sample_players):
        """Test retention metrics calculation"""
        simulator = ABTestSimulator(random_state=42)
        df = simulator.assign_test_groups(sample_players, 'test1')
        df = simulator.simulate_intervention_effect(
            df,
            test_name='test1',
            intervention_type='discount_offer'
        )
        
        metrics = simulator.calculate_retention_metrics(df, 'test1', 30)
        
        assert 'test_name' in metrics
        assert 'control' in metrics
        assert 'treatment' in metrics
        assert 'lift' in metrics
        
        # Check metric structure
        assert 'retention_rate' in metrics['control']
        assert 'retention_rate' in metrics['treatment']
        assert 'absolute' in metrics['lift']
        assert 'relative' in metrics['lift']
    
    def test_statistical_test(self, sample_players):
        """Test statistical significance testing"""
        simulator = ABTestSimulator(random_state=42)
        df = simulator.assign_test_groups(sample_players, 'test1')
        df = simulator.simulate_intervention_effect(
            df,
            test_name='test1',
            intervention_type='discount_offer'
        )
        
        results = simulator.statistical_test(df, 'test1', alpha=0.05)
        
        assert 'chi_square' in results
        assert 'z_test' in results
        assert 'confidence_interval_95' in results
        assert 'conclusion' in results
        
        # Check result structure
        assert 'p_value' in results['chi_square']
        assert 'p_value' in results['z_test']
        assert results['conclusion'] in ['SIGNIFICANT', 'NOT SIGNIFICANT']
    
    def test_calculate_sample_size(self):
        """Test sample size calculation"""
        simulator = ABTestSimulator()
        
        sample_size = simulator.calculate_sample_size(
            baseline_rate=0.70,
            mde=0.05,
            alpha=0.05,
            power=0.8
        )
        
        assert sample_size > 0
        assert isinstance(sample_size, int)
        
        # Larger effect should require smaller sample
        larger_effect_size = simulator.calculate_sample_size(
            baseline_rate=0.70,
            mde=0.10,
            alpha=0.05,
            power=0.8
        )
        
        assert larger_effect_size < sample_size
    
    def test_run_ab_test(self, sample_players):
        """Test complete A/B test run"""
        simulator = ABTestSimulator(random_state=42)
        
        results = simulator.run_ab_test(
            sample_players,
            test_name='test_campaign',
            intervention_type='discount_offer',
            effect_size=0.15,
            control_size=0.5,
            retention_period=30
        )
        
        assert 'test_config' in results
        assert 'metrics' in results
        assert 'statistical_test' in results
        assert 'data' in results
        
        # Check that data has group assignments
        assert 'test_campaign_group' in results['data'].columns
    
    def test_calculate_roi(self, sample_players):
        """Test ROI calculation"""
        simulator = ABTestSimulator(random_state=42)
        
        results = simulator.run_ab_test(
            sample_players,
            test_name='test1',
            intervention_type='discount_offer',
            effect_size=0.15
        )
        
        roi_analysis = simulator.calculate_roi(
            results,
            intervention_cost=5.0,
            player_ltv=300.0
        )
        
        assert 'intervention_cost' in roi_analysis
        assert 'player_ltv' in roi_analysis
        assert 'players_saved' in roi_analysis
        assert 'revenue_retained' in roi_analysis
        assert 'total_cost' in roi_analysis
        assert 'net_benefit' in roi_analysis
        assert 'roi_percentage' in roi_analysis
        
        # ROI should be positive for effective intervention
        # (though this depends on costs and LTV)
        assert isinstance(roi_analysis['roi_percentage'], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
