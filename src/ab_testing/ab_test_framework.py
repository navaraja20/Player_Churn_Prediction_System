"""
A/B Testing Framework
Simulates and analyzes retention interventions
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestSimulator:
    """Simulate A/B tests for retention interventions"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def assign_test_groups(
        self,
        df: pd.DataFrame,
        test_name: str,
        control_size: float = 0.5
    ) -> pd.DataFrame:
        """
        Assign players to control and treatment groups
        
        Args:
            df: Player DataFrame
            test_name: Name of the test
            control_size: Proportion in control group (0-1)
            
        Returns:
            DataFrame with group assignments
        """
        df = df.copy()
        
        # Random assignment
        df[f'{test_name}_group'] = np.random.choice(
            ['control', 'treatment'],
            size=len(df),
            p=[control_size, 1 - control_size]
        )
        
        logger.info(f"Assigned {len(df)} players to {test_name}")
        logger.info(f"  Control: {(df[f'{test_name}_group'] == 'control').sum()}")
        logger.info(f"  Treatment: {(df[f'{test_name}_group'] == 'treatment').sum()}")
        
        return df
    
    def simulate_intervention_effect(
        self,
        df: pd.DataFrame,
        test_name: str,
        intervention_type: str,
        effect_size: float = 0.15
    ) -> pd.DataFrame:
        """
        Simulate effect of an intervention on churn probability
        
        Args:
            df: Player DataFrame with group assignments
            test_name: Name of the test
            intervention_type: Type of intervention
            effect_size: Expected reduction in churn probability
            
        Returns:
            DataFrame with simulated outcomes
        """
        df = df.copy()
        
        group_col = f'{test_name}_group'
        
        if group_col not in df.columns:
            raise ValueError(f"No group assignment found for test: {test_name}")
        
        # Base churn probability (from model or historical)
        base_churn = df.get('churn_probability', df.get('churned', 0.25))
        
        # Apply intervention effect to treatment group
        treatment_mask = df[group_col] == 'treatment'
        
        # Different interventions have different effectiveness
        intervention_multipliers = {
            'discount_offer': 0.85,      # 15% reduction
            'engagement_campaign': 0.90,  # 10% reduction
            'social_nudge': 0.92,        # 8% reduction
            'content_unlock': 0.88,      # 12% reduction
            'personalized_message': 0.93  # 7% reduction
        }
        
        multiplier = intervention_multipliers.get(intervention_type, 1 - effect_size)
        
        # Simulate outcomes
        df['simulated_churn_prob'] = base_churn.copy()
        df.loc[treatment_mask, 'simulated_churn_prob'] *= multiplier
        
        # Add noise
        noise = np.random.normal(0, 0.02, len(df))
        df['simulated_churn_prob'] = (df['simulated_churn_prob'] + noise).clip(0, 1)
        
        # Simulate actual churn based on probability
        df['simulated_churned'] = (
            np.random.random(len(df)) < df['simulated_churn_prob']
        ).astype(int)
        
        logger.info(f"Simulated {intervention_type} intervention")
        logger.info(f"  Control churn rate: {df[~treatment_mask]['simulated_churned'].mean():.2%}")
        logger.info(f"  Treatment churn rate: {df[treatment_mask]['simulated_churned'].mean():.2%}")
        
        return df
    
    def calculate_retention_metrics(
        self,
        df: pd.DataFrame,
        test_name: str,
        retention_period_days: int = 30
    ) -> Dict:
        """
        Calculate retention metrics for both groups
        
        Args:
            df: DataFrame with test results
            test_name: Name of the test
            retention_period_days: Period to measure (7, 30, etc.)
            
        Returns:
            Dictionary of metrics
        """
        group_col = f'{test_name}_group'
        
        control = df[df[group_col] == 'control']
        treatment = df[df[group_col] == 'treatment']
        
        # Calculate retention (1 - churn)
        control_retained = 1 - control['simulated_churned'].mean()
        treatment_retained = 1 - treatment['simulated_churned'].mean()
        
        # Calculate lift
        absolute_lift = treatment_retained - control_retained
        relative_lift = (treatment_retained / control_retained - 1) if control_retained > 0 else 0
        
        metrics = {
            'test_name': test_name,
            'retention_period_days': retention_period_days,
            'control': {
                'size': len(control),
                'retention_rate': control_retained,
                'churn_rate': control['simulated_churned'].mean()
            },
            'treatment': {
                'size': len(treatment),
                'retention_rate': treatment_retained,
                'churn_rate': treatment['simulated_churned'].mean()
            },
            'lift': {
                'absolute': absolute_lift,
                'relative': relative_lift,
                'relative_pct': relative_lift * 100
            }
        }
        
        return metrics
    
    def statistical_test(
        self,
        df: pd.DataFrame,
        test_name: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform statistical significance test
        
        Args:
            df: DataFrame with test results
            test_name: Name of the test
            alpha: Significance level
            
        Returns:
            Dictionary with statistical test results
        """
        group_col = f'{test_name}_group'
        
        control = df[df[group_col] == 'control']['simulated_churned']
        treatment = df[df[group_col] == 'treatment']['simulated_churned']
        
        # Chi-square test
        contingency_table = pd.crosstab(
            df[group_col],
            df['simulated_churned']
        )
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Two-proportion z-test
        control_success = (control == 0).sum()  # Retained
        treatment_success = (treatment == 0).sum()
        
        control_total = len(control)
        treatment_total = len(treatment)
        
        # Pooled proportion
        pooled_p = (control_success + treatment_success) / (control_total + treatment_total)
        
        # Standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/control_total + 1/treatment_total))
        
        # Z-score
        p_control = control_success / control_total
        p_treatment = treatment_success / treatment_total
        z_score = (p_treatment - p_control) / se if se > 0 else 0
        
        # P-value (two-tailed)
        z_p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval for lift
        ci_95 = 1.96 * se
        
        results = {
            'chi_square': {
                'statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof
            },
            'z_test': {
                'z_score': z_score,
                'p_value': z_p_value,
                'significant': z_p_value < alpha
            },
            'confidence_interval_95': {
                'lower': (p_treatment - p_control) - ci_95,
                'upper': (p_treatment - p_control) + ci_95
            },
            'conclusion': 'SIGNIFICANT' if z_p_value < alpha else 'NOT SIGNIFICANT',
            'alpha': alpha
        }
        
        logger.info(f"Statistical test results for {test_name}:")
        logger.info(f"  Chi-square p-value: {p_value:.4f}")
        logger.info(f"  Z-test p-value: {z_p_value:.4f}")
        logger.info(f"  Result: {results['conclusion']} at α={alpha}")
        
        return results
    
    def calculate_sample_size(
        self,
        baseline_rate: float,
        mde: float,  # Minimum detectable effect
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for A/B test
        
        Args:
            baseline_rate: Current retention/churn rate
            mde: Minimum detectable effect (absolute)
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Required sample size per group
        """
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Pooled standard deviation
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_pool = (p1 + p2) / 2
        
        # Sample size formula
        n = (
            (z_alpha + z_beta) ** 2 * 
            (p1 * (1 - p1) + p2 * (1 - p2))
        ) / (mde ** 2)
        
        return int(np.ceil(n))
    
    def run_ab_test(
        self,
        df: pd.DataFrame,
        test_name: str,
        intervention_type: str,
        effect_size: float = 0.15,
        control_size: float = 0.5,
        retention_period: int = 30
    ) -> Dict:
        """
        Run complete A/B test simulation
        
        Args:
            df: Player DataFrame
            test_name: Name of the test
            intervention_type: Type of intervention
            effect_size: Expected effect size
            control_size: Proportion in control group
            retention_period: Days to measure retention
            
        Returns:
            Complete test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running A/B Test: {test_name}")
        logger.info(f"Intervention: {intervention_type}")
        logger.info(f"{'='*60}\n")
        
        # Assign groups
        df = self.assign_test_groups(df, test_name, control_size)
        
        # Simulate intervention
        df = self.simulate_intervention_effect(df, test_name, intervention_type, effect_size)
        
        # Calculate metrics
        metrics = self.calculate_retention_metrics(df, test_name, retention_period)
        
        # Statistical test
        stat_results = self.statistical_test(df, test_name)
        
        # Combine results
        results = {
            'test_config': {
                'test_name': test_name,
                'intervention_type': intervention_type,
                'expected_effect_size': effect_size,
                'control_size': control_size,
                'retention_period_days': retention_period,
                'timestamp': datetime.now().isoformat()
            },
            'metrics': metrics,
            'statistical_test': stat_results,
            'data': df
        }
        
        return results
    
    def calculate_roi(
        self,
        test_results: Dict,
        intervention_cost: float,
        player_ltv: float
    ) -> Dict:
        """
        Calculate ROI of intervention
        
        Args:
            test_results: Results from run_ab_test
            intervention_cost: Cost per player to apply intervention
            player_ltv: Lifetime value of retained player
            
        Returns:
            ROI analysis
        """
        metrics = test_results['metrics']
        
        treatment_size = metrics['treatment']['size']
        absolute_lift = metrics['lift']['absolute']
        
        # Players saved from churning
        players_saved = treatment_size * absolute_lift
        
        # Revenue retained
        revenue_retained = players_saved * player_ltv
        
        # Cost of intervention
        total_cost = treatment_size * intervention_cost
        
        # Net benefit
        net_benefit = revenue_retained - total_cost
        
        # ROI
        roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
        
        roi_analysis = {
            'intervention_cost': intervention_cost,
            'player_ltv': player_ltv,
            'treatment_group_size': treatment_size,
            'players_saved': players_saved,
            'revenue_retained': revenue_retained,
            'total_cost': total_cost,
            'net_benefit': net_benefit,
            'roi_percentage': roi,
            'cost_per_retained_player': total_cost / players_saved if players_saved > 0 else 0
        }
        
        logger.info(f"\nROI Analysis:")
        logger.info(f"  Players Saved: {players_saved:.0f}")
        logger.info(f"  Revenue Retained: ${revenue_retained:,.2f}")
        logger.info(f"  Total Cost: ${total_cost:,.2f}")
        logger.info(f"  Net Benefit: ${net_benefit:,.2f}")
        logger.info(f"  ROI: {roi:.1f}%")
        
        return roi_analysis


if __name__ == "__main__":
    print("A/B Testing Framework initialized")
