"""
Synthetic Player Data Generator
Generates realistic player behavior data for churn prediction modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List
import json

class PlayerDataGenerator:
    """Generate synthetic player data with realistic patterns"""
    
    def __init__(self, n_players: int = 100000, n_months: int = 6, seed: int = 42):
        self.n_players = n_players
        self.n_months = n_months
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Player archetypes for realistic patterns
        self.archetypes = {
            'hardcore': 0.15,      # High engagement, unlikely to churn
            'casual': 0.35,        # Moderate engagement
            'whale': 0.05,         # High spenders
            'social': 0.20,        # Friend-driven players
            'at_risk': 0.25        # Declining engagement
        }
        
    def generate_player_profiles(self) -> pd.DataFrame:
        """Generate base player profiles"""
        player_ids = [f"PLAYER_{i:08d}" for i in range(self.n_players)]
        
        # Assign archetypes
        archetype_choices = np.random.choice(
            list(self.archetypes.keys()),
            size=self.n_players,
            p=list(self.archetypes.values())
        )
        
        # Registration dates (over past 2 years)
        start_date = datetime.now() - timedelta(days=730)
        reg_dates = [
            start_date + timedelta(days=random.randint(0, 730))
            for _ in range(self.n_players)
        ]
        
        profiles = pd.DataFrame({
            'player_id': player_ids,
            'archetype': archetype_choices,
            'registration_date': reg_dates,
            'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'JP', 'KR', 'BR'], 
                                       size=self.n_players),
            'platform': np.random.choice(['PC', 'Console', 'Mobile'], 
                                        size=self.n_players, p=[0.5, 0.3, 0.2])
        })
        
        return profiles
    
    def generate_behavioral_features(self, profiles: pd.DataFrame) -> pd.DataFrame:
        """Generate behavioral metrics for each player"""
        
        data = []
        end_date = datetime.now()
        
        for _, player in profiles.iterrows():
            archetype = player['archetype']
            
            # Base parameters by archetype
            params = self._get_archetype_params(archetype)
            
            # Generate 6 months of data
            for month in range(self.n_months):
                current_date = end_date - timedelta(days=30 * (self.n_months - month - 1))
                
                # Add decay for at-risk players
                decay_factor = 1.0
                if archetype == 'at_risk':
                    decay_factor = 1.0 - (month * 0.15)  # 15% decline per month
                
                # Sessions per week
                sessions_per_week = max(0, np.random.poisson(
                    params['base_sessions'] * decay_factor
                ))
                
                # Average session length (minutes)
                avg_session_length = max(5, np.random.normal(
                    params['avg_session_length'] * decay_factor, 
                    params['avg_session_length'] * 0.3
                ))
                
                # Days since last login
                if sessions_per_week > 0:
                    days_since_last_login = random.randint(0, 7 // max(1, sessions_per_week))
                else:
                    days_since_last_login = random.randint(20, 60)
                
                # Session decay rate
                if month > 0:
                    prev_sessions = data[-1]['sessions_per_week'] if len(data) > 0 and data[-1]['player_id'] == player['player_id'] else sessions_per_week
                    session_decay_rate = (prev_sessions - sessions_per_week) / max(1, prev_sessions)
                else:
                    session_decay_rate = 0
                
                data.append({
                    'player_id': player['player_id'],
                    'date': current_date,
                    'month': month + 1,
                    'archetype': archetype,
                    'sessions_per_week': sessions_per_week,
                    'avg_session_length_mins': round(avg_session_length, 2),
                    'days_since_last_login': days_since_last_login,
                    'session_decay_rate': round(session_decay_rate, 4)
                })
        
        return pd.DataFrame(data)
    
    def generate_performance_features(self, behavioral_df: pd.DataFrame) -> pd.DataFrame:
        """Generate performance metrics"""
        
        performance_data = []
        
        for _, row in behavioral_df.iterrows():
            archetype = row['archetype']
            params = self._get_archetype_params(archetype)
            
            # Win rate
            win_rate = np.clip(np.random.normal(
                params['base_win_rate'], 0.1
            ), 0, 1)
            
            # K/D ratio
            kd_ratio = max(0.1, np.random.gamma(
                params['kd_shape'], params['kd_scale']
            ))
            
            # Rank (1-100, higher is better)
            rank = int(np.clip(np.random.normal(
                params['avg_rank'], 15
            ), 1, 100))
            
            # Rank change from previous month
            rank_change = random.randint(-10, 10)
            if archetype == 'at_risk':
                rank_change = random.randint(-15, 0)  # Declining performance
            elif archetype == 'hardcore':
                rank_change = random.randint(-5, 15)  # Generally improving
            
            performance_data.append({
                'player_id': row['player_id'],
                'date': row['date'],
                'win_rate': round(win_rate, 4),
                'kd_ratio': round(kd_ratio, 2),
                'current_rank': rank,
                'rank_change': rank_change
            })
        
        return pd.DataFrame(performance_data)
    
    def generate_social_features(self, behavioral_df: pd.DataFrame) -> pd.DataFrame:
        """Generate social interaction metrics"""
        
        social_data = []
        
        for _, row in behavioral_df.iterrows():
            archetype = row['archetype']
            params = self._get_archetype_params(archetype)
            
            # Friends online
            friends_online = int(np.random.poisson(params['friends_online']))
            
            # Party play percentage
            party_play_pct = np.clip(np.random.beta(
                params['party_alpha'], params['party_beta']
            ), 0, 1)
            
            # Toxicity reports
            toxicity_reports = np.random.poisson(0.1) if random.random() > 0.9 else 0
            
            social_data.append({
                'player_id': row['player_id'],
                'date': row['date'],
                'friends_online': friends_online,
                'party_play_percentage': round(party_play_pct, 4),
                'toxicity_reports_received': toxicity_reports
            })
        
        return pd.DataFrame(social_data)
    
    def generate_monetization_features(self, behavioral_df: pd.DataFrame) -> pd.DataFrame:
        """Generate monetization metrics"""
        
        monetization_data = []
        
        for _, row in behavioral_df.iterrows():
            archetype = row['archetype']
            params = self._get_archetype_params(archetype)
            
            # Monthly spending
            if random.random() < params['purchase_probability']:
                monthly_spent = np.random.gamma(
                    params['spend_shape'], params['spend_scale']
                )
            else:
                monthly_spent = 0
            
            # Days since last purchase
            if monthly_spent > 0:
                days_since_purchase = random.randint(0, 30)
            else:
                days_since_purchase = random.randint(30, 180)
            
            # Purchase frequency (purchases per month)
            purchase_frequency = np.random.poisson(params['purchase_frequency'])
            
            monetization_data.append({
                'player_id': row['player_id'],
                'date': row['date'],
                'monthly_spent': round(monthly_spent, 2),
                'days_since_last_purchase': days_since_purchase,
                'purchase_frequency': purchase_frequency
            })
        
        return pd.DataFrame(monetization_data)
    
    def generate_engagement_features(self, behavioral_df: pd.DataFrame) -> pd.DataFrame:
        """Generate engagement metrics"""
        
        engagement_data = []
        
        for _, row in behavioral_df.iterrows():
            archetype = row['archetype']
            params = self._get_archetype_params(archetype)
            
            # Achievements unlocked (cumulative)
            achievements_unlocked = int(np.random.poisson(
                params['achievements_rate'] * (row['month'])
            ))
            
            # Content completion percentage
            content_completion = np.clip(np.random.beta(
                params['completion_alpha'], params['completion_beta']
            ), 0, 1)
            
            # Feature adoption (% of available features used)
            feature_adoption = np.clip(np.random.beta(
                params['adoption_alpha'], params['adoption_beta']
            ), 0, 1)
            
            engagement_data.append({
                'player_id': row['player_id'],
                'date': row['date'],
                'achievements_unlocked': achievements_unlocked,
                'content_completion_pct': round(content_completion, 4),
                'feature_adoption_pct': round(feature_adoption, 4)
            })
        
        return pd.DataFrame(engagement_data)
    
    def generate_churn_labels(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Generate churn labels based on player behavior"""
        
        # Get the most recent month for each player
        latest_data = final_df.sort_values('date').groupby('player_id').last().reset_index()
        
        churn_labels = []
        
        for _, player in latest_data.iterrows():
            # Churn probability based on multiple factors
            churn_prob = 0.0
            
            # Behavioral factors
            if player['sessions_per_week'] < 1:
                churn_prob += 0.3
            if player['days_since_last_login'] > 14:
                churn_prob += 0.3
            if player['session_decay_rate'] > 0.5:
                churn_prob += 0.2
            
            # Social factors
            if player['friends_online'] < 2:
                churn_prob += 0.1
            
            # Monetization factors
            if player['days_since_last_purchase'] > 90:
                churn_prob += 0.1
            
            # Archetype influence
            if player['archetype'] == 'at_risk':
                churn_prob += 0.4
            elif player['archetype'] == 'hardcore':
                churn_prob -= 0.3
            
            churn_prob = np.clip(churn_prob, 0, 1)
            
            # Determine churn
            churned = 1 if random.random() < churn_prob else 0
            
            churn_labels.append({
                'player_id': player['player_id'],
                'churned': churned,
                'churn_probability': round(churn_prob, 4)
            })
        
        return pd.DataFrame(churn_labels)
    
    def _get_archetype_params(self, archetype: str) -> Dict:
        """Get parameters for each player archetype"""
        
        params = {
            'hardcore': {
                'base_sessions': 15,
                'avg_session_length': 120,
                'base_win_rate': 0.55,
                'kd_shape': 3, 'kd_scale': 0.5,
                'avg_rank': 75,
                'friends_online': 8,
                'party_alpha': 5, 'party_beta': 2,
                'purchase_probability': 0.7,
                'spend_shape': 3, 'spend_scale': 15,
                'purchase_frequency': 2,
                'achievements_rate': 10,
                'completion_alpha': 5, 'completion_beta': 2,
                'adoption_alpha': 5, 'adoption_beta': 2
            },
            'casual': {
                'base_sessions': 5,
                'avg_session_length': 45,
                'base_win_rate': 0.48,
                'kd_shape': 2, 'kd_scale': 0.5,
                'avg_rank': 45,
                'friends_online': 3,
                'party_alpha': 2, 'party_beta': 3,
                'purchase_probability': 0.3,
                'spend_shape': 2, 'spend_scale': 8,
                'purchase_frequency': 0.5,
                'achievements_rate': 3,
                'completion_alpha': 2, 'completion_beta': 4,
                'adoption_alpha': 2, 'adoption_beta': 4
            },
            'whale': {
                'base_sessions': 10,
                'avg_session_length': 90,
                'base_win_rate': 0.50,
                'kd_shape': 2.5, 'kd_scale': 0.5,
                'avg_rank': 60,
                'friends_online': 5,
                'party_alpha': 3, 'party_beta': 3,
                'purchase_probability': 0.95,
                'spend_shape': 5, 'spend_scale': 40,
                'purchase_frequency': 5,
                'achievements_rate': 8,
                'completion_alpha': 4, 'completion_beta': 2,
                'adoption_alpha': 4, 'adoption_beta': 2
            },
            'social': {
                'base_sessions': 8,
                'avg_session_length': 75,
                'base_win_rate': 0.50,
                'kd_shape': 2, 'kd_scale': 0.5,
                'avg_rank': 50,
                'friends_online': 12,
                'party_alpha': 6, 'party_beta': 1,
                'purchase_probability': 0.5,
                'spend_shape': 2, 'spend_scale': 12,
                'purchase_frequency': 1,
                'achievements_rate': 5,
                'completion_alpha': 3, 'completion_beta': 3,
                'adoption_alpha': 3, 'adoption_beta': 3
            },
            'at_risk': {
                'base_sessions': 8,  # Starts normal but decays
                'avg_session_length': 60,
                'base_win_rate': 0.42,
                'kd_shape': 1.5, 'kd_scale': 0.5,
                'avg_rank': 35,
                'friends_online': 2,
                'party_alpha': 1, 'party_beta': 5,
                'purchase_probability': 0.2,
                'spend_shape': 1.5, 'spend_scale': 5,
                'purchase_frequency': 0.3,
                'achievements_rate': 2,
                'completion_alpha': 1, 'completion_beta': 5,
                'adoption_alpha': 1, 'adoption_beta': 5
            }
        }
        
        return params[archetype]
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate complete player dataset with all features"""
        
        print("Generating player profiles...")
        profiles = self.generate_player_profiles()
        
        print("Generating behavioral features...")
        behavioral = self.generate_behavioral_features(profiles)
        
        print("Generating performance features...")
        performance = self.generate_performance_features(behavioral)
        
        print("Generating social features...")
        social = self.generate_social_features(behavioral)
        
        print("Generating monetization features...")
        monetization = self.generate_monetization_features(behavioral)
        
        print("Generating engagement features...")
        engagement = self.generate_engagement_features(behavioral)
        
        # Merge all features
        print("Merging all features...")
        final_df = behavioral.merge(performance, on=['player_id', 'date'])
        final_df = final_df.merge(social, on=['player_id', 'date'])
        final_df = final_df.merge(monetization, on=['player_id', 'date'])
        final_df = final_df.merge(engagement, on=['player_id', 'date'])
        
        # Add player profile info
        final_df = final_df.merge(
            profiles[['player_id', 'registration_date', 'country', 'platform']], 
            on='player_id'
        )
        
        # Generate churn labels
        print("Generating churn labels...")
        churn_labels = self.generate_churn_labels(final_df)
        
        # Add churn labels to final dataset
        final_df = final_df.merge(churn_labels, on='player_id', how='left')
        
        print(f"Generated {len(final_df)} records for {self.n_players} players")
        
        return final_df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str = 'player_data.csv'):
        """Save dataset to CSV"""
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        
        # Save summary statistics
        summary = {
            'total_players': df['player_id'].nunique(),
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'churn_rate': df.groupby('player_id')['churned'].last().mean(),
            'archetype_distribution': df.groupby('player_id')['archetype'].first().value_counts().to_dict()
        }
        
        with open(output_path.replace('.csv', '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary statistics saved")
        return summary


if __name__ == "__main__":
    # Generate dataset
    generator = PlayerDataGenerator(n_players=100000, n_months=6)
    df = generator.generate_complete_dataset()
    
    # Save to data folder
    summary = generator.save_dataset(df, 'data/raw/player_data.csv')
    
    print("\n=== Dataset Summary ===")
    print(f"Total Players: {summary['total_players']:,}")
    print(f"Total Records: {summary['total_records']:,}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Overall Churn Rate: {summary['churn_rate']:.2%}")
    print("\nArchetype Distribution:")
    for archetype, count in summary['archetype_distribution'].items():
        print(f"  {archetype}: {count:,} ({count/summary['total_players']:.1%})")
