# Model Card: Player Churn Prediction

## Model Details

### Basic Information
- **Model Name**: Player Churn Prediction Ensemble
- **Model Version**: 1.0.0
- **Model Date**: December 2024
- **Model Type**: Binary Classification (Churn / No Churn)
- **Framework**: XGBoost, Random Forest, LightGBM (Ensemble)

### Intended Use
- **Primary Use**: Predict probability of player churn within next 30 days
- **Target Users**: Game operations team, product managers, marketing teams
- **Out-of-Scope Uses**: 
  - Real-money gambling predictions
  - Individual player psychological profiling
  - Automated player banning decisions

## Training Data

### Dataset
- **Size**: 100,000 players, 600,000 total records (6 months history)
- **Source**: Synthetic data generator (based on gaming industry patterns)
- **Time Period**: 6 months of player activity data
- **Features**: 40+ engineered features across 5 categories

### Feature Categories

#### 1. Behavioral Features (30%)
- Sessions per week
- Average session length
- Days since last login
- Session decay rate
- Activity consistency

#### 2. Performance Features (20%)
- Win rate
- K/D ratio
- Current rank
- Rank progression
- Performance score

#### 3. Social Features (15%)
- Friends online
- Party play percentage
- Social engagement score
- Toxicity reports

#### 4. Monetization Features (20%)
- Total spent
- Purchase frequency
- Days since last purchase
- Spending velocity
- Player LTV

#### 5. Engagement Features (15%)
- Achievements unlocked
- Content completion
- Feature adoption
- Content engagement score

### Class Distribution
- **Churned**: ~25% (varies by player segment)
- **Retained**: ~75%
- **Balancing**: Used class weights in training

## Model Performance

### Overall Metrics (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.87 | 0.82 | 0.79 | 0.80 | 0.92 |
| Random Forest | 0.85 | 0.80 | 0.77 | 0.78 | 0.90 |
| LightGBM | 0.86 | 0.81 | 0.78 | 0.79 | 0.91 |
| **Ensemble** | **0.88** | **0.84** | **0.81** | **0.82** | **0.93** |

### Performance by Player Segment

#### High-Value Players (Top 20% spenders)
- Precision: 0.91
- Recall: 0.85
- F1-Score: 0.88
- **Note**: Higher precision to avoid false positives on valuable players

#### Casual Players (Low engagement)
- Precision: 0.78
- Recall: 0.83
- F1-Score: 0.80
- **Note**: Higher recall to catch at-risk casual players

#### Social Players (High friend activity)
- Precision: 0.82
- Recall: 0.79
- F1-Score: 0.80

### Performance by Platform
| Platform | ROC-AUC | F1-Score | Sample Size |
|----------|---------|----------|-------------|
| PC | 0.93 | 0.83 | 50,000 |
| Console | 0.92 | 0.81 | 30,000 |
| Mobile | 0.91 | 0.80 | 20,000 |

## Feature Importance

### Top 10 Most Important Features
1. **days_since_last_login** (12.5%) - Strong indicator of disengagement
2. **session_decay_rate** (10.8%) - Declining engagement pattern
3. **overall_risk_score** (9.2%) - Composite risk indicator
4. **sessions_per_week** (8.7%) - Activity level
5. **social_engagement** (7.5%) - Social connection strength
6. **spending_velocity** (6.9%) - Monetization behavior
7. **engagement_score** (6.3%) - Content engagement
8. **activity_consistency** (5.8%) - Behavioral stability
9. **performance_score** (5.2%) - Player skill/success
10. **friends_online** (4.9%) - Social network size

## Evaluation Metrics

### Business Impact Metrics
- **Cost of False Negative**: $300 (lost player LTV)
- **Cost of False Positive**: $5 (wasted intervention cost)
- **Optimal Threshold**: 0.45 (weighted by business costs)

### Model Calibration
- **Brier Score**: 0.12 (well-calibrated probabilities)
- **Expected Calibration Error**: 0.03

### Fairness Metrics
- **Demographic Parity**: 0.92 (across platforms)
- **Equal Opportunity**: 0.89 (across player types)

## Limitations

### Known Limitations
1. **New Player Cold Start**: Limited predictive power for players with <1 month history
2. **Seasonal Events**: May not capture irregular events (holidays, major updates)
3. **External Factors**: Cannot predict churn from external factors (competitor releases)
4. **Data Lag**: Uses historical data; may miss sudden behavioral changes
5. **Synthetic Data**: Model trained on simulated data; requires retraining on real data

### Edge Cases
- Players returning from long breaks may be incorrectly flagged
- Streamers/content creators have unusual patterns
- Beta testers may show different engagement patterns

## Ethical Considerations

### Privacy
- No personally identifiable information used
- Aggregated behavioral metrics only
- Compliant with GDPR/CCPA requirements

### Fairness
- Model evaluated across different player segments
- No discrimination based on protected attributes
- Equal treatment of free and paying players

### Transparency
- SHAP values provided for each prediction
- Feature importance available to operations team
- Clear explanation of risk factors

## Recommendations

### Deployment Guidelines
1. **Monitoring**: Track prediction distribution and model drift weekly
2. **Retraining**: Retrain model monthly with fresh data
3. **Threshold Tuning**: Adjust decision threshold based on campaign costs
4. **A/B Testing**: Validate interventions before full rollout

### Intervention Strategy
- **High Risk (>0.7)**: Immediate personalized intervention
- **Medium Risk (0.3-0.7)**: Automated engagement campaigns
- **Low Risk (<0.3)**: Standard retention marketing

### Model Updates
- **Next Version**: Incorporate time-series models for trajectory prediction
- **Planned Features**: 
  - Community toxicity metrics
  - Cross-game engagement patterns
  - Real-time event participation

## Contact

- **Model Owner**: Data Science Team
- **Technical Contact**: ml-team@company.com
- **Last Updated**: December 2024
- **Review Cycle**: Quarterly

## References

1. XGBoost Documentation: https://xgboost.readthedocs.io/
2. SHAP Documentation: https://shap.readthedocs.io/
3. Gaming Industry Churn Benchmarks (2024)
4. Responsible AI Guidelines for Gaming Applications
