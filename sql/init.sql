-- PostgreSQL Initialization Script
-- Creates tables and indices for Player Churn Prediction System

-- Create database (run separately if needed)
-- CREATE DATABASE churn_db;

-- Connect to database
\c churn_db;

-- Create player_features table
CREATE TABLE IF NOT EXISTS player_features (
    player_id VARCHAR(50) NOT NULL,
    date TIMESTAMP NOT NULL,
    archetype VARCHAR(20),
    
    -- Behavioral features
    sessions_per_week INTEGER,
    avg_session_length_mins REAL,
    days_since_last_login INTEGER,
    session_decay_rate REAL,
    activity_consistency REAL,
    engagement_score REAL,
    total_playtime_hours REAL,
    sessions_trend REAL,
    
    -- Performance features
    win_rate REAL,
    kd_ratio REAL,
    current_rank INTEGER,
    rank_change INTEGER,
    performance_score REAL,
    skill_improving INTEGER,
    skill_declining INTEGER,
    win_rate_std REAL,
    
    -- Social features
    friends_online INTEGER,
    party_play_percentage REAL,
    toxicity_reports_received INTEGER,
    social_engagement REAL,
    low_social_engagement INTEGER,
    solo_player INTEGER,
    has_toxicity_issues INTEGER,
    
    -- Monetization features
    monthly_spent REAL,
    days_since_last_purchase INTEGER,
    purchase_frequency INTEGER,
    total_spent REAL,
    is_spender INTEGER,
    is_whale INTEGER,
    recent_spender INTEGER,
    spending_velocity REAL,
    avg_transaction_value REAL,
    spending_trend REAL,
    
    -- Engagement features
    achievements_unlocked INTEGER,
    content_completion_pct REAL,
    feature_adoption_pct REAL,
    content_engagement REAL,
    achievement_velocity REAL,
    content_exhausted INTEGER,
    
    -- Risk indicators
    activity_risk REAL,
    engagement_risk REAL,
    monetization_risk REAL,
    overall_risk_score REAL,
    
    -- Metadata
    registration_date TIMESTAMP,
    country VARCHAR(5),
    platform VARCHAR(20),
    account_age_days INTEGER,
    day_of_week INTEGER,
    month INTEGER,
    is_weekend INTEGER,
    
    -- Target
    churned INTEGER,
    churn_probability REAL,
    
    -- Constraints
    PRIMARY KEY (player_id, date)
);

-- Create indices
CREATE INDEX idx_player_id ON player_features(player_id);
CREATE INDEX idx_date ON player_features(date);
CREATE INDEX idx_risk_score ON player_features(overall_risk_score);
CREATE INDEX idx_churned ON player_features(churned);
CREATE INDEX idx_platform ON player_features(platform);
CREATE INDEX idx_archetype ON player_features(archetype);

-- Create predictions table
CREATE TABLE IF NOT EXISTS daily_predictions (
    prediction_id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    churn_probability REAL,
    risk_category VARCHAR(10),
    model_version VARCHAR(20),
    top_risk_factors JSONB
);

CREATE INDEX idx_pred_player ON daily_predictions(player_id);
CREATE INDEX idx_pred_date ON daily_predictions(prediction_date);
CREATE INDEX idx_pred_risk ON daily_predictions(risk_category);

-- Create interventions table
CREATE TABLE IF NOT EXISTS interventions (
    intervention_id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    intervention_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    intervention_type VARCHAR(50),
    intervention_cost REAL,
    status VARCHAR(20),
    result VARCHAR(20),
    notes TEXT
);

CREATE INDEX idx_int_player ON interventions(player_id);
CREATE INDEX idx_int_date ON interventions(intervention_date);
CREATE INDEX idx_int_type ON interventions(intervention_type);

-- Create ab_tests table
CREATE TABLE IF NOT EXISTS ab_tests (
    test_id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    intervention_type VARCHAR(50),
    control_size REAL,
    treatment_size REAL,
    control_churn_rate REAL,
    treatment_churn_rate REAL,
    lift_percentage REAL,
    p_value REAL,
    is_significant BOOLEAN,
    status VARCHAR(20),
    results JSONB
);

CREATE INDEX idx_test_name ON ab_tests(test_name);
CREATE INDEX idx_test_date ON ab_tests(start_date);

-- Create monitoring table
CREATE TABLE IF NOT EXISTS model_monitoring (
    monitor_id SERIAL PRIMARY KEY,
    monitor_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(50),
    metric_value REAL,
    threshold_value REAL,
    alert_triggered BOOLEAN,
    notes TEXT
);

CREATE INDEX idx_monitor_date ON model_monitoring(monitor_date);
CREATE INDEX idx_monitor_metric ON model_monitoring(metric_name);

-- Create materialized view for quick analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS player_risk_summary AS
SELECT 
    player_id,
    MAX(date) as last_update,
    MAX(overall_risk_score) as risk_score,
    CASE 
        WHEN MAX(overall_risk_score) < 0.3 THEN 'Low'
        WHEN MAX(overall_risk_score) < 0.7 THEN 'Medium'
        ELSE 'High'
    END as risk_category,
    AVG(sessions_per_week) as avg_sessions,
    AVG(monthly_spent) as avg_spent,
    MAX(total_spent) as total_spent,
    MAX(friends_online) as friends_count,
    MAX(platform) as platform,
    MAX(archetype) as archetype
FROM player_features
GROUP BY player_id;

CREATE INDEX idx_risk_summary_player ON player_risk_summary(player_id);
CREATE INDEX idx_risk_summary_category ON player_risk_summary(risk_category);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW player_risk_summary;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO churn_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO churn_user;

-- Print success message
SELECT 'Database initialization completed successfully!' as status;
