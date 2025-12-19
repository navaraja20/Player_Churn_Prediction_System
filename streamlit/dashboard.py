"""
Streamlit Dashboard for Player Churn Prediction
Interactive dashboard for monitoring and analyzing churn risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ab_testing.ab_test_framework import ABTestSimulator

# Page config
st.set_page_config(
    page_title="Player Churn Prediction Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load player data"""
    try:
        df = pd.read_csv('data/processed/player_features.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the ETL pipeline first.")
        return pd.DataFrame()


@st.cache_data
def load_predictions():
    """Load predictions"""
    try:
        df = pd.read_csv('data/predictions/daily_predictions.csv')
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def calculate_ltv(monthly_spent_avg: float, retention_months: float = 12) -> float:
    """Calculate player lifetime value"""
    return monthly_spent_avg * retention_months


def main():
    """Main dashboard function"""
    
    st.markdown('<p class="main-header">🎮 Player Churn Prediction Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Risk Analysis", "Player Segmentation", "A/B Test Simulator", "Interventions", "ROI Calculator"]
    )
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No data available. Please run the data pipeline first.")
        return
    
    # Get latest data per player
    df_latest = df.sort_values('date').groupby('player_id').last().reset_index()
    
    # Load predictions if available
    df_predictions = load_predictions()
    if not df_predictions.empty:
        df_latest = df_latest.merge(
            df_predictions[['player_id', 'churn_risk_score', 'risk_category']],
            on='player_id',
            how='left'
        )
    else:
        # Use model predictions if available, otherwise use churned column
        if 'churned' in df_latest.columns:
            df_latest['churn_risk_score'] = df_latest['churned']
            df_latest['risk_category'] = pd.cut(
                df_latest['churned'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low', 'Medium', 'High']
            )
    
    # Route to selected page
    if page == "Overview":
        show_overview(df_latest)
    elif page == "Risk Analysis":
        show_risk_analysis(df_latest)
    elif page == "Player Segmentation":
        show_segmentation(df_latest)
    elif page == "A/B Test Simulator":
        show_ab_test_simulator(df_latest)
    elif page == "Interventions":
        show_interventions(df_latest)
    elif page == "ROI Calculator":
        show_roi_calculator(df_latest)


def show_overview(df: pd.DataFrame):
    """Overview dashboard"""
    st.header("📊 Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Players",
            f"{len(df):,}",
            help="Total number of active players"
        )
    
    with col2:
        high_risk = (df['risk_category'] == 'High').sum()
        high_risk_pct = high_risk / len(df) * 100
        st.metric(
            "High Risk Players",
            f"{high_risk:,}",
            f"{high_risk_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        avg_risk = df['churn_risk_score'].mean()
        st.metric(
            "Average Risk Score",
            f"{avg_risk:.2%}",
            delta_color="inverse"
        )
    
    with col4:
        if 'monthly_spent' in df.columns:
            total_revenue = df['monthly_spent'].sum()
            st.metric(
                "Monthly Revenue",
                f"${total_revenue:,.0f}"
            )
    
    st.markdown("---")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Category Distribution")
        risk_dist = df['risk_category'].value_counts().reset_index()
        risk_dist.columns = ['Risk Category', 'Count']
        
        colors = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}
        
        fig = px.pie(
            risk_dist,
            values='Count',
            names='Risk Category',
            color='Risk Category',
            color_discrete_map=colors,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df,
            x='churn_risk_score',
            nbins=50,
            title="",
            labels={'churn_risk_score': 'Churn Risk Score'}
        )
        fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Low/Med")
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Med/High")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn heatmap by archetype and platform
    st.subheader("📈 Churn Risk Heatmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'archetype' in df.columns:
            heatmap_data = df.groupby('archetype')['churn_risk_score'].mean().reset_index()
            heatmap_data = heatmap_data.sort_values('churn_risk_score', ascending=False)
            
            fig = px.bar(
                heatmap_data,
                x='archetype',
                y='churn_risk_score',
                title="Average Risk by Player Type",
                color='churn_risk_score',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'platform' in df.columns:
            platform_risk = df.groupby('platform')['churn_risk_score'].mean().reset_index()
            
            fig = px.bar(
                platform_risk,
                x='platform',
                y='churn_risk_score',
                title="Average Risk by Platform",
                color='churn_risk_score',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)


def show_risk_analysis(df: pd.DataFrame):
    """Risk analysis page"""
    st.header("🔍 Risk Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Category",
            options=['Low', 'Medium', 'High'],
            default=['High', 'Medium']
        )
    
    with col2:
        if 'platform' in df.columns:
            platform_filter = st.multiselect(
                "Filter by Platform",
                options=df['platform'].unique(),
                default=df['platform'].unique()
            )
        else:
            platform_filter = []
    
    with col3:
        top_n = st.slider("Show Top N Players", 10, 100, 20)
    
    # Apply filters
    df_filtered = df[df['risk_category'].isin(risk_filter)]
    if platform_filter and 'platform' in df.columns:
        df_filtered = df_filtered[df_filtered['platform'].isin(platform_filter)]
    
    # High risk players table
    st.subheader(f"Top {top_n} High-Risk Players")
    
    display_cols = ['player_id', 'churn_risk_score', 'risk_category', 
                   'sessions_per_week', 'days_since_last_login', 
                   'monthly_spent', 'friends_online']
    
    # Filter columns that exist
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    
    df_display = df_filtered.nlargest(top_n, 'churn_risk_score')[display_cols]
    
    # Format the dataframe
    st.dataframe(
        df_display.style.background_gradient(subset=['churn_risk_score'], cmap='RdYlGn_r'),
        use_container_width=True
    )
    
    # Risk factor analysis
    st.subheader("🎯 Key Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Behavioral risk factors
        st.markdown("**Behavioral Risk Indicators**")
        
        low_activity = (df_filtered['sessions_per_week'] < 2).sum()
        inactive = (df_filtered['days_since_last_login'] > 14).sum()
        declining = (df_filtered.get('session_decay_rate', 0) > 0.3).sum() if 'session_decay_rate' in df_filtered.columns else 0
        
        risk_factors = pd.DataFrame({
            'Factor': ['Low Activity', 'Inactive (>14 days)', 'Declining Engagement'],
            'Affected Players': [low_activity, inactive, declining],
            'Percentage': [
                low_activity / len(df_filtered) * 100,
                inactive / len(df_filtered) * 100,
                declining / len(df_filtered) * 100
            ]
        })
        
        fig = px.bar(
            risk_factors,
            x='Factor',
            y='Affected Players',
            text='Percentage',
            title=""
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Social risk factors
        st.markdown("**Social Risk Indicators**")
        
        low_social = (df_filtered['friends_online'] < 2).sum() if 'friends_online' in df_filtered.columns else 0
        solo_player = (df_filtered.get('party_play_percentage', 0) < 0.2).sum() if 'party_play_percentage' in df_filtered.columns else 0
        
        social_factors = pd.DataFrame({
            'Factor': ['Low Social Connection', 'Solo Player'],
            'Affected Players': [low_social, solo_player],
            'Percentage': [
                low_social / len(df_filtered) * 100 if len(df_filtered) > 0 else 0,
                solo_player / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            ]
        })
        
        fig = px.bar(
            social_factors,
            x='Factor',
            y='Affected Players',
            text='Percentage',
            title="",
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation with Churn Risk")
    
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    corr_cols = [col for col in numeric_cols if col != 'churn_risk_score'][:15]  # Top 15
    
    if corr_cols:
        correlations = df_filtered[corr_cols + ['churn_risk_score']].corr()['churn_risk_score'].drop('churn_risk_score')
        correlations = correlations.sort_values(ascending=False)
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title="",
            labels={'x': 'Correlation with Churn Risk', 'y': 'Feature'},
            color=correlations.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


def show_segmentation(df: pd.DataFrame):
    """Player segmentation page"""
    st.header("👥 Player Segmentation")
    
    # Segmentation by risk and spending
    st.subheader("Player Segments Matrix")
    
    # Create segments
    df['spending_segment'] = pd.cut(
        df.get('monthly_spent', 0),
        bins=[-0.1, 0, 10, 50, float('inf')],
        labels=['Non-Spender', 'Low Spender', 'Medium Spender', 'High Spender']
    )
    
    # Create pivot table
    segment_matrix = pd.crosstab(
        df['risk_category'],
        df['spending_segment']
    )
    
    fig = px.imshow(
        segment_matrix,
        labels=dict(x="Spending Segment", y="Risk Category", color="Player Count"),
        x=segment_matrix.columns,
        y=segment_matrix.index,
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment profiles
    st.subheader("Segment Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_risk = st.selectbox("Select Risk Category", ['High', 'Medium', 'Low'])
        selected_spending = st.selectbox(
            "Select Spending Segment",
            ['Non-Spender', 'Low Spender', 'Medium Spender', 'High Spender']
        )
    
    # Filter segment
    segment_df = df[
        (df['risk_category'] == selected_risk) &
        (df['spending_segment'] == selected_spending)
    ]
    
    with col2:
        st.metric("Segment Size", f"{len(segment_df):,}")
        st.metric(
            "Percentage of Total",
            f"{len(segment_df) / len(df) * 100:.1f}%"
        )
    
    if len(segment_df) > 0:
        st.subheader(f"Profile: {selected_risk} Risk, {selected_spending}")
        
        # Key characteristics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Sessions/Week",
                f"{segment_df['sessions_per_week'].mean():.1f}"
            )
        
        with col2:
            st.metric(
                "Avg Session Length",
                f"{segment_df.get('avg_session_length_mins', pd.Series([0])).mean():.0f} min"
            )
        
        with col3:
            st.metric(
                "Days Since Login",
                f"{segment_df.get('days_since_last_login', pd.Series([0])).mean():.0f}"
            )
        
        with col4:
            st.metric(
                "Avg Friends Online",
                f"{segment_df.get('friends_online', pd.Series([0])).mean():.1f}"
            )


def show_ab_test_simulator(df: pd.DataFrame):
    """A/B test simulator page"""
    st.header("🧪 A/B Test Simulator")
    
    st.markdown("""
    Simulate retention interventions and measure their impact on churn rates.
    Configure test parameters below to see projected results.
    """)
    
    # Test configuration
    col1, col2 = st.columns(2)
    
    with col1:
        test_name = st.text_input("Test Name", "retention_campaign_q1")
        
        intervention_type = st.selectbox(
            "Intervention Type",
            [
                "discount_offer",
                "engagement_campaign",
                "social_nudge",
                "content_unlock",
                "personalized_message"
            ]
        )
        
        effect_size = st.slider(
            "Expected Effect Size",
            0.05, 0.30, 0.15,
            help="Expected reduction in churn probability"
        )
    
    with col2:
        control_size = st.slider(
            "Control Group Size",
            0.3, 0.7, 0.5,
            help="Proportion of players in control group"
        )
        
        retention_period = st.selectbox(
            "Retention Period (days)",
            [7, 14, 30, 60, 90],
            index=2
        )
        
        sample_size = st.slider(
            "Sample Size",
            1000, min(50000, len(df)), 10000,
            help="Number of players to include in test"
        )
    
    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary"):
        
        with st.spinner("Running A/B test simulation..."):
            # Sample players
            df_sample = df.sample(n=sample_size, random_state=42)
            
            # Ensure churn_probability column exists
            if 'churn_probability' not in df_sample.columns:
                if 'churn_risk_score' in df_sample.columns:
                    df_sample['churn_probability'] = df_sample['churn_risk_score']
                else:
                    df_sample['churn_probability'] = 0.25  # Default
            
            # Run test
            simulator = ABTestSimulator()
            results = simulator.run_ab_test(
                df_sample,
                test_name=test_name,
                intervention_type=intervention_type,
                effect_size=effect_size,
                control_size=control_size,
                retention_period=retention_period
            )
            
            # Display results
            st.success("✅ Simulation Complete!")
            
            # Key metrics
            st.subheader("📊 Results")
            
            metrics = results['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Control Retention",
                    f"{metrics['control']['retention_rate']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Treatment Retention",
                    f"{metrics['treatment']['retention_rate']:.2%}",
                    f"+{metrics['lift']['absolute']:.2%}"
                )
            
            with col3:
                st.metric(
                    "Relative Lift",
                    f"{metrics['lift']['relative_pct']:.1f}%"
                )
            
            # Statistical significance
            st.subheader("📈 Statistical Significance")
            
            stat_results = results['statistical_test']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "P-Value",
                    f"{stat_results['z_test']['p_value']:.4f}"
                )
                
                if stat_results['z_test']['significant']:
                    st.success(f"✅ **SIGNIFICANT** at α=0.05")
                else:
                    st.warning(f"⚠️ **NOT SIGNIFICANT** at α=0.05")
            
            with col2:
                ci = stat_results['confidence_interval_95']
                st.metric(
                    "95% Confidence Interval",
                    f"[{ci['lower']:.2%}, {ci['upper']:.2%}]"
                )
            
            # Visualization
            st.subheader("📊 Retention Comparison")
            
            comparison_data = pd.DataFrame({
                'Group': ['Control', 'Treatment'],
                'Retention Rate': [
                    metrics['control']['retention_rate'],
                    metrics['treatment']['retention_rate']
                ],
                'Churn Rate': [
                    metrics['control']['churn_rate'],
                    metrics['treatment']['churn_rate']
                ]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Retention Rate',
                x=comparison_data['Group'],
                y=comparison_data['Retention Rate'],
                marker_color=['#1f77b4', '#2ca02c']
            ))
            fig.update_layout(
                title=f"{retention_period}-Day Retention Comparison",
                yaxis_title="Rate",
                yaxis_tickformat='.0%',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


def show_interventions(df: pd.DataFrame):
    """Intervention recommendations page"""
    st.header("💡 Intervention Recommendations")
    
    # High-risk player interventions
    high_risk_df = df[df['risk_category'] == 'High']
    
    st.subheader(f"🎯 Targeting {len(high_risk_df):,} High-Risk Players")
    
    # Recommended interventions by segment
    st.markdown("### Personalized Intervention Strategy")
    
    # Low activity players
    low_activity = high_risk_df[high_risk_df['sessions_per_week'] < 2]
    st.markdown(f"""
    **1. Re-engagement Campaign** ({len(low_activity):,} players)
    - Target: Players with < 2 sessions/week
    - Action: Send personalized email/push notification
    - Offer: Exclusive in-game rewards for returning
    - Expected Impact: 12-15% churn reduction
    """)
    
    # Low social connection
    low_social = high_risk_df[high_risk_df.get('friends_online', 0) < 2]
    st.markdown(f"""
    **2. Social Connection Nudge** ({len(low_social):,} players)
    - Target: Players with few friends online
    - Action: Facilitate team-up with similar players
    - Offer: Bonus rewards for party play
    - Expected Impact: 8-10% churn reduction
    """)
    
    # Non-spenders at risk
    non_spenders = high_risk_df[high_risk_df.get('monthly_spent', 0) == 0]
    st.markdown(f"""
    **3. First Purchase Incentive** ({len(non_spenders):,} players)
    - Target: High-risk non-spenders
    - Action: Limited-time discount offer (50% off)
    - Benefit: Convert to paying users
    - Expected Impact: 15-20% churn reduction
    """)
    
    # Content exhaustion
    content_exhausted = high_risk_df[high_risk_df.get('content_completion_pct', 0) > 0.8]
    st.markdown(f"""
    **4. New Content Preview** ({len(content_exhausted):,} players)
    - Target: Players who completed most content
    - Action: Early access to upcoming features
    - Offer: Beta tester privileges
    - Expected Impact: 10-12% churn reduction
    """)
    
    # Priority queue
    st.subheader("📋 Intervention Priority Queue")
    
    # Calculate priority score
    priority_df = high_risk_df.copy()
    priority_df['ltv'] = priority_df.get('monthly_spent', 0) * 12
    priority_df['priority_score'] = (
        priority_df['churn_risk_score'] * 0.5 +
        (priority_df['ltv'] / priority_df['ltv'].max()) * 0.3 +
        (priority_df.get('engagement_score', 0)) * 0.2
    )
    
    top_priority = priority_df.nlargest(20, 'priority_score')[
        ['player_id', 'churn_risk_score', 'ltv', 'priority_score',
         'sessions_per_week', 'days_since_last_login']
    ]
    
    top_priority.columns = ['Player ID', 'Churn Risk', 'LTV ($)', 'Priority',
                            'Sessions/Week', 'Days Inactive']
    
    st.dataframe(
        top_priority.style.background_gradient(subset=['Priority'], cmap='RdYlGn'),
        use_container_width=True
    )


def show_roi_calculator(df: pd.DataFrame):
    """ROI calculator page"""
    st.header("💰 ROI Calculator")
    
    st.markdown("""
    Calculate the return on investment for retention interventions.
    Adjust the parameters below to see projected financial impact.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        players_targeted = st.number_input(
            "Players Targeted",
            min_value=100,
            max_value=len(df),
            value=min(10000, len(df)),
            step=1000
        )
        
        intervention_cost = st.number_input(
            "Cost per Player ($)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5,
            help="Cost to deliver intervention per player"
        )
        
        expected_lift = st.slider(
            "Expected Churn Reduction",
            0.05, 0.30, 0.15,
            format="%.0f%%",
            help="Expected reduction in churn rate"
        )
        
        avg_monthly_spend = st.number_input(
            "Avg Monthly Spend ($)",
            min_value=0.0,
            max_value=1000.0,
            value=float(df.get('monthly_spent', pd.Series([25])).mean()),
            step=5.0
        )
        
        retention_months = st.slider(
            "Retention Period (months)",
            3, 24, 12,
            help="Expected months player stays after intervention"
        )
    
    with col2:
        st.subheader("Calculated Results")
        
        # Calculate metrics
        player_ltv = avg_monthly_spend * retention_months
        players_saved = players_targeted * expected_lift
        revenue_retained = players_saved * player_ltv
        total_cost = players_targeted * intervention_cost
        net_benefit = revenue_retained - total_cost
        roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
        
        st.metric("Player LTV", f"${player_ltv:,.2f}")
        st.metric("Players Saved", f"{players_saved:,.0f}")
        st.metric("Revenue Retained", f"${revenue_retained:,.2f}")
        st.metric("Total Cost", f"${total_cost:,.2f}")
        st.metric("Net Benefit", f"${net_benefit:,.2f}",
                 delta=f"{net_benefit:,.0f}")
        
        # ROI display
        if roi > 0:
            st.success(f"### ROI: {roi:.1f}%")
        else:
            st.error(f"### ROI: {roi:.1f}%")
        
        st.metric("Breakeven Players Saved",
                 f"{total_cost / player_ltv:,.0f}")
    
    # Sensitivity analysis
    st.subheader("📊 Sensitivity Analysis")
    
    # Vary lift and cost
    lift_range = np.arange(0.05, 0.31, 0.05)
    cost_range = np.array([1, 2, 5, 10, 15, 20])
    
    roi_matrix = []
    for cost in cost_range:
        roi_row = []
        for lift in lift_range:
            saved = players_targeted * lift
            revenue = saved * player_ltv
            costs = players_targeted * cost
            net = revenue - costs
            roi_val = (net / costs * 100) if costs > 0 else 0
            roi_row.append(roi_val)
        roi_matrix.append(roi_row)
    
    roi_df = pd.DataFrame(
        roi_matrix,
        columns=[f"{l:.0%}" for l in lift_range],
        index=[f"${c}" for c in cost_range]
    )
    
    fig = px.imshow(
        roi_df,
        labels=dict(x="Churn Reduction", y="Cost per Player", color="ROI %"),
        x=roi_df.columns,
        y=roi_df.index,
        color_continuous_scale='RdYlGn',
        text_auto='.0f'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    💡 **Insight**: The sensitivity analysis shows how ROI varies with different intervention costs
    and effectiveness levels. Green indicates positive ROI, while red indicates negative ROI.
    """)


if __name__ == "__main__":
    main()
