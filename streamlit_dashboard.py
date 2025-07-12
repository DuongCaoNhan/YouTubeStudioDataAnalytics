"""
YouTube Analytics Streamlit Dashboard
Interactive web dashboard for YouTube Studio analytics with real-time insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from youtube_analytics import YouTubeAnalytics
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF0000;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #FF0000;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF0000;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_analytics_data():
    """Load and cache the analytics data"""
    analytics = YouTubeAnalytics()
    analytics.load_data()
    return analytics

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📺 YouTube Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    try:
        analytics = load_analytics_data()
        
        if analytics.videos_df is None:
            st.error("❌ Unable to load video data. Please check your CSV files.")
            return
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Date range filter
    min_date = analytics.videos_df['Publish Date'].min()
    max_date = analytics.videos_df['Publish Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = analytics.videos_df[
            (analytics.videos_df['Publish Date'].dt.date >= start_date) &
            (analytics.videos_df['Publish Date'].dt.date <= end_date)
        ]
    else:
        filtered_df = analytics.videos_df
    
    # View selection
    view_option = st.sidebar.selectbox(
        "📈 Select Analysis View",
        ["Overview", "Video Performance", "Engagement Analysis", "Subscriber Insights", "Predictions"]
    )
    
    # Main content area
    if view_option == "Overview":
        show_overview(filtered_df, analytics)
    elif view_option == "Video Performance":
        show_video_performance(filtered_df)
    elif view_option == "Engagement Analysis":
        show_engagement_analysis(filtered_df)
    elif view_option == "Subscriber Insights":
        show_subscriber_insights(analytics)
    elif view_option == "Predictions":
        show_predictions(analytics)

def show_overview(df, analytics):
    """Show overview dashboard"""
    st.header("📊 Channel Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_views = df['Views'].sum()
        st.metric("👀 Total Views", f"{total_views:,}")
    
    with col2:
        total_likes = df['Likes'].sum()
        st.metric("👍 Total Likes", f"{total_likes:,}")
    
    with col3:
        total_comments = df['Comments'].sum()
        st.metric("💬 Total Comments", f"{total_comments:,}")
    
    with col4:
        avg_engagement = df['Engagement Rate (%)'].mean()
        st.metric("📈 Avg Engagement Rate", f"{avg_engagement:.2f}%")
    
    st.markdown("---")
    
    # Charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Views Over Time")
        fig_views = px.line(
            df, x='Publish Date', y='Views',
            markers=True,
            hover_data=['Title', 'Likes', 'Comments']
        )
        fig_views.update_traces(line_color='#FF0000')
        st.plotly_chart(fig_views, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 5 Videos by Views")
        top_videos = df.nlargest(5, 'Views')[['Title', 'Views']]
        top_videos['Short Title'] = top_videos['Title'].str[:30] + '...'
        
        fig_top = px.bar(
            top_videos, x='Views', y='Short Title',
            orientation='h',
            color='Views',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Performance table
    st.subheader("📋 Recent Video Performance")
    display_df = df[['Title', 'Publish Date', 'Views', 'Likes', 'Comments', 'Like Rate (%)', 'Engagement Rate (%)']].copy()
    display_df['Publish Date'] = display_df['Publish Date'].dt.strftime('%Y-%m-%d')
    display_df = display_df.sort_values('Publish Date', ascending=False)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def show_video_performance(df):
    """Show detailed video performance analysis"""
    st.header("🎥 Video Performance Analysis")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Views vs Engagement")
        fig_scatter = px.scatter(
            df, x='Views', y='Like Rate (%)',
            size='Comments',
            hover_data=['Title'],
            color='Engagement Rate (%)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Duration vs Performance")
        fig_duration = px.scatter(
            df, x='Duration (minutes)', y='Views',
            size='Likes',
            hover_data=['Title'],
            color='Like Rate (%)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    # Engagement comparison
    st.subheader("📈 Engagement Metrics Comparison")
    
    # Prepare data for grouped bar chart
    df_viz = df.copy()
    df_viz['Short Title'] = df_viz['Title'].str[:25] + '...'
    
    fig_engagement = go.Figure()
    
    fig_engagement.add_trace(go.Bar(
        name='Views',
        x=df_viz['Short Title'],
        y=df_viz['Views'],
        marker_color='#FF0000'
    ))
    
    fig_engagement.add_trace(go.Bar(
        name='Likes',
        x=df_viz['Short Title'],
        y=df_viz['Likes'],
        marker_color='#00FF00'
    ))
    
    fig_engagement.add_trace(go.Bar(
        name='Comments',
        x=df_viz['Short Title'],
        y=df_viz['Comments'],
        marker_color='#0099FF'
    ))
    
    fig_engagement.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig_engagement, use_container_width=True)

def show_engagement_analysis(df):
    """Show engagement analysis"""
    st.header("💡 Engagement Analysis")
    
    # Engagement rates
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Like Rate Distribution")
        fig_like_dist = px.histogram(
            df, x='Like Rate (%)',
            nbins=20,
            color_discrete_sequence=['#FF0000']
        )
        st.plotly_chart(fig_like_dist, use_container_width=True)
    
    with col2:
        st.subheader("💬 Comment Rate Distribution")
        fig_comment_dist = px.histogram(
            df, x='Comment Rate (%)',
            nbins=20,
            color_discrete_sequence=['#0099FF']
        )
        st.plotly_chart(fig_comment_dist, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Performance Correlation Matrix")
    
    metrics = ['Views', 'Likes', 'Comments', 'Like Rate (%)', 'Comment Rate (%)', 'Duration (minutes)']
    corr_matrix = df[metrics].corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlBu',
        zmid=0,
        text=np.around(corr_matrix.values, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Videos by Like Rate")
        top_like_rate = df.nlargest(5, 'Like Rate (%)')[['Title', 'Views', 'Like Rate (%)']]
        st.dataframe(top_like_rate, hide_index=True)
    
    with col2:
        st.subheader("💬 Top Videos by Comment Rate")
        top_comment_rate = df.nlargest(5, 'Comment Rate (%)')[['Title', 'Views', 'Comment Rate (%)']]
        st.dataframe(top_comment_rate, hide_index=True)

def show_subscriber_insights(analytics):
    """Show subscriber insights"""
    st.header("👥 Subscriber Insights")
    
    if analytics.subscribers_df is None:
        st.warning("⚠️ No subscriber data available. Please ensure subscribers.csv is in the project directory.")
        return
    
    df_subs = analytics.subscribers_df
    
    # Subscriber metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_gained = df_subs['Subscribers Gained'].sum()
        st.metric("📈 Total Gained", f"{total_gained:,}")
    
    with col2:
        total_lost = df_subs['Subscribers Lost'].sum()
        st.metric("📉 Total Lost", f"{total_lost:,}")
    
    with col3:
        net_change = df_subs['Net Subscribers'].sum()
        st.metric("📊 Net Change", f"{net_change:,}")
    
    with col4:
        avg_daily_gain = df_subs['Subscribers Gained'].mean()
        st.metric("📅 Avg Daily Gain", f"{avg_daily_gain:.1f}")
    
    # Subscriber activity chart
    st.subheader("📈 Subscriber Activity Over Time")
    
    fig_subs = go.Figure()
    
    fig_subs.add_trace(go.Scatter(
        x=df_subs['Date'],
        y=df_subs['Subscribers Gained'],
        mode='lines+markers',
        name='Gained',
        line=dict(color='#00FF00', width=3)
    ))
    
    fig_subs.add_trace(go.Scatter(
        x=df_subs['Date'],
        y=df_subs['Subscribers Lost'],
        mode='lines+markers',
        name='Lost',
        line=dict(color='#FF0000', width=3)
    ))
    
    fig_subs.add_trace(go.Scatter(
        x=df_subs['Date'],
        y=df_subs['Net Subscribers'],
        mode='lines+markers',
        name='Net Change',
        line=dict(color='#0099FF', width=3)
    ))
    
    st.plotly_chart(fig_subs, use_container_width=True)
    
    # Recent subscriber activity
    st.subheader("📋 Recent Subscriber Activity")
    recent_subs = df_subs.tail(10)[['Date', 'Subscribers Gained', 'Subscribers Lost', 'Net Subscribers']]
    recent_subs['Date'] = recent_subs['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(recent_subs, hide_index=True, use_container_width=True)

def show_predictions(analytics):
    """Show ML predictions and insights"""
    st.header("🤖 AI-Powered Predictions")
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        df = analytics.videos_df
        
        # Prepare data for ML
        features = ['Duration (minutes)', 'Like Rate (%)', 'Comment Rate (%)']
        df_ml = df.dropna(subset=features + ['Views'])
        
        if len(df_ml) < 5:
            st.warning("⚠️ Not enough data for machine learning predictions.")
            return
        
        X = df_ml[features]
        y = df_ml['Views']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Model performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 R² Score", f"{r2:.3f}")
        with col2:
            st.metric("📊 RMSE", f"{rmse:.0f}")
        with col3:
            st.metric("📈 Model Accuracy", f"{r2*100:.1f}%")
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.coef_
        })
        
        fig_importance = px.bar(
            importance_df, x='Feature', y='Importance',
            color='Importance',
            color_continuous_scale='RdYlBu'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Prediction vs Actual
        st.subheader("🔮 Prediction vs Actual Views")
        
        comparison_df = pd.DataFrame({
            'Actual Views': y_test,
            'Predicted Views': y_pred
        })
        
        fig_pred = px.scatter(
            comparison_df, x='Actual Views', y='Predicted Views',
            title='Prediction Accuracy'
        )
        fig_pred.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Interactive prediction tool
        st.subheader("🎮 Interactive View Predictor")
        st.write("Adjust the parameters below to predict views for a new video:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration = st.slider("Video Duration (minutes)", 5.0, 30.0, 15.0, 0.5)
        with col2:
            like_rate = st.slider("Expected Like Rate (%)", 0.5, 5.0, 2.5, 0.1)
        with col3:
            comment_rate = st.slider("Expected Comment Rate (%)", 0.1, 1.0, 0.3, 0.05)
        
        # Make prediction
        prediction_features = np.array([[duration, like_rate, comment_rate]])
        predicted_views = model.predict(prediction_features)[0]
        
        st.success(f"🔮 Predicted Views: **{predicted_views:,.0f}** views")
        
        # Confidence interval (simple estimation)
        confidence = predicted_views * 0.2  # ±20% confidence interval
        st.info(f"📊 Confidence Range: {predicted_views-confidence:,.0f} - {predicted_views+confidence:,.0f} views")
        
    except ImportError:
        st.error("❌ Scikit-learn not installed. Please install it to use prediction features.")
    except Exception as e:
        st.error(f"❌ Error in prediction analysis: {str(e)}")

if __name__ == "__main__":
    main()
