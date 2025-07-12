"""
Streamlit Dashboard for YouTube Analytics
Interactive web interface using Streamlit.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from analytics import YouTubeAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitDashboard:
    """Streamlit dashboard for YouTube Analytics."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.analytics = None
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="YouTube Analytics Dashboard",
            page_icon="📺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_custom_css(self):
        """Load custom CSS styling."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #FF0000;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e1e5eb;
        }
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_header(self):
        """Display the main dashboard header."""
        st.markdown('<h1 class="main-header">📺 YouTube Analytics Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def display_sidebar(self):
        """Display the sidebar with controls."""
        with st.sidebar:
            st.header("📊 Dashboard Controls")
            
            # Data source selection
            st.subheader("Data Source")
            use_sample_data = st.checkbox("Use Sample Data", value=True)
            
            if not use_sample_data:
                videos_file = st.file_uploader(
                    "Upload Videos CSV", 
                    type=['csv'],
                    help="Upload your YouTube Studio videos data"
                )
                subscribers_file = st.file_uploader(
                    "Upload Subscribers CSV", 
                    type=['csv'],
                    help="Upload your YouTube Studio subscribers data"
                )
            else:
                videos_file = "data/sample/videos.csv"
                subscribers_file = "data/sample/subscribers.csv"
            
            # Analysis options
            st.subheader("Analysis Options")
            show_ml_predictions = st.checkbox("Show ML Predictions", value=True)
            show_insights = st.checkbox("Show Insights", value=True)
            
            # Chart options
            st.subheader("Visualization Options")
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"]
            )
            
            return {
                'videos_file': videos_file,
                'subscribers_file': subscribers_file,
                'show_ml_predictions': show_ml_predictions,
                'show_insights': show_insights,
                'chart_theme': chart_theme,
                'use_sample_data': use_sample_data
            }
    
    def initialize_analytics(self, config):
        """Initialize the analytics system."""
        try:
            if config['use_sample_data']:
                self.analytics = YouTubeAnalytics(
                    videos_file=config['videos_file'],
                    subscribers_file=config['subscribers_file']
                )
            else:
                # Handle uploaded files
                if config['videos_file'] is not None:
                    # Save uploaded file temporarily
                    temp_videos_path = f"temp_videos_{config['videos_file'].name}"
                    with open(temp_videos_path, "wb") as f:
                        f.write(config['videos_file'].getbuffer())
                    
                    temp_subs_path = None
                    if config['subscribers_file'] is not None:
                        temp_subs_path = f"temp_subs_{config['subscribers_file'].name}"
                        with open(temp_subs_path, "wb") as f:
                            f.write(config['subscribers_file'].getbuffer())
                    
                    self.analytics = YouTubeAnalytics(
                        videos_file=temp_videos_path,
                        subscribers_file=temp_subs_path
                    )
                else:
                    st.error("Please upload a videos CSV file to proceed.")
                    return False
            
            # Load data
            with st.spinner("Loading and processing data..."):
                self.analytics.load_data()
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing analytics: {e}")
            logger.error(f"Analytics initialization error: {e}")
            return False
    
    def display_overview_metrics(self):
        """Display overview metrics in the main area."""
        st.header("📈 Channel Overview")
        
        if self.analytics is None:
            st.warning("Please configure data source in the sidebar.")
            return
        
        try:
            # Generate summary statistics
            summary = self.analytics.generate_summary_statistics()
            overview = summary['overview']
            engagement = summary['engagement_metrics']
            
            # Create metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Videos",
                    value=f"{overview['total_videos']:,}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Total Views",
                    value=f"{overview['total_views']:,}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="Total Likes",
                    value=f"{overview['total_likes']:,}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="Total Comments",
                    value=f"{overview['total_comments']:,}",
                    delta=None
                )
            
            # Engagement metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    label="Avg Like Rate",
                    value=f"{engagement['average_like_rate']:.2f}%",
                    delta=None
                )
            
            with col6:
                st.metric(
                    label="Avg Comment Rate",
                    value=f"{engagement['average_comment_rate']:.2f}%",
                    delta=None
                )
            
            with col7:
                st.metric(
                    label="Avg Engagement",
                    value=f"{engagement['average_engagement_rate']:.2f}%",
                    delta=None
                )
            
            with col8:
                st.metric(
                    label="Median Views",
                    value=f"{engagement['median_views']:,.0f}",
                    delta=None
                )
            
            # Date range info
            st.info(f"📅 Data range: {overview['date_range']['start']} to {overview['date_range']['end']}")
            
        except Exception as e:
            st.error(f"Error displaying overview metrics: {e}")
            logger.error(f"Overview metrics error: {e}")
    
    def display_visualizations(self, chart_theme):
        """Display all visualization charts."""
        if self.analytics is None:
            return
        
        st.header("📊 Analytics Visualizations")
        
        try:
            with st.spinner("Creating visualizations..."):
                charts = self.analytics.create_all_visualizations()
            
            # Views timeline
            st.subheader("📈 Views Over Time")
            if 'views_timeline' in charts:
                st.plotly_chart(charts['views_timeline'], use_container_width=True)
            
            # Engagement comparison
            st.subheader("🎯 Engagement Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'engagement_comparison' in charts:
                    st.plotly_chart(charts['engagement_comparison'], use_container_width=True)
            
            with col2:
                if 'engagement_rates' in charts:
                    st.plotly_chart(charts['engagement_rates'], use_container_width=True)
            
            # Performance analysis
            st.subheader("🔍 Performance Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                if 'correlation_heatmap' in charts:
                    st.plotly_chart(charts['correlation_heatmap'], use_container_width=True)
            
            with col4:
                if 'performance_scatter' in charts:
                    st.plotly_chart(charts['performance_scatter'], use_container_width=True)
            
            # Top performers
            st.subheader("🏆 Top Performers")
            if 'top_performers' in charts:
                st.plotly_chart(charts['top_performers'], use_container_width=True)
            
            # Distributions
            st.subheader("📊 Data Distributions")
            col5, col6 = st.columns(2)
            
            with col5:
                if 'views_distribution' in charts:
                    st.plotly_chart(charts['views_distribution'], use_container_width=True)
            
            with col6:
                if 'engagement_distribution' in charts:
                    st.plotly_chart(charts['engagement_distribution'], use_container_width=True)
            
            # Subscriber activity
            if 'subscriber_activity' in charts:
                st.subheader("👥 Subscriber Activity")
                st.plotly_chart(charts['subscriber_activity'], use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating visualizations: {e}")
            logger.error(f"Visualization error: {e}")
    
    def display_ml_predictions(self):
        """Display ML prediction interface."""
        if self.analytics is None:
            return
        
        st.header("🤖 ML Performance Predictions")
        
        try:
            # Train model
            with st.spinner("Training ML model..."):
                training_results = self.analytics.train_prediction_model(hyperparameter_tuning=True)
            
            # Display training results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance")
                metrics = training_results['performance_metrics']
                st.metric("R² Score", f"{metrics['r2_score']:.3f}")
                st.metric("Mean Absolute Error", f"{metrics['mae']:.0f}")
                st.metric("Root Mean Square Error", f"{metrics['rmse']:.0f}")
            
            with col2:
                st.subheader("🎯 Feature Importance")
                if 'feature_analysis' in training_results:
                    importance_data = training_results['feature_analysis']['feature_importance']
                    features_df = pd.DataFrame(list(importance_data.items()), 
                                             columns=['Feature', 'Importance'])
                    features_df = features_df.sort_values('Importance', ascending=False)
                    
                    fig = px.bar(features_df, x='Importance', y='Feature', orientation='h',
                               title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Prediction interface
            st.subheader("🔮 Predict New Video Performance")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                duration = st.slider("Duration (minutes)", 1, 60, 10)
                likes_est = st.number_input("Estimated Likes", 0, 10000, 100)
            
            with col4:
                comments_est = st.number_input("Estimated Comments", 0, 1000, 20)
                like_rate = st.slider("Expected Like Rate (%)", 0.0, 10.0, 2.5)
            
            with col5:
                comment_rate = st.slider("Expected Comment Rate (%)", 0.0, 5.0, 0.5)
                engagement_rate = st.slider("Expected Engagement Rate (%)", 0.0, 15.0, 3.0)
            
            if st.button("🎯 Predict Views"):
                prediction_features = {
                    'Duration (minutes)': duration,
                    'Likes': likes_est,
                    'Comments': comments_est,
                    'Like Rate (%)': like_rate,
                    'Comment Rate (%)': comment_rate,
                    'Engagement Rate (%)': engagement_rate
                }
                
                try:
                    prediction = self.analytics.predict_video_performance(prediction_features)
                    
                    st.success(f"🎯 Predicted Views: {prediction['predicted_views']:,.0f}")
                    
                    if 'confidence_interval' in prediction:
                        ci = prediction['confidence_interval']
                        st.info(f"📊 95% Confidence Interval: {ci[0]:,.0f} - {ci[1]:,.0f} views")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        except Exception as e:
            st.error(f"Error with ML predictions: {e}")
            logger.error(f"ML predictions error: {e}")
    
    def display_insights(self):
        """Display actionable insights."""
        if self.analytics is None:
            return
        
        st.header("💡 Insights & Recommendations")
        
        try:
            insights = self.analytics.generate_insights()
            
            # Display insights by category
            for category, recommendations in insights.items():
                if recommendations and category != 'error':
                    st.subheader(f"📋 {category.replace('_', ' ').title()}")
                    
                    for i, recommendation in enumerate(recommendations, 1):
                        st.write(f"{i}. {recommendation}")
                    
                    st.markdown("---")
            
            if 'error' in insights:
                st.warning(f"Could not generate insights: {insights['error']}")
        
        except Exception as e:
            st.error(f"Error generating insights: {e}")
            logger.error(f"Insights error: {e}")
    
    def display_data_export(self):
        """Display data export options."""
        if self.analytics is None:
            return
        
        st.header("📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to Excel"):
                try:
                    with st.spinner("Generating Excel report..."):
                        self.analytics.export_results("data/exports")
                    st.success("✅ Excel report exported to data/exports/")
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        with col2:
            if st.button("📈 Save Charts"):
                try:
                    with st.spinner("Saving charts..."):
                        charts = self.analytics.create_all_visualizations(
                            save_charts=True, 
                            output_dir="data/exports/charts"
                        )
                    st.success("✅ Charts saved to data/exports/charts/")
                except Exception as e:
                    st.error(f"Chart save error: {e}")
        
        with col3:
            if st.button("🤖 Save ML Model"):
                try:
                    if not self.analytics.ml_predictor.is_trained:
                        self.analytics.train_prediction_model()
                    
                    self.analytics.ml_predictor.save_model("data/exports/ml_model.joblib")
                    st.success("✅ ML model saved to data/exports/")
                except Exception as e:
                    st.error(f"Model save error: {e}")
    
    def run(self):
        """Run the complete Streamlit dashboard."""
        # Load custom CSS
        self.load_custom_css()
        
        # Display header
        self.display_header()
        
        # Display sidebar and get configuration
        config = self.display_sidebar()
        
        # Initialize analytics if data source is configured
        if config['use_sample_data'] or config['videos_file'] is not None:
            if self.initialize_analytics(config):
                
                # Display main content
                self.display_overview_metrics()
                
                st.markdown("---")
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "🤖 ML Predictions", "💡 Insights", "📤 Export"])
                
                with tab1:
                    self.display_visualizations(config['chart_theme'])
                
                with tab2:
                    if config['show_ml_predictions']:
                        self.display_ml_predictions()
                    else:
                        st.info("ML Predictions disabled in sidebar settings.")
                
                with tab3:
                    if config['show_insights']:
                        self.display_insights()
                    else:
                        st.info("Insights disabled in sidebar settings.")
                
                with tab4:
                    self.display_data_export()
        else:
            st.info("👈 Please configure your data source in the sidebar to get started.")

def main():
    """Main function to run the Streamlit dashboard."""
    dashboard = StreamlitDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
