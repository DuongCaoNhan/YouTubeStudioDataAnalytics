"""
YouTube Analytics Dashboard
A comprehensive tool for analyzing YouTube Studio data including views, likes, comments, and subscriber analytics.

Author: YouTube Analytics Team
Date: 2024
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class YouTubeAnalytics:
    """Main class for YouTube analytics operations"""
    
    def __init__(self, videos_file='videos.csv', subscribers_file='subscribers.csv'):
        """
        Initialize the YouTube Analytics class
        
        Args:
            videos_file (str): Path to the videos CSV file
            subscribers_file (str): Path to the subscribers CSV file
        """
        self.videos_file = videos_file
        self.subscribers_file = subscribers_file
        self.videos_df = None
        self.subscribers_df = None
        
    def load_data(self):
        """Load and preprocess the YouTube data"""
        try:
            # Load videos data
            print("📊 Loading YouTube analytics data...")
            self.videos_df = pd.read_csv(self.videos_file)
            print(f"✅ Loaded {len(self.videos_df)} videos from {self.videos_file}")
            
            # Display columns for debugging
            print(f"📋 Video data columns: {list(self.videos_df.columns)}")
            
            # Convert and sort by date
            self.videos_df['Publish Date'] = pd.to_datetime(self.videos_df['Publish Date'])
            self.videos_df = self.videos_df.sort_values('Publish Date')
            
            # Calculate additional metrics
            self.videos_df['Like Rate (%)'] = (self.videos_df['Likes'] / self.videos_df['Views']) * 100
            self.videos_df['Comment Rate (%)'] = (self.videos_df['Comments'] / self.videos_df['Views']) * 100
            self.videos_df['Engagement Rate (%)'] = ((self.videos_df['Likes'] + self.videos_df['Comments']) / self.videos_df['Views']) * 100
            
            # Load subscribers data if available
            try:
                self.subscribers_df = pd.read_csv(self.subscribers_file)
                self.subscribers_df['Date'] = pd.to_datetime(self.subscribers_df['Date'])
                self.subscribers_df = self.subscribers_df.sort_values('Date')
                print(f"✅ Loaded {len(self.subscribers_df)} subscriber records from {self.subscribers_file}")
            except FileNotFoundError:
                print(f"⚠️  Subscriber file {self.subscribers_file} not found. Skipping subscriber analysis.")
                
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            raise
        except Exception as e:
            print(f"❌ Error processing data: {e}")
            raise
    
    def display_summary_stats(self):
        """Display summary statistics of the YouTube data"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return
            
        print("\n" + "="*60)
        print("📈 YOUTUBE CHANNEL ANALYTICS SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_views = self.videos_df['Views'].sum()
        total_likes = self.videos_df['Likes'].sum()
        total_comments = self.videos_df['Comments'].sum()
        avg_like_rate = self.videos_df['Like Rate (%)'].mean()
        avg_comment_rate = self.videos_df['Comment Rate (%)'].mean()
        
        print(f"📺 Total Videos: {len(self.videos_df):,}")
        print(f"👀 Total Views: {total_views:,}")
        print(f"👍 Total Likes: {total_likes:,}")
        print(f"💬 Total Comments: {total_comments:,}")
        print(f"📊 Average Like Rate: {avg_like_rate:.2f}%")
        print(f"💭 Average Comment Rate: {avg_comment_rate:.2f}%")
        
        # Top performing videos
        print(f"\n🏆 TOP 5 VIDEOS BY VIEWS:")
        top_views = self.videos_df.nlargest(5, 'Views')[['Title', 'Views', 'Likes', 'Comments']]
        for idx, row in top_views.iterrows():
            print(f"   {row['Title'][:40]}... - {row['Views']:,} views, {row['Likes']:,} likes")
        
        # Top videos by engagement rate
        print(f"\n🎯 TOP 5 VIDEOS BY LIKE RATE:")
        top_engagement = self.videos_df.nlargest(5, 'Like Rate (%)')[['Title', 'Views', 'Likes', 'Like Rate (%)']]
        for idx, row in top_engagement.iterrows():
            print(f"   {row['Title'][:40]}... - {row['Like Rate (%)']:.2f}% like rate")
    
    def create_views_over_time_chart(self):
        """Create an interactive line chart showing views over time"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
            
        fig = px.line(
            self.videos_df, 
            x='Publish Date', 
            y='Views',
            title='📈 YouTube Views Over Time',
            markers=True,
            hover_data=['Title', 'Likes', 'Comments']
        )
        
        fig.update_layout(
            title_font_size=20,
            xaxis_title="Publication Date",
            yaxis_title="Views",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_traces(
            line=dict(color='#FF0000', width=3),
            marker=dict(size=8, color='#FF0000')
        )
        
        return fig
    
    def create_engagement_comparison_chart(self):
        """Create a grouped bar chart comparing views, likes, and comments"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        # Create shortened titles for better visualization
        df_viz = self.videos_df.copy()
        df_viz['Short Title'] = df_viz['Title'].str[:30] + '...'
        
        fig = go.Figure()
        
        # Add bars for each metric
        fig.add_trace(go.Bar(
            name='Views',
            x=df_viz['Short Title'],
            y=df_viz['Views'],
            marker_color='#FF0000'
        ))
        
        fig.add_trace(go.Bar(
            name='Likes',
            x=df_viz['Short Title'],
            y=df_viz['Likes'],
            marker_color='#00FF00'
        ))
        
        fig.add_trace(go.Bar(
            name='Comments',
            x=df_viz['Short Title'],
            y=df_viz['Comments'],
            marker_color='#0099FF'
        ))
        
        fig.update_layout(
            title='📊 Engagement Metrics Comparison Across Videos',
            title_font_size=20,
            xaxis_title="Video Title",
            yaxis_title="Count",
            barmode='group',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_engagement_rates_chart(self):
        """Create a chart showing engagement rates (Like Rate, Comment Rate)"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        df_viz = self.videos_df.copy()
        df_viz['Short Title'] = df_viz['Title'].str[:30] + '...'
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Like Rate (%)',
            x=df_viz['Short Title'],
            y=df_viz['Like Rate (%)'],
            marker_color='#FFD700'
        ))
        
        fig.add_trace(go.Bar(
            name='Comment Rate (%)',
            x=df_viz['Short Title'],
            y=df_viz['Comment Rate (%)'],
            marker_color='#FF69B4'
        ))
        
        fig.update_layout(
            title='💯 Engagement Rates by Video',
            title_font_size=20,
            xaxis_title="Video Title",
            yaxis_title="Rate (%)",
            barmode='group',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_subscriber_activity_chart(self):
        """Create a line chart showing subscriber activity over time"""
        if self.subscribers_df is None:
            print("⚠️  No subscriber data available.")
            return None
            
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.subscribers_df['Date'],
            y=self.subscribers_df['Subscribers Gained'],
            mode='lines+markers',
            name='Subscribers Gained',
            line=dict(color='#00FF00', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.subscribers_df['Date'],
            y=self.subscribers_df['Subscribers Lost'],
            mode='lines+markers',
            name='Subscribers Lost',
            line=dict(color='#FF0000', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.subscribers_df['Date'],
            y=self.subscribers_df['Net Subscribers'],
            mode='lines+markers',
            name='Net Subscriber Change',
            line=dict(color='#0099FF', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='📈 Subscriber Activity Over Time',
            title_font_size=20,
            xaxis_title="Date",
            yaxis_title="Subscribers",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_performance_heatmap(self):
        """Create a heatmap showing video performance correlation"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        # Create correlation matrix
        metrics = ['Views', 'Likes', 'Comments', 'Like Rate (%)', 'Comment Rate (%)', 'Duration (minutes)']
        corr_matrix = self.videos_df[metrics].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='🔥 Video Performance Correlation Heatmap',
            title_font_size=20,
            width=700,
            height=700
        )
        
        return fig
    
    def predict_views(self, features=['Duration (minutes)', 'Like Rate (%)', 'Comment Rate (%)']):
        """Predict views for future videos using machine learning"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return None
        
        print("\n🤖 MACHINE LEARNING PREDICTION MODEL")
        print("="*50)
        
        # Prepare data for ML
        df_ml = self.videos_df.dropna(subset=features + ['Views'])
        
        if len(df_ml) < 5:
            print("❌ Not enough data for machine learning prediction.")
            return None
        
        X = df_ml[features]
        y = df_ml['Views']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   R² Score: {r2:.3f}")
        print(f"   Mean Squared Error: {mse:.0f}")
        print(f"   Root Mean Squared Error: {np.sqrt(mse):.0f}")
        
        # Feature importance
        print(f"\n🎯 Feature Importance:")
        for feature, coef in zip(features, model.coef_):
            print(f"   {feature}: {coef:.2f}")
        
        # Predict for a sample video
        if len(df_ml) > 0:
            sample_features = df_ml[features].mean().values.reshape(1, -1)
            predicted_views = model.predict(sample_features)[0]
            print(f"\n🔮 Predicted views for average video: {predicted_views:.0f}")
        
        return model
    
    def export_analysis_report(self, filename='youtube_analytics_report.xlsx'):
        """Export analysis results to Excel file"""
        if self.videos_df is None:
            print("❌ No data loaded. Please run load_data() first.")
            return
        
        print(f"\n📁 Exporting analysis report to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main video data
            self.videos_df.to_excel(writer, sheet_name='Video Analytics', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': ['Total Videos', 'Total Views', 'Total Likes', 'Total Comments', 
                          'Average Like Rate (%)', 'Average Comment Rate (%)', 'Average Engagement Rate (%)'],
                'Value': [
                    len(self.videos_df),
                    self.videos_df['Views'].sum(),
                    self.videos_df['Likes'].sum(),
                    self.videos_df['Comments'].sum(),
                    self.videos_df['Like Rate (%)'].mean(),
                    self.videos_df['Comment Rate (%)'].mean(),
                    self.videos_df['Engagement Rate (%)'].mean()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Top performers
            top_views = self.videos_df.nlargest(10, 'Views')[['Title', 'Views', 'Likes', 'Comments', 'Like Rate (%)']]
            top_views.to_excel(writer, sheet_name='Top Videos by Views', index=False)
            
            top_engagement = self.videos_df.nlargest(10, 'Like Rate (%)')[['Title', 'Views', 'Likes', 'Like Rate (%)']]
            top_engagement.to_excel(writer, sheet_name='Top Videos by Engagement', index=False)
            
            # Subscriber data if available
            if self.subscribers_df is not None:
                self.subscribers_df.to_excel(writer, sheet_name='Subscriber Activity', index=False)
        
        print(f"✅ Report exported successfully to {filename}")
    
    def run_complete_analysis(self):
        """Run the complete YouTube analytics pipeline"""
        print("🚀 Starting YouTube Analytics Dashboard...")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Display summary
        self.display_summary_stats()
        
        # Create and show visualizations
        print(f"\n📊 Creating interactive visualizations...")
        
        # Views over time
        fig1 = self.create_views_over_time_chart()
        if fig1:
            fig1.show()
        
        # Engagement comparison
        fig2 = self.create_engagement_comparison_chart()
        if fig2:
            fig2.show()
        
        # Engagement rates
        fig3 = self.create_engagement_rates_chart()
        if fig3:
            fig3.show()
        
        # Subscriber activity
        fig4 = self.create_subscriber_activity_chart()
        if fig4:
            fig4.show()
        
        # Performance heatmap
        fig5 = self.create_performance_heatmap()
        if fig5:
            fig5.show()
        
        # Machine learning prediction
        try:
            self.predict_views()
        except Exception as e:
            print(f"⚠️  ML prediction failed: {e}")
        
        # Export report
        try:
            self.export_analysis_report()
        except Exception as e:
            print(f"⚠️  Report export failed: {e}")
        
        print(f"\n🎉 Analysis complete! Check your browser for interactive charts.")


def main():
    """Main function to run the YouTube Analytics"""
    # Initialize the analytics class
    analytics = YouTubeAnalytics()
    
    # Run complete analysis
    analytics.run_complete_analysis()


if __name__ == "__main__":
    main()
