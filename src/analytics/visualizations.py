"""
Chart Generation Module for YouTube Analytics
Handles creating all types of interactive visualizations using Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Generates interactive charts and visualizations for YouTube analytics data.
    """
    
    def __init__(self, theme_colors: Optional[Dict[str, str]] = None):
        """
        Initialize the ChartGenerator.
        
        Args:
            theme_colors: Custom color scheme for charts
        """
        self.colors = theme_colors or {
            'youtube_red': '#FF0000',
            'like_green': '#00FF00',
            'comment_blue': '#0099FF',
            'engagement_orange': '#FF6600',
            'purple': '#8B5CF6',
            'background': 'rgba(0,0,0,0)'
        }
    
    def create_views_timeline(self, videos_df: pd.DataFrame, 
                            title: str = "📈 YouTube Views Over Time") -> go.Figure:
        """
        Create a line chart showing views over time.
        
        Args:
            videos_df: DataFrame with video data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.line(
                videos_df, 
                x='Publish Date', 
                y='Views',
                title=title,
                markers=True,
                hover_data=['Title', 'Likes', 'Comments']
            )
            
            fig.update_layout(
                title_font_size=20,
                xaxis_title="Publication Date",
                yaxis_title="Views",
                hovermode='x unified',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            fig.update_traces(
                line=dict(color=self.colors['youtube_red'], width=3),
                marker=dict(size=8, color=self.colors['youtube_red'])
            )
            
            logger.info("Views timeline chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating views timeline: {e}")
            raise
    
    def create_engagement_comparison(self, videos_df: pd.DataFrame,
                                   title: str = "📊 Engagement Metrics Comparison") -> go.Figure:
        """
        Create a grouped bar chart comparing engagement metrics.
        
        Args:
            videos_df: DataFrame with video data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Create shortened titles for better visualization
            df_viz = videos_df.copy()
            df_viz['Short Title'] = df_viz['Title'].str[:30] + '...'
            
            fig = go.Figure()
            
            # Add bars for each metric
            fig.add_trace(go.Bar(
                name='Views',
                x=df_viz['Short Title'],
                y=df_viz['Views'],
                marker_color=self.colors['youtube_red'],
                hovertemplate='<b>%{x}</b><br>Views: %{y:,}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='Likes',
                x=df_viz['Short Title'],
                y=df_viz['Likes'],
                marker_color=self.colors['like_green'],
                hovertemplate='<b>%{x}</b><br>Likes: %{y:,}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='Comments',
                x=df_viz['Short Title'],
                y=df_viz['Comments'],
                marker_color=self.colors['comment_blue'],
                hovertemplate='<b>%{x}</b><br>Comments: %{y:,}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                title_font_size=20,
                xaxis_title="Video Title",
                yaxis_title="Count",
                barmode='group',
                hovermode='x unified',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                xaxis_tickangle=-45
            )
            
            logger.info("Engagement comparison chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating engagement comparison: {e}")
            raise
    
    def create_engagement_rates_chart(self, videos_df: pd.DataFrame,
                                    title: str = "💯 Engagement Rates by Video") -> go.Figure:
        """
        Create a chart showing engagement rates.
        
        Args:
            videos_df: DataFrame with video data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            df_viz = videos_df.copy()
            df_viz['Short Title'] = df_viz['Title'].str[:30] + '...'
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Like Rate (%)',
                x=df_viz['Short Title'],
                y=df_viz['Like Rate (%)'],
                marker_color='#FFD700',
                hovertemplate='<b>%{x}</b><br>Like Rate: %{y:.2f}%<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                name='Comment Rate (%)',
                x=df_viz['Short Title'],
                y=df_viz['Comment Rate (%)'],
                marker_color='#FF69B4',
                hovertemplate='<b>%{x}</b><br>Comment Rate: %{y:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                title_font_size=20,
                xaxis_title="Video Title",
                yaxis_title="Rate (%)",
                barmode='group',
                hovermode='x unified',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                xaxis_tickangle=-45
            )
            
            logger.info("Engagement rates chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating engagement rates chart: {e}")
            raise
    
    def create_correlation_heatmap(self, videos_df: pd.DataFrame,
                                 title: str = "🔥 Performance Correlation Heatmap") -> go.Figure:
        """
        Create a correlation heatmap for video performance metrics.
        
        Args:
            videos_df: DataFrame with video data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            # Select metrics for correlation analysis
            metrics = ['Views', 'Likes', 'Comments', 'Like Rate (%)', 
                      'Comment Rate (%)', 'Duration (minutes)']
            
            # Filter metrics that exist in the dataframe
            available_metrics = [col for col in metrics if col in videos_df.columns]
            
            if len(available_metrics) < 2:
                raise ValueError("Not enough metrics available for correlation analysis")
            
            corr_matrix = videos_df[available_metrics].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlBu',
                zmid=0,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False,
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                title_font_size=20,
                width=700,
                height=700
            )
            
            logger.info("Correlation heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            raise
    
    def create_subscriber_activity_chart(self, subscribers_df: pd.DataFrame,
                                       title: str = "📈 Subscriber Activity Over Time") -> go.Figure:
        """
        Create a line chart showing subscriber activity.
        
        Args:
            subscribers_df: DataFrame with subscriber data
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=subscribers_df['Date'],
                y=subscribers_df['Subscribers Gained'],
                mode='lines+markers',
                name='Subscribers Gained',
                line=dict(color=self.colors['like_green'], width=3),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Gained: %{y}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=subscribers_df['Date'],
                y=subscribers_df['Subscribers Lost'],
                mode='lines+markers',
                name='Subscribers Lost',
                line=dict(color=self.colors['youtube_red'], width=3),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Lost: %{y}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=subscribers_df['Date'],
                y=subscribers_df['Net Subscribers'],
                mode='lines+markers',
                name='Net Change',
                line=dict(color=self.colors['comment_blue'], width=3),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Net: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title=title,
                title_font_size=20,
                xaxis_title="Date",
                yaxis_title="Subscribers",
                hovermode='x unified',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            logger.info("Subscriber activity chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating subscriber activity chart: {e}")
            raise
    
    def create_performance_scatter(self, videos_df: pd.DataFrame,
                                 x_metric: str = 'Views',
                                 y_metric: str = 'Like Rate (%)',
                                 size_metric: str = 'Comments',
                                 color_metric: str = 'Engagement Rate (%)',
                                 title: str = "💡 Performance Scatter Analysis") -> go.Figure:
        """
        Create a scatter plot for performance analysis.
        
        Args:
            videos_df: DataFrame with video data
            x_metric: Column name for x-axis
            y_metric: Column name for y-axis
            size_metric: Column name for bubble size
            color_metric: Column name for color scale
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            fig = px.scatter(
                videos_df,
                x=x_metric,
                y=y_metric,
                size=size_metric,
                color=color_metric,
                hover_data=['Title'],
                title=title,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                title_font_size=20,
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            logger.info(f"Performance scatter plot created: {x_metric} vs {y_metric}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance scatter plot: {e}")
            raise
    
    def create_top_performers_chart(self, videos_df: pd.DataFrame,
                                  metric: str = 'Views',
                                  top_n: int = 5,
                                  chart_type: str = 'bar') -> go.Figure:
        """
        Create a chart showing top performing videos.
        
        Args:
            videos_df: DataFrame with video data
            metric: Metric to rank by
            top_n: Number of top videos to show
            chart_type: 'bar' or 'horizontal_bar'
            
        Returns:
            Plotly figure object
        """
        try:
            top_videos = videos_df.nlargest(top_n, metric).copy()
            top_videos['Short Title'] = top_videos['Title'].str[:25] + '...'
            
            if chart_type == 'horizontal_bar':
                fig = px.bar(
                    top_videos,
                    x=metric,
                    y='Short Title',
                    orientation='h',
                    title=f'🏆 Top {top_n} Videos by {metric}',
                    color=metric,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            else:
                fig = px.bar(
                    top_videos,
                    x='Short Title',
                    y=metric,
                    title=f'🏆 Top {top_n} Videos by {metric}',
                    color=metric,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(xaxis_tickangle=-45)
            
            fig.update_layout(
                title_font_size=20,
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            logger.info(f"Top performers chart created for {metric}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating top performers chart: {e}")
            raise
    
    def create_distribution_histogram(self, videos_df: pd.DataFrame,
                                    metric: str,
                                    bins: int = 20,
                                    title: Optional[str] = None) -> go.Figure:
        """
        Create a histogram showing distribution of a metric.
        
        Args:
            videos_df: DataFrame with video data
            metric: Column name for the metric
            bins: Number of bins for histogram
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        try:
            if title is None:
                title = f"📊 Distribution of {metric}"
            
            fig = px.histogram(
                videos_df,
                x=metric,
                nbins=bins,
                title=title,
                color_discrete_sequence=[self.colors['youtube_red']]
            )
            
            fig.update_layout(
                title_font_size=20,
                xaxis_title=metric,
                yaxis_title="Frequency",
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            logger.info(f"Distribution histogram created for {metric}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution histogram: {e}")
            raise
    
    def create_multi_metric_dashboard(self, videos_df: pd.DataFrame,
                                    subscribers_df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts.
        
        Args:
            videos_df: DataFrame with video data
            subscribers_df: Optional DataFrame with subscriber data
            
        Returns:
            Plotly figure object with subplots
        """
        try:
            # Determine subplot configuration
            rows = 3 if subscribers_df is not None else 2
            cols = 2
            
            subplot_titles = [
                'Views Over Time',
                'Top 5 Videos by Views',
                'Engagement Rates',
                'Performance Correlation'
            ]
            
            if subscribers_df is not None:
                subplot_titles.extend(['Subscriber Activity', 'Growth Rate'])
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]] if rows == 3 else 
                      [[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Views over time
            fig.add_trace(
                go.Scatter(
                    x=videos_df['Publish Date'],
                    y=videos_df['Views'],
                    mode='lines+markers',
                    name='Views',
                    line=dict(color=self.colors['youtube_red'])
                ),
                row=1, col=1
            )
            
            # Top 5 videos
            top_5 = videos_df.nlargest(5, 'Views')
            top_5_short = top_5['Title'].str[:20] + '...'
            
            fig.add_trace(
                go.Bar(
                    x=top_5['Views'],
                    y=top_5_short,
                    orientation='h',
                    name='Top Videos',
                    marker_color=self.colors['youtube_red']
                ),
                row=1, col=2
            )
            
            # Engagement rates
            fig.add_trace(
                go.Bar(
                    x=videos_df['Title'].str[:15] + '...',
                    y=videos_df['Like Rate (%)'],
                    name='Like Rate',
                    marker_color=self.colors['like_green']
                ),
                row=2, col=1
            )
            
            # Correlation heatmap (simplified)
            metrics = ['Views', 'Likes', 'Comments', 'Like Rate (%)']
            available_metrics = [col for col in metrics if col in videos_df.columns]
            
            if len(available_metrics) >= 2:
                corr_matrix = videos_df[available_metrics].corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdYlBu',
                        name='Correlation'
                    ),
                    row=2, col=2
                )
            
            # Subscriber data if available
            if subscribers_df is not None:
                fig.add_trace(
                    go.Scatter(
                        x=subscribers_df['Date'],
                        y=subscribers_df['Net Subscribers'],
                        mode='lines+markers',
                        name='Net Subscribers',
                        line=dict(color=self.colors['comment_blue'])
                    ),
                    row=3, col=1
                )
                
                if 'Growth Rate (%)' in subscribers_df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=subscribers_df['Date'],
                            y=subscribers_df['Growth Rate (%)'],
                            name='Growth Rate',
                            marker_color=self.colors['engagement_orange']
                        ),
                        row=3, col=2
                    )
            
            fig.update_layout(
                title_text="📊 YouTube Analytics Dashboard",
                title_font_size=24,
                showlegend=False,
                height=800 if rows == 3 else 600
            )
            
            logger.info("Multi-metric dashboard created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-metric dashboard: {e}")
            raise
    
    def save_chart(self, fig: go.Figure, filename: str, 
                  format: str = 'html', width: int = 1200, height: int = 600) -> None:
        """
        Save chart to file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format ('html', 'png', 'jpg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
        """
        try:
            if format == 'html':
                fig.write_html(filename)
            else:
                fig.write_image(filename, width=width, height=height, format=format)
            
            logger.info(f"Chart saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            raise
