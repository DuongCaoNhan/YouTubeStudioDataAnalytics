"""
Dash Dashboard for YouTube Analytics
Professional dashboard using Plotly Dash.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys
import logging
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from analytics import YouTubeAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashDashboard:
    """Dash dashboard for YouTube Analytics."""
    
    def __init__(self, host='127.0.0.1', port=8050, debug=True):
        """Initialize the Dash dashboard."""
        self.host = host
        self.port = port
        self.debug = debug
        self.analytics = None
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "YouTube Analytics Dashboard"
        
        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("📺 YouTube Analytics Dashboard", 
                       className="header-title"),
                html.P("Professional analytics for your YouTube channel",
                      className="header-subtitle")
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.H3("🎛️ Controls"),
                    
                    # Data source selection
                    html.Div([
                        html.Label("Data Source:"),
                        dcc.RadioItems(
                            id='data-source',
                            options=[
                                {'label': 'Sample Data', 'value': 'sample'},
                                {'label': 'Upload Files', 'value': 'upload'}
                            ],
                            value='sample',
                            className="radio-items"
                        )
                    ], className="control-group"),
                    
                    # File upload (hidden by default)
                    html.Div([
                        html.Label("Videos CSV:"),
                        dcc.Upload(
                            id='upload-videos',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Videos CSV')
                            ]),
                            className="upload-area"
                        ),
                        html.Label("Subscribers CSV:"),
                        dcc.Upload(
                            id='upload-subscribers',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Subscribers CSV')
                            ]),
                            className="upload-area"
                        )
                    ], id='upload-section', style={'display': 'none'}),
                    
                    # Analysis options
                    html.Div([
                        html.Label("Analysis Options:"),
                        dcc.Checklist(
                            id='analysis-options',
                            options=[
                                {'label': 'Show ML Predictions', 'value': 'ml'},
                                {'label': 'Show Insights', 'value': 'insights'},
                                {'label': 'Show Advanced Charts', 'value': 'advanced'}
                            ],
                            value=['ml', 'insights', 'advanced'],
                            className="checklist"
                        )
                    ], className="control-group"),
                    
                    # Load data button
                    html.Button("📊 Load Analytics", id="load-button", 
                              className="btn btn-primary"),
                    
                    # Status indicator
                    html.Div(id="status-indicator", className="status")
                    
                ], className="control-panel")
            ], className="controls-container"),
            
            # Main content area
            html.Div([
                # Loading indicator
                dcc.Loading(
                    id="loading-main",
                    children=[
                        # Overview metrics
                        html.Div(id="overview-metrics"),
                        
                        # Tabs for different sections
                        dcc.Tabs(id="main-tabs", value='overview', children=[
                            dcc.Tab(label='📈 Overview', value='overview'),
                            dcc.Tab(label='📊 Visualizations', value='viz'),
                            dcc.Tab(label='🤖 ML Predictions', value='ml'),
                            dcc.Tab(label='💡 Insights', value='insights'),
                            dcc.Tab(label='📤 Export', value='export')
                        ]),
                        
                        # Tab content
                        html.Div(id="tab-content")
                    ]
                )
            ], className="main-content"),
            
            # Hidden data storage
            dcc.Store(id='analytics-data'),
            dcc.Store(id='charts-data'),
            dcc.Store(id='ml-results'),
            
        ], className="dashboard-container")
        
        # Add CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .dashboard-container {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                        max-width: 1400px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .header {
                        text-align: center;
                        margin-bottom: 30px;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-radius: 10px;
                    }
                    .header-title {
                        margin: 0;
                        font-size: 2.5rem;
                        font-weight: 700;
                    }
                    .header-subtitle {
                        margin: 10px 0 0 0;
                        font-size: 1.2rem;
                        opacity: 0.9;
                    }
                    .controls-container {
                        margin-bottom: 30px;
                    }
                    .control-panel {
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        border: 1px solid #e9ecef;
                    }
                    .control-group {
                        margin-bottom: 20px;
                    }
                    .upload-area {
                        border: 2px dashed #007bff;
                        border-radius: 5px;
                        padding: 20px;
                        text-align: center;
                        margin: 10px 0;
                        cursor: pointer;
                        background: #f8f9ff;
                    }
                    .upload-area:hover {
                        background: #e6f3ff;
                    }
                    .btn {
                        padding: 10px 20px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    .btn-primary {
                        background: #007bff;
                        color: white;
                    }
                    .btn-primary:hover {
                        background: #0056b3;
                    }
                    .main-content {
                        background: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }
                    .metric-card {
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 8px;
                        text-align: center;
                        margin: 10px;
                        border: 1px solid #e9ecef;
                    }
                    .metric-value {
                        font-size: 2rem;
                        font-weight: 700;
                        color: #007bff;
                        margin: 0;
                    }
                    .metric-label {
                        font-size: 0.9rem;
                        color: #6c757d;
                        margin: 5px 0 0 0;
                    }
                    .status {
                        margin-top: 15px;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    .status-success {
                        background: #d4edda;
                        color: #155724;
                        border: 1px solid #c3e6cb;
                    }
                    .status-error {
                        background: #f8d7da;
                        color: #721c24;
                        border: 1px solid #f5c6cb;
                    }
                    .insights-section {
                        margin: 20px 0;
                    }
                    .insight-category {
                        background: #e9ecef;
                        padding: 15px;
                        border-radius: 8px;
                        margin: 15px 0;
                    }
                    .insight-item {
                        background: white;
                        padding: 10px;
                        margin: 8px 0;
                        border-radius: 5px;
                        border-left: 4px solid #007bff;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        # Toggle upload section based on data source
        @self.app.callback(
            Output('upload-section', 'style'),
            [Input('data-source', 'value')]
        )
        def toggle_upload_section(data_source):
            if data_source == 'upload':
                return {'display': 'block'}
            return {'display': 'none'}
        
        # Load analytics data
        @self.app.callback(
            [Output('analytics-data', 'data'),
             Output('status-indicator', 'children'),
             Output('status-indicator', 'className')],
            [Input('load-button', 'n_clicks')],
            [State('data-source', 'value'),
             State('upload-videos', 'contents'),
             State('upload-subscribers', 'contents')]
        )
        def load_analytics_data(n_clicks, data_source, videos_content, subs_content):
            if n_clicks is None:
                return {}, "", "status"
            
            try:
                if data_source == 'sample':
                    self.analytics = YouTubeAnalytics(
                        videos_file="data/sample/videos.csv",
                        subscribers_file="data/sample/subscribers.csv"
                    )
                else:
                    # Handle uploaded files (simplified for this example)
                    if videos_content is None:
                        return {}, "❌ Please upload videos CSV file", "status status-error"
                    
                    # In a real implementation, you'd decode and save the uploaded files
                    return {}, "❌ File upload not implemented in this demo", "status status-error"
                
                # Load data
                self.analytics.load_data()
                
                # Generate summary statistics
                summary = self.analytics.generate_summary_statistics()
                
                return summary, "✅ Data loaded successfully!", "status status-success"
                
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                return {}, f"❌ Error loading data: {str(e)}", "status status-error"
        
        # Display overview metrics
        @self.app.callback(
            Output('overview-metrics', 'children'),
            [Input('analytics-data', 'data')]
        )
        def display_overview_metrics(analytics_data):
            if not analytics_data:
                return html.Div("No data loaded yet.", className="text-center")
            
            try:
                overview = analytics_data['overview']
                engagement = analytics_data['engagement_metrics']
                
                metrics = [
                    {'label': 'Total Videos', 'value': f"{overview['total_videos']:,}"},
                    {'label': 'Total Views', 'value': f"{overview['total_views']:,}"},
                    {'label': 'Total Likes', 'value': f"{overview['total_likes']:,}"},
                    {'label': 'Total Comments', 'value': f"{overview['total_comments']:,}"},
                    {'label': 'Avg Like Rate', 'value': f"{engagement['average_like_rate']:.2f}%"},
                    {'label': 'Avg Comment Rate', 'value': f"{engagement['average_comment_rate']:.2f}%"},
                    {'label': 'Avg Engagement', 'value': f"{engagement['average_engagement_rate']:.2f}%"},
                    {'label': 'Median Views', 'value': f"{engagement['median_views']:,.0f}"}
                ]
                
                metric_cards = []
                for metric in metrics:
                    card = html.Div([
                        html.P(metric['value'], className="metric-value"),
                        html.P(metric['label'], className="metric-label")
                    ], className="metric-card")
                    metric_cards.append(card)
                
                return html.Div([
                    html.H3("📊 Channel Overview"),
                    html.Div(metric_cards, style={
                        'display': 'grid',
                        'grid-template-columns': 'repeat(auto-fit, minmax(200px, 1fr))',
                        'gap': '10px',
                        'margin': '20px 0'
                    }),
                    html.P(f"📅 Data range: {overview['date_range']['start']} to {overview['date_range']['end']}",
                          style={'text-align': 'center', 'margin-top': '20px', 'color': '#6c757d'})
                ])
                
            except Exception as e:
                logger.error(f"Error displaying metrics: {e}")
                return html.Div(f"Error displaying metrics: {str(e)}", style={'color': 'red'})
        
        # Handle tab content
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('analytics-data', 'data')],
            [State('analysis-options', 'value')]
        )
        def display_tab_content(active_tab, analytics_data, analysis_options):
            if not analytics_data:
                return html.Div("Please load data first.", className="text-center")
            
            if active_tab == 'overview':
                return self.create_overview_tab()
            elif active_tab == 'viz':
                return self.create_visualizations_tab()
            elif active_tab == 'ml':
                if 'ml' in (analysis_options or []):
                    return self.create_ml_tab()
                else:
                    return html.Div("ML Predictions disabled in controls.", className="text-center")
            elif active_tab == 'insights':
                if 'insights' in (analysis_options or []):
                    return self.create_insights_tab()
                else:
                    return html.Div("Insights disabled in controls.", className="text-center")
            elif active_tab == 'export':
                return self.create_export_tab()
            
            return html.Div("Select a tab to view content.")
    
    def create_overview_tab(self):
        """Create overview tab content."""
        if not self.analytics:
            return html.Div("No analytics data available.")
        
        try:
            # Create basic charts
            charts = self.analytics.create_all_visualizations()
            
            return html.Div([
                html.H3("📈 Key Visualizations"),
                
                # Views timeline
                html.Div([
                    html.H4("Views Over Time"),
                    dcc.Graph(figure=charts.get('views_timeline', {}))
                ]),
                
                # Engagement comparison
                html.Div([
                    html.H4("Engagement Metrics"),
                    dcc.Graph(figure=charts.get('engagement_comparison', {}))
                ]),
                
                # Top performers
                html.Div([
                    html.H4("Top Performing Videos"),
                    dcc.Graph(figure=charts.get('top_performers', {}))
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error creating overview tab: {e}")
            return html.Div(f"Error creating overview: {str(e)}", style={'color': 'red'})
    
    def create_visualizations_tab(self):
        """Create visualizations tab content."""
        if not self.analytics:
            return html.Div("No analytics data available.")
        
        try:
            charts = self.analytics.create_all_visualizations()
            
            return html.Div([
                html.H3("📊 Detailed Analytics"),
                
                # Correlation heatmap
                html.Div([
                    html.H4("Performance Correlation"),
                    dcc.Graph(figure=charts.get('correlation_heatmap', {}))
                ]),
                
                # Performance scatter
                html.Div([
                    html.H4("Views vs Engagement"),
                    dcc.Graph(figure=charts.get('performance_scatter', {}))
                ]),
                
                # Distribution histograms
                html.Div([
                    html.H4("Data Distributions"),
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=charts.get('views_distribution', {}))
                        ], style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Graph(figure=charts.get('engagement_distribution', {}))
                        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                    ])
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error creating visualizations tab: {e}")
            return html.Div(f"Error creating visualizations: {str(e)}", style={'color': 'red'})
    
    def create_ml_tab(self):
        """Create ML predictions tab content."""
        if not self.analytics:
            return html.Div("No analytics data available.")
        
        try:
            # Train model
            training_results = self.analytics.train_prediction_model(hyperparameter_tuning=True)
            
            return html.Div([
                html.H3("🤖 Machine Learning Predictions"),
                
                # Model performance
                html.Div([
                    html.H4("Model Performance"),
                    html.Div([
                        html.Div([
                            html.P(f"R² Score: {training_results['performance_metrics']['r2_score']:.3f}"),
                            html.P(f"MAE: {training_results['performance_metrics']['mae']:.0f}"),
                            html.P(f"RMSE: {training_results['performance_metrics']['rmse']:.0f}")
                        ], className="metric-card")
                    ])
                ]),
                
                # Prediction interface
                html.Div([
                    html.H4("Predict Video Performance"),
                    html.Div([
                        html.Div([
                            html.Label("Duration (minutes):"),
                            dcc.Slider(id='duration-slider', min=1, max=60, value=10, marks={i: str(i) for i in range(0, 61, 10)})
                        ], style={'margin': '10px'}),
                        html.Div([
                            html.Label("Expected Likes:"),
                            dcc.Input(id='likes-input', type='number', value=100, style={'width': '100%'})
                        ], style={'margin': '10px'}),
                        html.Div([
                            html.Label("Expected Comments:"),
                            dcc.Input(id='comments-input', type='number', value=20, style={'width': '100%'})
                        ], style={'margin': '10px'}),
                        html.Button("🎯 Predict Views", id="predict-button", className="btn btn-primary"),
                        html.Div(id="prediction-result", style={'margin-top': '20px'})
                    ])
                ])
            ])
            
        except Exception as e:
            logger.error(f"Error creating ML tab: {e}")
            return html.Div(f"Error creating ML predictions: {str(e)}", style={'color': 'red'})
    
    def create_insights_tab(self):
        """Create insights tab content."""
        if not self.analytics:
            return html.Div("No analytics data available.")
        
        try:
            insights = self.analytics.generate_insights()
            
            insight_sections = []
            for category, recommendations in insights.items():
                if recommendations and category != 'error':
                    section = html.Div([
                        html.H4(category.replace('_', ' ').title()),
                        html.Div([
                            html.Div(f"• {rec}", className="insight-item")
                            for rec in recommendations
                        ])
                    ], className="insight-category")
                    insight_sections.append(section)
            
            return html.Div([
                html.H3("💡 Insights & Recommendations"),
                html.Div(insight_sections, className="insights-section")
            ])
            
        except Exception as e:
            logger.error(f"Error creating insights tab: {e}")
            return html.Div(f"Error generating insights: {str(e)}", style={'color': 'red'})
    
    def create_export_tab(self):
        """Create export tab content."""
        return html.Div([
            html.H3("📤 Export Options"),
            html.Div([
                html.Button("📊 Export to Excel", id="export-excel", className="btn btn-primary", style={'margin': '10px'}),
                html.Button("📈 Save Charts", id="export-charts", className="btn btn-primary", style={'margin': '10px'}),
                html.Button("🤖 Save ML Model", id="export-model", className="btn btn-primary", style={'margin': '10px'})
            ]),
            html.Div(id="export-status", style={'margin-top': '20px'})
        ])
    
    def run(self):
        """Run the Dash dashboard."""
        logger.info(f"Starting Dash dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=self.debug)

def main():
    """Main function to run the Dash dashboard."""
    dashboard = DashDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
