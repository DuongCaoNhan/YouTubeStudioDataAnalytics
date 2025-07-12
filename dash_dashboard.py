"""
YouTube Analytics Dash Dashboard
Interactive web dashboard using Dash and Plotly for YouTube Studio analytics.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date
from youtube_analytics import YouTubeAnalytics
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "YouTube Analytics Dashboard"

# Load data
analytics = YouTubeAnalytics()
analytics.load_data()

# Custom CSS styles
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define colors
colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'youtube_red': '#FF0000',
    'like_green': '#00FF00',
    'comment_blue': '#0099FF'
}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('📺 YouTube Analytics Dashboard', 
                style={'textAlign': 'center', 'color': colors['youtube_red'], 'marginBottom': 30}),
        html.Hr(style={'borderColor': colors['youtube_red'], 'borderWidth': 2})
    ]),
    
    # Controls
    html.Div([
        html.Div([
            html.Label('Select Date Range:', style={'fontWeight': 'bold'}),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=analytics.videos_df['Publish Date'].min(),
                end_date=analytics.videos_df['Publish Date'].max(),
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('Select Metrics:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='metrics-dropdown',
                options=[
                    {'label': 'Views', 'value': 'Views'},
                    {'label': 'Likes', 'value': 'Likes'},
                    {'label': 'Comments', 'value': 'Comments'},
                    {'label': 'Like Rate (%)', 'value': 'Like Rate (%)'},
                    {'label': 'Comment Rate (%)', 'value': 'Comment Rate (%)'}
                ],
                value=['Views', 'Likes'],
                multi=True
            )
        ], className='three columns'),
        
        html.Div([
            html.Label('Analysis Type:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='analysis-type',
                options=[
                    {'label': 'Time Series', 'value': 'timeseries'},
                    {'label': 'Performance Comparison', 'value': 'comparison'},
                    {'label': 'Correlation Analysis', 'value': 'correlation'}
                ],
                value='timeseries'
            )
        ], className='three columns'),
        
        html.Div([
            html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                       style={'backgroundColor': colors['youtube_red'], 'color': 'white',
                             'border': 'none', 'padding': '10px 20px', 'marginTop': 25})
        ], className='three columns')
    ], className='row', style={'marginBottom': 30}),
    
    # Key metrics cards
    html.Div(id='metrics-cards', className='row', style={'marginBottom': 30}),
    
    # Main charts
    html.Div([
        html.Div([
            dcc.Graph(id='main-chart')
        ], className='eight columns'),
        
        html.Div([
            dcc.Graph(id='side-chart')
        ], className='four columns')
    ], className='row'),
    
    # Secondary charts
    html.Div([
        html.Div([
            dcc.Graph(id='engagement-chart')
        ], className='six columns'),
        
        html.Div([
            dcc.Graph(id='performance-chart')
        ], className='six columns')
    ], className='row'),
    
    # Data table
    html.Div([
        html.H3('📋 Video Performance Table', style={'color': colors['youtube_red']}),
        html.Div(id='data-table')
    ], style={'marginTop': 30}),
    
    # Store for filtered data
    dcc.Store(id='filtered-data')
])

@app.callback(
    Output('filtered-data', 'data'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def filter_data(start_date, end_date):
    """Filter data based on date range"""
    if start_date and end_date:
        filtered_df = analytics.videos_df[
            (analytics.videos_df['Publish Date'] >= start_date) &
            (analytics.videos_df['Publish Date'] <= end_date)
        ]
    else:
        filtered_df = analytics.videos_df
    
    return filtered_df.to_dict('records')

@app.callback(
    Output('metrics-cards', 'children'),
    [Input('filtered-data', 'data')]
)
def update_metrics_cards(filtered_data):
    """Update key metrics cards"""
    if not filtered_data:
        return []
    
    df = pd.DataFrame(filtered_data)
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])
    
    # Calculate metrics
    total_views = df['Views'].sum()
    total_likes = df['Likes'].sum()
    total_comments = df['Comments'].sum()
    avg_engagement = df['Engagement Rate (%)'].mean()
    
    # Create metric cards
    cards = [
        html.Div([
            html.H4(f"{total_views:,}", style={'color': colors['youtube_red'], 'marginBottom': 0}),
            html.P('Total Views', style={'marginTop': 0})
        ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'padding': 20, 'borderRadius': 5}),
        
        html.Div([
            html.H4(f"{total_likes:,}", style={'color': colors['like_green'], 'marginBottom': 0}),
            html.P('Total Likes', style={'marginTop': 0})
        ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'padding': 20, 'borderRadius': 5}),
        
        html.Div([
            html.H4(f"{total_comments:,}", style={'color': colors['comment_blue'], 'marginBottom': 0}),
            html.P('Total Comments', style={'marginTop': 0})
        ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'padding': 20, 'borderRadius': 5}),
        
        html.Div([
            html.H4(f"{avg_engagement:.2f}%", style={'color': '#FF6600', 'marginBottom': 0}),
            html.P('Avg Engagement', style={'marginTop': 0})
        ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'padding': 20, 'borderRadius': 5})
    ]
    
    return cards

@app.callback(
    Output('main-chart', 'figure'),
    [Input('filtered-data', 'data'),
     Input('metrics-dropdown', 'value'),
     Input('analysis-type', 'value')]
)
def update_main_chart(filtered_data, selected_metrics, analysis_type):
    """Update the main chart based on selections"""
    if not filtered_data or not selected_metrics:
        return {}
    
    df = pd.DataFrame(filtered_data)
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])
    df = df.sort_values('Publish Date')
    
    if analysis_type == 'timeseries':
        # Time series chart
        fig = go.Figure()
        
        colors_map = {
            'Views': colors['youtube_red'],
            'Likes': colors['like_green'],
            'Comments': colors['comment_blue'],
            'Like Rate (%)': '#FFD700',
            'Comment Rate (%)': '#FF69B4'
        }
        
        for metric in selected_metrics:
            fig.add_trace(go.Scatter(
                x=df['Publish Date'],
                y=df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors_map.get(metric, '#000000'), width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='📈 YouTube Metrics Over Time',
            xaxis_title='Publication Date',
            yaxis_title='Value',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
    elif analysis_type == 'comparison':
        # Performance comparison chart
        df['Short Title'] = df['Title'].str[:25] + '...'
        
        fig = go.Figure()
        
        colors_list = [colors['youtube_red'], colors['like_green'], colors['comment_blue'], '#FFD700', '#FF69B4']
        
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Short Title'],
                y=df[metric],
                marker_color=colors_list[i % len(colors_list)]
            ))
        
        fig.update_layout(
            title='📊 Performance Comparison Across Videos',
            xaxis_title='Video Title',
            yaxis_title='Value',
            barmode='group',
            xaxis_tickangle=-45
        )
        
    elif analysis_type == 'correlation':
        # Correlation heatmap
        metrics_for_corr = ['Views', 'Likes', 'Comments', 'Like Rate (%)', 'Comment Rate (%)', 'Duration (minutes)']
        corr_matrix = df[metrics_for_corr].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='🔥 Performance Correlation Matrix',
            width=600,
            height=600
        )
    
    return fig

@app.callback(
    Output('side-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_side_chart(filtered_data):
    """Update the side chart with top videos"""
    if not filtered_data:
        return {}
    
    df = pd.DataFrame(filtered_data)
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])
    
    # Top 5 videos by views
    top_videos = df.nlargest(5, 'Views')
    top_videos['Short Title'] = top_videos['Title'].str[:20] + '...'
    
    fig = px.bar(
        top_videos,
        x='Views',
        y='Short Title',
        orientation='h',
        title='🏆 Top 5 Videos by Views',
        color='Views',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    
    return fig

@app.callback(
    Output('engagement-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_engagement_chart(filtered_data):
    """Update engagement analysis chart"""
    if not filtered_data:
        return {}
    
    df = pd.DataFrame(filtered_data)
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])
    
    # Scatter plot: Views vs Engagement Rate
    fig = px.scatter(
        df,
        x='Views',
        y='Like Rate (%)',
        size='Comments',
        color='Engagement Rate (%)',
        hover_data=['Title'],
        title='💡 Views vs Engagement Rate',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    
    return fig

@app.callback(
    Output('performance-chart', 'figure'),
    [Input('filtered-data', 'data')]
)
def update_performance_chart(filtered_data):
    """Update performance distribution chart"""
    if not filtered_data:
        return {}
    
    df = pd.DataFrame(filtered_data)
    df['Publish Date'] = pd.to_datetime(df['Publish Date'])
    
    # Distribution of engagement rates
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['Like Rate (%)'],
        name='Like Rate (%)',
        opacity=0.7,
        marker_color=colors['like_green']
    ))
    
    fig.add_trace(go.Histogram(
        x=df['Comment Rate (%)'],
        name='Comment Rate (%)',
        opacity=0.7,
        marker_color=colors['comment_blue']
    ))
    
    fig.update_layout(
        title='📊 Engagement Rate Distribution',
        xaxis_title='Rate (%)',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400
    )
    
    return fig

@app.callback(
    Output('data-table', 'children'),
    [Input('filtered-data', 'data')]
)
def update_data_table(filtered_data):
    """Update the data table"""
    if not filtered_data:
        return "No data available"
    
    df = pd.DataFrame(filtered_data)
    df['Publish Date'] = pd.to_datetime(df['Publish Date']).dt.strftime('%Y-%m-%d')
    
    # Select columns for display
    display_columns = ['Title', 'Publish Date', 'Views', 'Likes', 'Comments', 'Like Rate (%)', 'Engagement Rate (%)']
    display_df = df[display_columns].sort_values('Views', ascending=False)
    
    # Create table
    table_header = [html.Thead([html.Tr([html.Th(col) for col in display_columns])])]
    
    table_body = [html.Tbody([
        html.Tr([
            html.Td(row[col] if col != 'Title' else row[col][:50] + ('...' if len(str(row[col])) > 50 else ''))
            for col in display_columns
        ]) for _, row in display_df.head(10).iterrows()
    ])]
    
    table = html.Table(
        table_header + table_body,
        className='table table-striped',
        style={'width': '100%', 'fontSize': 12}
    )
    
    return table

# Subscriber dashboard (if data available)
if analytics.subscribers_df is not None:
    @app.callback(
        Output('subscriber-chart', 'figure'),
        [Input('refresh-button', 'n_clicks')]
    )
    def update_subscriber_chart(n_clicks):
        """Update subscriber activity chart"""
        df_subs = analytics.subscribers_df
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_subs['Date'],
            y=df_subs['Subscribers Gained'],
            mode='lines+markers',
            name='Subscribers Gained',
            line=dict(color=colors['like_green'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_subs['Date'],
            y=df_subs['Subscribers Lost'],
            mode='lines+markers',
            name='Subscribers Lost',
            line=dict(color=colors['youtube_red'], width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_subs['Date'],
            y=df_subs['Net Subscribers'],
            mode='lines+markers',
            name='Net Change',
            line=dict(color=colors['comment_blue'], width=3)
        ))
        
        fig.update_layout(
            title='👥 Subscriber Activity Over Time',
            xaxis_title='Date',
            yaxis_title='Subscribers',
            hovermode='x unified'
        )
        
        return fig

def run_dashboard(debug=True, port=8050):
    """Run the Dash dashboard"""
    print("🚀 Starting YouTube Analytics Dash Dashboard...")
    print(f"📊 Dashboard will be available at: http://127.0.0.1:{port}")
    print("🔄 Press Ctrl+C to stop the server")
    
    app.run_server(debug=debug, port=port, host='127.0.0.1')

if __name__ == '__main__':
    run_dashboard()
