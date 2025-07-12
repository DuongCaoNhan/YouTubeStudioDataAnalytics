"""
Data Loading Module for YouTube Analytics
Handles loading and preprocessing of YouTube Studio CSV data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and preprocessing YouTube Studio data from CSV files.
    """
    
    def __init__(self, videos_file: str = "data/sample/videos.csv", 
                 subscribers_file: str = "data/sample/subscribers.csv"):
        """
        Initialize the DataLoader.
        
        Args:
            videos_file: Path to the videos CSV file
            subscribers_file: Path to the subscribers CSV file
        """
        self.videos_file = videos_file
        self.subscribers_file = subscribers_file
        self.videos_df: Optional[pd.DataFrame] = None
        self.subscribers_df: Optional[pd.DataFrame] = None
    
    def load_videos_data(self) -> pd.DataFrame:
        """
        Load and preprocess videos data from CSV.
        
        Returns:
            Processed videos DataFrame
            
        Raises:
            FileNotFoundError: If the videos file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            logger.info(f"Loading videos data from {self.videos_file}")
            
            # Check if file exists
            if not Path(self.videos_file).exists():
                raise FileNotFoundError(f"Videos file not found: {self.videos_file}")
            
            # Load data
            self.videos_df = pd.read_csv(self.videos_file)
            logger.info(f"Loaded {len(self.videos_df)} videos from {self.videos_file}")
            
            # Validate required columns
            required_columns = ['Title', 'Publish Date', 'Views', 'Likes', 'Comments']
            missing_columns = [col for col in required_columns if col not in self.videos_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Preprocess data
            self._preprocess_videos_data()
            
            logger.info("Videos data preprocessing completed")
            return self.videos_df
            
        except Exception as e:
            logger.error(f"Error loading videos data: {e}")
            raise
    
    def load_subscribers_data(self) -> Optional[pd.DataFrame]:
        """
        Load and preprocess subscribers data from CSV.
        
        Returns:
            Processed subscribers DataFrame or None if file doesn't exist
        """
        try:
            if not Path(self.subscribers_file).exists():
                logger.warning(f"Subscribers file not found: {self.subscribers_file}")
                return None
            
            logger.info(f"Loading subscribers data from {self.subscribers_file}")
            self.subscribers_df = pd.read_csv(self.subscribers_file)
            logger.info(f"Loaded {len(self.subscribers_df)} subscriber records")
            
            # Validate required columns
            required_columns = ['Date', 'Subscribers Gained', 'Subscribers Lost']
            missing_columns = [col for col in required_columns if col not in self.subscribers_df.columns]
            
            if missing_columns:
                logger.warning(f"Missing subscriber columns: {missing_columns}")
                return None
            
            # Preprocess data
            self._preprocess_subscribers_data()
            
            logger.info("Subscribers data preprocessing completed")
            return self.subscribers_df
            
        except Exception as e:
            logger.error(f"Error loading subscribers data: {e}")
            return None
    
    def _preprocess_videos_data(self) -> None:
        """Preprocess videos data with calculated metrics."""
        # Convert date column
        self.videos_df['Publish Date'] = pd.to_datetime(self.videos_df['Publish Date'])
        
        # Sort by date
        self.videos_df = self.videos_df.sort_values('Publish Date')
        
        # Handle missing duration column
        if 'Duration (minutes)' not in self.videos_df.columns:
            # Generate random durations if not provided (for demo purposes)
            np.random.seed(42)
            self.videos_df['Duration (minutes)'] = np.random.uniform(5, 30, len(self.videos_df))
            logger.warning("Duration column missing, generated random values for demo")
        
        # Calculate engagement metrics
        self.videos_df['Like Rate (%)'] = (self.videos_df['Likes'] / self.videos_df['Views']) * 100
        self.videos_df['Comment Rate (%)'] = (self.videos_df['Comments'] / self.videos_df['Views']) * 100
        self.videos_df['Engagement Rate (%)'] = (
            (self.videos_df['Likes'] + self.videos_df['Comments']) / self.videos_df['Views']
        ) * 100
        
        # Handle any infinite or NaN values
        self.videos_df = self.videos_df.replace([np.inf, -np.inf], np.nan)
        self.videos_df = self.videos_df.fillna(0)
        
        # Add additional derived metrics
        self.videos_df['Days Since Publication'] = (
            datetime.now() - self.videos_df['Publish Date']
        ).dt.days
        
        self.videos_df['Views per Day'] = self.videos_df['Views'] / (
            self.videos_df['Days Since Publication'] + 1
        )
        
        logger.info("Videos data preprocessing completed with calculated metrics")
    
    def _preprocess_subscribers_data(self) -> None:
        """Preprocess subscribers data."""
        # Convert date column
        self.subscribers_df['Date'] = pd.to_datetime(self.subscribers_df['Date'])
        
        # Sort by date
        self.subscribers_df = self.subscribers_df.sort_values('Date')
        
        # Calculate net subscribers if not present
        if 'Net Subscribers' not in self.subscribers_df.columns:
            self.subscribers_df['Net Subscribers'] = (
                self.subscribers_df['Subscribers Gained'] - self.subscribers_df['Subscribers Lost']
            )
        
        # Calculate cumulative metrics
        self.subscribers_df['Cumulative Net Growth'] = self.subscribers_df['Net Subscribers'].cumsum()
        self.subscribers_df['Growth Rate (%)'] = (
            self.subscribers_df['Subscribers Gained'] / 
            (self.subscribers_df['Subscribers Gained'] + self.subscribers_df['Subscribers Lost'])
        ) * 100
        
        # Handle any infinite or NaN values
        self.subscribers_df = self.subscribers_df.replace([np.inf, -np.inf], np.nan)
        self.subscribers_df = self.subscribers_df.fillna(0)
        
        logger.info("Subscribers data preprocessing completed")
    
    def load_all_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load both videos and subscribers data.
        
        Returns:
            Tuple of (videos_df, subscribers_df)
        """
        videos_df = self.load_videos_data()
        subscribers_df = self.load_subscribers_data()
        return videos_df, subscribers_df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary information about loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {}
        
        if self.videos_df is not None:
            summary['videos'] = {
                'count': len(self.videos_df),
                'date_range': {
                    'start': self.videos_df['Publish Date'].min().strftime('%Y-%m-%d'),
                    'end': self.videos_df['Publish Date'].max().strftime('%Y-%m-%d')
                },
                'total_views': int(self.videos_df['Views'].sum()),
                'total_likes': int(self.videos_df['Likes'].sum()),
                'total_comments': int(self.videos_df['Comments'].sum()),
                'avg_engagement_rate': float(self.videos_df['Engagement Rate (%)'].mean())
            }
        
        if self.subscribers_df is not None:
            summary['subscribers'] = {
                'count': len(self.subscribers_df),
                'date_range': {
                    'start': self.subscribers_df['Date'].min().strftime('%Y-%m-%d'),
                    'end': self.subscribers_df['Date'].max().strftime('%Y-%m-%d')
                },
                'total_gained': int(self.subscribers_df['Subscribers Gained'].sum()),
                'total_lost': int(self.subscribers_df['Subscribers Lost'].sum()),
                'net_growth': int(self.subscribers_df['Net Subscribers'].sum())
            }
        
        return summary
    
    def export_processed_data(self, output_dir: str = "data/exports") -> None:
        """
        Export processed data to CSV files.
        
        Args:
            output_dir: Directory to save exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.videos_df is not None:
            videos_export_path = output_path / "processed_videos.csv"
            self.videos_df.to_csv(videos_export_path, index=False)
            logger.info(f"Exported processed videos data to {videos_export_path}")
        
        if self.subscribers_df is not None:
            subscribers_export_path = output_path / "processed_subscribers.csv"
            self.subscribers_df.to_csv(subscribers_export_path, index=False)
            logger.info(f"Exported processed subscribers data to {subscribers_export_path}")
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and return issues found.
        
        Returns:
            Dictionary with data quality report
        """
        quality_report = {
            'videos': {'issues': [], 'quality_score': 100},
            'subscribers': {'issues': [], 'quality_score': 100}
        }
        
        if self.videos_df is not None:
            # Check for missing values
            missing_values = self.videos_df.isnull().sum()
            if missing_values.any():
                quality_report['videos']['issues'].append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
                quality_report['videos']['quality_score'] -= 10
            
            # Check for outliers in views
            q1 = self.videos_df['Views'].quantile(0.25)
            q3 = self.videos_df['Views'].quantile(0.75)
            iqr = q3 - q1
            outliers = self.videos_df[
                (self.videos_df['Views'] < q1 - 1.5 * iqr) | 
                (self.videos_df['Views'] > q3 + 1.5 * iqr)
            ]
            if len(outliers) > 0:
                quality_report['videos']['issues'].append(f"Found {len(outliers)} potential outliers in views")
                quality_report['videos']['quality_score'] -= 5
        
        if self.subscribers_df is not None:
            # Check for negative values where they shouldn't be
            if (self.subscribers_df['Subscribers Gained'] < 0).any():
                quality_report['subscribers']['issues'].append("Negative values found in Subscribers Gained")
                quality_report['subscribers']['quality_score'] -= 15
            
            if (self.subscribers_df['Subscribers Lost'] < 0).any():
                quality_report['subscribers']['issues'].append("Negative values found in Subscribers Lost")
                quality_report['subscribers']['quality_score'] -= 15
        
        return quality_report
