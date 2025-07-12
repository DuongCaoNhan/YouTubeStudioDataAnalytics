"""
Data Utilities
Helper functions for data validation, processing, and statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation utilities."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return validation_results
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_info = missing_data[missing_data > 0].to_dict()
            validation_results['warnings'].append(f"Missing values found: {missing_info}")
        
        # Data type checks
        for col in df.columns:
            if col in required_columns:
                # Check numeric columns
                if 'Views' in col or 'Likes' in col or 'Comments' in col or 'Rate' in col:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation_results['errors'].append(f"Column '{col}' should be numeric")
                
                # Check date columns
                if 'Date' in col:
                    try:
                        pd.to_datetime(df[col])
                    except:
                        validation_results['errors'].append(f"Column '{col}' contains invalid dates")
        
        # Summary statistics
        validation_results['summary'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': duplicates,
            'missing_values': missing_data.sum()
        }
        
        return validation_results
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers in a pandas Series.
        
        Args:
            series: Pandas Series to analyze
            method: Method to use ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            return pd.Series(z_scores > threshold, index=series.index)
        
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def clean_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Clean numeric columns by removing non-numeric values.
        
        Args:
            df: DataFrame to clean
            columns: List of column names to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        
        for col in columns:
            if col in df_cleaned.columns:
                # Convert to numeric, coercing errors to NaN
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                
                # Fill NaN values with 0 for count-based metrics
                if any(keyword in col.lower() for keyword in ['views', 'likes', 'comments', 'subscribers']):
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                
                # Fill NaN values with median for rate-based metrics
                elif 'rate' in col.lower():
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
        
        return df_cleaned
    
    @staticmethod
    def validate_date_range(df: pd.DataFrame, date_column: str, 
                          min_date: Optional[datetime] = None,
                          max_date: Optional[datetime] = None) -> bool:
        """
        Validate date range in DataFrame.
        
        Args:
            df: DataFrame to validate
            date_column: Name of date column
            min_date: Minimum allowed date
            max_date: Maximum allowed date
            
        Returns:
            True if date range is valid
        """
        if date_column not in df.columns:
            return False
        
        try:
            dates = pd.to_datetime(df[date_column])
            
            if min_date and dates.min() < min_date:
                logger.warning(f"Dates before {min_date} found in {date_column}")
                return False
            
            if max_date and dates.max() > max_date:
                logger.warning(f"Dates after {max_date} found in {date_column}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating date range: {e}")
            return False


class DateUtils:
    """Date utility functions."""
    
    @staticmethod
    def parse_youtube_date(date_str: str) -> datetime:
        """
        Parse YouTube date string to datetime.
        
        Args:
            date_str: Date string from YouTube
            
        Returns:
            Parsed datetime object
        """
        # Common YouTube date formats
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try pandas parser as fallback
        try:
            return pd.to_datetime(date_str)
        except:
            raise ValueError(f"Could not parse date: {date_str}")
    
    @staticmethod
    def get_date_range_stats(df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """
        Get statistics about date range in DataFrame.
        
        Args:
            df: DataFrame with date column
            date_column: Name of date column
            
        Returns:
            Dictionary with date range statistics
        """
        dates = pd.to_datetime(df[date_column])
        
        return {
            'start_date': dates.min(),
            'end_date': dates.max(),
            'total_days': (dates.max() - dates.min()).days,
            'unique_dates': dates.nunique(),
            'missing_dates': dates.isnull().sum(),
            'date_frequency': dates.dt.date.value_counts().describe().to_dict()
        }
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Add time-based features to DataFrame.
        
        Args:
            df: DataFrame with date column
            date_column: Name of date column
            
        Returns:
            DataFrame with additional time features
        """
        df_enhanced = df.copy()
        dates = pd.to_datetime(df_enhanced[date_column])
        
        # Basic time features
        df_enhanced[f'{date_column}_year'] = dates.dt.year
        df_enhanced[f'{date_column}_month'] = dates.dt.month
        df_enhanced[f'{date_column}_day'] = dates.dt.day
        df_enhanced[f'{date_column}_dayofweek'] = dates.dt.dayofweek
        df_enhanced[f'{date_column}_dayofyear'] = dates.dt.dayofyear
        df_enhanced[f'{date_column}_quarter'] = dates.dt.quarter
        df_enhanced[f'{date_column}_week'] = dates.dt.isocalendar().week
        
        # Categorical features
        df_enhanced[f'{date_column}_day_name'] = dates.dt.day_name()
        df_enhanced[f'{date_column}_month_name'] = dates.dt.month_name()
        df_enhanced[f'{date_column}_is_weekend'] = dates.dt.dayofweek >= 5
        
        # Relative features
        min_date = dates.min()
        df_enhanced[f'{date_column}_days_since_start'] = (dates - min_date).dt.days
        
        return df_enhanced
    
    @staticmethod
    def resample_time_series(df: pd.DataFrame, date_column: str, 
                           value_column: str, frequency: str = 'D',
                           aggregation: str = 'sum') -> pd.DataFrame:
        """
        Resample time series data.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column to aggregate
            frequency: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregation: Aggregation method ('sum', 'mean', 'median', 'max', 'min')
            
        Returns:
            Resampled DataFrame
        """
        df_ts = df.copy()
        df_ts[date_column] = pd.to_datetime(df_ts[date_column])
        df_ts = df_ts.set_index(date_column)
        
        if aggregation == 'sum':
            return df_ts[value_column].resample(frequency).sum().reset_index()
        elif aggregation == 'mean':
            return df_ts[value_column].resample(frequency).mean().reset_index()
        elif aggregation == 'median':
            return df_ts[value_column].resample(frequency).median().reset_index()
        elif aggregation == 'max':
            return df_ts[value_column].resample(frequency).max().reset_index()
        elif aggregation == 'min':
            return df_ts[value_column].resample(frequency).min().reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")


class StatisticsUtils:
    """Statistical utility functions."""
    
    @staticmethod
    def calculate_correlation_matrix(df: pd.DataFrame, 
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Args:
            df: DataFrame to analyze
            columns: Specific columns to include (if None, use all numeric)
            
        Returns:
            Correlation matrix DataFrame
        """
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        return numeric_df.corr()
    
    @staticmethod
    def calculate_descriptive_stats(df: pd.DataFrame, 
                                  columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            df: DataFrame to analyze
            columns: Specific columns to analyze
            
        Returns:
            Dictionary with descriptive statistics
        """
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        stats_dict = {}
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            
            stats_dict[col] = {
                'count': len(series),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'cv': float(series.std() / series.mean()) if series.mean() != 0 else 0
            }
        
        return stats_dict
    
    @staticmethod
    def perform_normality_test(series: pd.Series, test: str = 'shapiro') -> Dict[str, Any]:
        """
        Perform normality test on data series.
        
        Args:
            series: Data series to test
            test: Test to perform ('shapiro', 'normaltest', 'jarque_bera')
            
        Returns:
            Dictionary with test results
        """
        clean_series = series.dropna()
        
        if len(clean_series) < 3:
            return {'error': 'Insufficient data for normality test'}
        
        try:
            if test == 'shapiro':
                if len(clean_series) > 5000:
                    # Shapiro-Wilk test is not reliable for large samples
                    clean_series = clean_series.sample(5000)
                statistic, p_value = stats.shapiro(clean_series)
                test_name = 'Shapiro-Wilk'
            
            elif test == 'normaltest':
                statistic, p_value = stats.normaltest(clean_series)
                test_name = "D'Agostino-Pearson"
            
            elif test == 'jarque_bera':
                statistic, p_value = stats.jarque_bera(clean_series)
                test_name = 'Jarque-Bera'
            
            else:
                raise ValueError(f"Unknown test: {test}")
            
            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05,
                'interpretation': 'Normal distribution' if p_value > 0.05 else 'Non-normal distribution'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def calculate_growth_rates(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate growth rates for a time series.
        
        Args:
            series: Time series data
            periods: Number of periods for growth calculation
            
        Returns:
            Series with growth rates
        """
        return series.pct_change(periods=periods) * 100
    
    @staticmethod
    def calculate_moving_averages(series: pd.Series, windows: List[int]) -> pd.DataFrame:
        """
        Calculate moving averages for different window sizes.
        
        Args:
            series: Time series data
            windows: List of window sizes
            
        Returns:
            DataFrame with moving averages
        """
        ma_df = pd.DataFrame(index=series.index)
        ma_df['original'] = series
        
        for window in windows:
            ma_df[f'ma_{window}'] = series.rolling(window=window).mean()
        
        return ma_df
    
    @staticmethod
    def detect_trend(series: pd.Series, method: str = 'linear') -> Dict[str, Any]:
        """
        Detect trend in time series data.
        
        Args:
            series: Time series data
            method: Method to use ('linear', 'seasonal')
            
        Returns:
            Dictionary with trend analysis
        """
        clean_series = series.dropna()
        x = np.arange(len(clean_series))
        
        if method == 'linear':
            # Linear regression to detect trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_series)
            
            return {
                'trend_slope': float(slope),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                'trend_strength': float(abs(r_value)),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'is_significant': p_value < 0.05
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def calculate_confidence_interval(data: Union[pd.Series, np.ndarray], 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for data.
        
        Args:
            data: Data series or array
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple with (lower_bound, upper_bound)
        """
        clean_data = np.array(data).flatten()
        clean_data = clean_data[~np.isnan(clean_data)]
        
        if len(clean_data) == 0:
            return (np.nan, np.nan)
        
        mean = np.mean(clean_data)
        std_err = stats.sem(clean_data)
        
        # Use t-distribution for small samples
        if len(clean_data) < 30:
            t_val = stats.t.ppf((1 + confidence) / 2, len(clean_data) - 1)
            margin_error = t_val * std_err
        else:
            z_val = stats.norm.ppf((1 + confidence) / 2)
            margin_error = z_val * std_err
        
        return (mean - margin_error, mean + margin_error)
