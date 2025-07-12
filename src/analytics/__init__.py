"""
YouTube Analytics Package
A comprehensive toolkit for analyzing YouTube Studio data.
"""

__version__ = "1.0.0"
__author__ = "Duong Cao Nhan"
__email__ = "duongcaonhan@example.com"

from .core import YouTubeAnalytics
from .data_loader import DataLoader
from .visualizations import ChartGenerator
from .ml_predictor import MLPredictor

__all__ = [
    "YouTubeAnalytics",
    "DataLoader", 
    "ChartGenerator",
    "MLPredictor"
]
