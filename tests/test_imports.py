"""
Basic test to verify the modular structure works.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

class TestModularStructure(unittest.TestCase):
    """Test that all modules can be imported correctly."""
    
    def test_import_analytics_core(self):
        """Test importing core analytics module."""
        try:
            from analytics import YouTubeAnalytics
            self.assertTrue(True, "Core analytics imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core analytics: {e}")
    
    def test_import_data_loader(self):
        """Test importing data loader module."""
        try:
            from analytics import DataLoader
            self.assertTrue(True, "DataLoader imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import DataLoader: {e}")
    
    def test_import_visualizations(self):
        """Test importing visualizations module."""
        try:
            from analytics import ChartGenerator
            self.assertTrue(True, "ChartGenerator imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import ChartGenerator: {e}")
    
    def test_import_ml_predictor(self):
        """Test importing ML predictor module."""
        try:
            from analytics import MLPredictor
            self.assertTrue(True, "MLPredictor imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import MLPredictor: {e}")
    
    def test_import_utils(self):
        """Test importing utility modules."""
        try:
            from utils import Config, DataValidator, ExportManager
            self.assertTrue(True, "Utils imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import utils: {e}")
    
    def test_import_dashboards(self):
        """Test importing dashboard modules."""
        try:
            from dashboards import StreamlitDashboard, DashDashboard
            self.assertTrue(True, "Dashboards imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import dashboards: {e}")

if __name__ == '__main__':
    unittest.main()
