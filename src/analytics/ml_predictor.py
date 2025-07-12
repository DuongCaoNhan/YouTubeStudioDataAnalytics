"""
Machine Learning Prediction Module for YouTube Analytics
Handles training models and making predictions for video performance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Machine learning predictor for YouTube video performance.
    """
    
    def __init__(self, model_type: str = 'linear'):
        """
        Initialize the ML predictor.
        
        Args:
            model_type: Type of model ('linear', 'ridge', 'lasso', 'random_forest', 'gradient_boost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_column = 'Views'
        self.is_trained = False
        self.performance_metrics = {}
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the specified model type."""
        model_configs = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=10
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=6
            )
        }
        
        if self.model_type not in model_configs:
            logger.warning(f"Unknown model type {self.model_type}, defaulting to linear")
            self.model_type = 'linear'
        
        self.model = model_configs[self.model_type]
        logger.info(f"Initialized {self.model_type} model")
    
    def prepare_features(self, videos_df: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning.
        
        Args:
            videos_df: DataFrame with video data
            feature_columns: List of columns to use as features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            # Default feature columns
            if feature_columns is None:
                feature_columns = [
                    'Duration (minutes)', 
                    'Like Rate (%)', 
                    'Comment Rate (%)',
                    'Days Since Publication'
                ]
            
            # Filter available columns
            available_features = [col for col in feature_columns if col in videos_df.columns]
            
            if len(available_features) == 0:
                raise ValueError("No valid feature columns found in the data")
            
            # Create feature matrix
            X = videos_df[available_features].copy()
            y = videos_df[self.target_column].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Remove rows with invalid target values
            valid_mask = (y > 0) & (~y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Store feature names
            self.feature_names = available_features
            
            # Add polynomial features for linear models
            if self.model_type in ['linear', 'ridge', 'lasso']:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly.fit_transform(X)
                feature_names_poly = poly.get_feature_names_out(self.feature_names)
                X = pd.DataFrame(X_poly, columns=feature_names_poly, index=X.index)
                self.feature_names = list(feature_names_poly)
            
            logger.info(f"Prepared features: {self.feature_names}")
            logger.info(f"Feature matrix shape: {X.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def train_model(self, videos_df: pd.DataFrame, 
                   feature_columns: Optional[List[str]] = None,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   perform_hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train the machine learning model.
        
        Args:
            videos_df: DataFrame with video data
            feature_columns: List of columns to use as features
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            perform_hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info(f"Training {self.model_type} model...")
            
            # Prepare features
            X, y = self.prepare_features(videos_df, feature_columns)
            
            if len(X) < 5:
                raise ValueError("Insufficient data for training (need at least 5 samples)")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features for linear models
            if self.model_type in ['linear', 'ridge', 'lasso']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Hyperparameter tuning
            if perform_hyperparameter_tuning:
                self._tune_hyperparameters(X_train_scaled, y_train)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=min(5, len(X_train)), scoring='r2'
            )
            metrics['cv_r2_mean'] = float(cv_scores.mean())
            metrics['cv_r2_std'] = float(cv_scores.std())
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                metrics['feature_importance'] = feature_importance
            elif hasattr(self.model, 'coef_'):
                feature_importance = dict(zip(self.feature_names, abs(self.model.coef_)))
                metrics['feature_importance'] = feature_importance
            
            self.performance_metrics = metrics
            self.is_trained = True
            
            logger.info(f"Model training completed. R² score: {metrics['test_r2']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grids = {
            'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        if self.model_type in param_grids:
            logger.info(f"Performing hyperparameter tuning for {self.model_type}")
            
            grid_search = GridSearchCV(
                self.model, 
                param_grids[self.model_type],
                cv=min(3, len(X_train) // 2),
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
    
    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray,
                          y_test: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
            'train_mape': float(np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100),
            'test_mape': float(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100)
        }
    
    def predict_views(self, video_features: Dict[str, float],
                     return_confidence: bool = True) -> Dict[str, Any]:
        """
        Predict views for a new video.
        
        Args:
            video_features: Dictionary with feature values
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with prediction and confidence interval
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Prepare features
            feature_values = []
            for feature in self.feature_names:
                if feature in video_features:
                    feature_values.append(video_features[feature])
                else:
                    # Use mean value for missing features
                    feature_values.append(0.0)  # or could use historical mean
            
            # Create feature array
            X_pred = np.array([feature_values])
            
            # Scale features if needed
            if self.model_type in ['linear', 'ridge', 'lasso']:
                X_pred = self.scaler.transform(X_pred)
            
            # Make prediction
            prediction = self.model.predict(X_pred)[0]
            
            result = {
                'predicted_views': float(max(0, prediction)),  # Ensure non-negative
                'model_type': self.model_type,
                'features_used': self.feature_names
            }
            
            # Add confidence interval (rough estimate)
            if return_confidence:
                # Use RMSE as a proxy for uncertainty
                rmse = self.performance_metrics.get('test_rmse', prediction * 0.2)
                confidence_interval = {
                    'lower_bound': float(max(0, prediction - 1.96 * rmse)),
                    'upper_bound': float(prediction + 1.96 * rmse),
                    'confidence_level': 0.95
                }
                result['confidence_interval'] = confidence_interval
            
            logger.info(f"Prediction made: {prediction:.0f} views")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze and return feature importance.
        
        Returns:
            Dictionary with feature importance analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before analyzing feature importance")
        
        analysis = {
            'model_type': self.model_type,
            'features': {}
        }
        
        if 'feature_importance' in self.performance_metrics:
            importance_dict = self.performance_metrics['feature_importance']
            
            # Sort by importance
            sorted_features = sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            total_importance = sum(importance_dict.values())
            
            for feature, importance in sorted_features:
                analysis['features'][feature] = {
                    'importance': float(importance),
                    'relative_importance': float(importance / total_importance * 100) if total_importance > 0 else 0
                }
        
        return analysis
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'performance_metrics': self.performance_metrics
            }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_model(self, videos_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance on new data.
        
        Args:
            videos_df: DataFrame with video data for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            X, y = self.prepare_features(videos_df)
            
            # Scale features if needed
            if self.model_type in ['linear', 'ridge', 'lasso']:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            evaluation_metrics = {
                'r2_score': float(r2_score(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'mae': float(mean_absolute_error(y, y_pred)),
                'mape': float(np.mean(np.abs((y - y_pred) / y)) * 100),
                'sample_size': len(y)
            }
            
            logger.info(f"Model evaluation completed. R² score: {evaluation_metrics['r2_score']:.3f}")
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def generate_prediction_report(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive prediction report.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary with prediction analysis
        """
        if not predictions:
            return {'error': 'No predictions provided'}
        
        predicted_views = [p['predicted_views'] for p in predictions]
        
        report = {
            'summary': {
                'total_predictions': len(predictions),
                'average_predicted_views': float(np.mean(predicted_views)),
                'median_predicted_views': float(np.median(predicted_views)),
                'min_predicted_views': float(np.min(predicted_views)),
                'max_predicted_views': float(np.max(predicted_views)),
                'std_predicted_views': float(np.std(predicted_views))
            },
            'model_info': {
                'model_type': self.model_type,
                'features_used': self.feature_names,
                'performance_metrics': self.performance_metrics
            },
            'predictions': predictions
        }
        
        return report
