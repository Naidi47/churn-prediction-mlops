"""
Model Factory
Creates model instances based on configuration
"""
import logging
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class to create model instances"""
    
    @staticmethod
    def create_model(config: Dict[str, Any]):
        """Create a model based on configuration dictionary"""
        # Lowercase the model type to be case-insensitive
        model_type = config.get("model_type", "xgboost").lower()
        
        # Copy config to avoid modifying the original dict
        params = config.copy()
        
        # Remove non-hyperparameter keys
        if "model_type" in params:
            del params["model_type"]
        if "budget_limit_usd" in params:
            del params["budget_limit_usd"]
            
        logger.info(f"Creating model of type: {model_type}")
        
        if model_type == "xgboost":
            return xgb.XGBClassifier(
                **params,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            )
            
        elif model_type == "random_forest":
            return RandomForestClassifier(
                **params,
                random_state=42
            )
            
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                **params,
                random_state=42
            )
            
        elif model_type == "logistic_regression":
            return LogisticRegression(
                **params,
                random_state=42
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")