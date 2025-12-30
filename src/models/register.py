"""
Model Registration Pipeline
Handles MLflow model registration with lifecycle management and governance
"""

import numpy as np
import pandas as pd
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple  # Fixed: Added List and Tuple
from datetime import datetime
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException
import joblib

# Fixed: Configure standard logging instead of structlog
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRegistrationError(Exception):
    """Custom exception for model registration failures"""
    pass


class ModelRegistry:
    """Production-grade model registry with lifecycle management"""
    
    def __init__(
        self,
        model_name: str,
        experiment_name: str = "churn_prediction_experiment",
        mlflow_tracking_uri: Optional[str] = None
    ):
        self.model_name = model_name
        self.experiment_name = experiment_name
        
        # Configure MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        self.client = MlflowClient()
        
        # Ensure experiment exists
        self._ensure_experiment_exists()
    
    def _ensure_experiment_exists(self) -> None:
        """Ensure MLflow experiment exists"""
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
            
            # Set as active experiment
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to create or get experiment: {e}")
            raise
    
    def register_model(
        self,
        model_path: str,
        model_config_path: str,
        evaluation_results_path: str,
        feature_transformer_path: str,
        stage: str = "Staging",
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register model with MLflow model registry"""
        
        logger.info(f"Starting model registration for {self.model_name}")
        
        try:
            # Load model and metadata
            model = self._load_model(model_path)
            model_config = self._load_json(model_config_path)
            evaluation_results = self._load_json(evaluation_results_path)
            
            # Start MLflow run
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                
                # Log model parameters
                mlflow.log_params(model_config.get('best_params', {}))
                
                # Log evaluation metrics
                metrics = evaluation_results.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None:
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log business metrics
                business_impact = evaluation_results.get('business_impact', {})
                for metric_name, metric_value in business_impact.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(f"business_{metric_name}", metric_value)
                
                # Log model artifacts
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(feature_transformer_path)
                mlflow.log_artifact(evaluation_results_path)
                
                # Create model signature (mocking input shape for XGBoost/sklearn)
                sample_input = np.random.randn(1, 38)  # Match your 38 features
                sample_output = model.predict(sample_input)
                signature = infer_signature(sample_input, sample_output)
                
                # Log model with signature
                if "xgboost" in self.model_name.lower() or "xgb" in str(type(model)).lower():
                    mlflow.xgboost.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        registered_model_name=self.model_name
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        signature=signature,
                        registered_model_name=self.model_name
                    )
                
                # Get latest model version (initially in stage "None")
                model_version = self.client.get_latest_versions(
                    self.model_name,
                    stages=["None"]
                )[0]
                
                # Default system tags
                default_tags = {
                    "owner": "ml-team",
                    "cost_center": "engineering",
                    "model_type": model_config.get('model_type', 'unknown'),
                    "training_timestamp": datetime.now().isoformat(),
                    "run_id": run_id
                }
                
                if tags:
                    default_tags.update(tags)
                
                for tag_key, tag_value in default_tags.items():
                    self.client.set_model_version_tag(
                        name=self.model_name,
                        version=model_version.version,
                        key=tag_key,
                        value=str(tag_value)
                    )
                
                if description:
                    self.client.update_model_version(
                        name=self.model_name,
                        version=model_version.version,
                        description=description
                    )
                
                logger.info(f"Model {self.model_name} version {model_version.version} registered.")
                
                return {
                    "model_name": self.model_name,
                    "version": model_version.version,
                    "run_id": run_id,
                    "stage": "None",
                    "metrics": metrics,
                    "tags": default_tags
                }
                
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise ModelRegistrationError(f"Registration failed: {str(e)}")
    
    def transition_model_stage(
        self,
        version: str,
        stage: str,
        archive_existing_versions: bool = True
    ) -> None:
        """Transition model to new stage with validation"""
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ModelRegistrationError(f"Invalid stage: {stage}")
        
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Model {self.model_name} v{version} transitioned to {stage}")
        except Exception as e:
            logger.error(f"Stage transition failed: {e}")
            raise ModelRegistrationError(f"Stage transition failed: {str(e)}")

    def validate_model_for_production(self, version: str) -> bool:
        """Validate if model meets production criteria based on tags/metrics"""
        try:
            model_version = self.client.get_model_version(name=self.model_name, version=version)
            # Thresholds
            min_accuracy = 0.80  # Your current is 0.8435
            
            # Search tags for metrics logged during registration
            accuracy = float(model_version.tags.get("accuracy", 0))
            if accuracy >= min_accuracy:
                return True
            logger.warning(f"Model v{version} failed validation: accuracy {accuracy} < {min_accuracy}")
            return False
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def get_model_versions(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all model versions, optionally filtered by stage"""
        try:
            if stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            return [{
                'version': v.version,
                'stage': v.current_stage,
                'run_id': v.run_id,
                'status': v.status
            } for v in versions]
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []

    def rollback_model(self, target_version: str) -> None:
        """Rollback to previous model version"""
        try:
            production_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
            for v in production_versions:
                self.client.transition_model_version_stage(self.model_name, v.version, "Archived")
            self.client.transition_model_version_stage(self.model_name, target_version, "Production")
            logger.info(f"Rollback to v{target_version} successful.")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise ModelRegistrationError(str(e))

    def _load_model(self, model_path: str) -> Any:
        return joblib.load(model_path)
    
    def _load_json(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r') as f:
            return json.load(f)


def main():
    """Main entry point for DVC pipeline"""
    model_path = "models/churn_model.pkl"
    model_config_path = "models/model_config.json"
    evaluation_results_path = "models/evaluation/evaluation_results.json"
    feature_transformer_path = "models/feature_transformer.joblib"
    
    try:
        registry = ModelRegistry(model_name="churn_prediction")
        
        # Register model to MLflow
        registration_result = registry.register_model(
            model_path=model_path,
            model_config_path=model_config_path,
            evaluation_results_path=evaluation_results_path,
            feature_transformer_path=feature_transformer_path,
            description="Production Churn Prediction Model"
        )
        
        # Promotion logic
        if registry.validate_model_for_production(registration_result['version']):
            registry.transition_model_stage(registration_result['version'], "Production")
        
        # Save registration metadata for DVC tracking
        os.makedirs("models", exist_ok=True)
        with open("models/registration_metrics.json", 'w') as f:
            json.dump(registration_result, f, indent=2)
            
        print(f"✅ Registration successful. Version: {registration_result['version']}")
        return 0
        
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())