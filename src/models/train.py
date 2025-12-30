"""
Model Training Pipeline
Production-grade model training with MLflow tracking, budget enforcement,
and dynamic hyperparameter search from params.yaml.
"""

import numpy as np
import logging
import json
import sys
import time
import os
import yaml
import pickle
import inspect
from typing import Dict, Any, Tuple, List
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------- CUSTOM ERROR --------------------
class ModelTrainingError(Exception):
    pass


# -------------------- MODEL FACTORY --------------------
class ModelFactory:
    """Factory for safe model creation"""

    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any]) -> Any:
        params = params.copy()  # Avoid mutation of the grid dictionary

        if model_type == "xgboost":
            model_class = xgb.XGBClassifier
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "auc")
            params.setdefault("random_state", 42)
            # XGBoost 1.6+ fix: use_label_encoder is deprecated
            if "use_label_encoder" in params:
                del params["use_label_encoder"]

        elif model_type == "random_forest":
            model_class = RandomForestClassifier
            params.setdefault("random_state", 42)

        elif model_type == "logistic_regression":
            model_class = LogisticRegression
            params.setdefault("random_state", 42)
            params.setdefault("max_iter", 1000)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Filter parameters to match the constructor signature
        sig = inspect.signature(model_class.__init__)
        clean_params = {k: v for k, v in params.items() if k in sig.parameters}

        return model_class(**clean_params)

    @staticmethod
    def get_hyperparameter_space(config: Dict[str, Any]) -> Dict[str, List[Any]]:
        excluded = {"model_type", "budget_limit_usd"}
        return {
            k: v if isinstance(v, list) else [v]
            for k, v in config.items()
            if k not in excluded
        }


# -------------------- TRAINER --------------------
class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.model_type = config.get("model_type", "xgboost").lower()
        self.budget_limit_usd = float(config.get("budget_limit_usd", 50.0))
        self.config = config
        self.start_time = None
        self.compute_cost_per_hour = 2.0  # Mock instance cost

    def _enforce_budget(self):
        elapsed_hours = (time.time() - self.start_time) / 3600
        cost = elapsed_hours * self.compute_cost_per_hour
        if cost > self.budget_limit_usd:
            raise ModelTrainingError(
                f"Budget exceeded: ${cost:.2f} > ${self.budget_limit_usd:.2f}"
            )

    def _search(self, X_train, y_train) -> Tuple[Any, Dict[str, Any], float]:
        param_space = ModelFactory.get_hyperparameter_space(self.config)
        grid = list(ParameterGrid(param_space))

        logger.info(f"Hyperparameter combinations to test: {len(grid)}")

        best_score = -np.inf
        best_params = None

        # Stratified K-Fold for reliable evaluation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for idx, params in enumerate(grid, 1):
            self._enforce_budget()
            model = ModelFactory.create_model(self.model_type, params)

            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring="accuracy"
            )

            mean_score = float(scores.mean())

            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()

            logger.info(
                f"[{idx}/{len(grid)}] CV accuracy={mean_score:.4f} (best={best_score:.4f})"
            )

        logger.info(f"Best hyperparameters: {best_params}")
        final_model = ModelFactory.create_model(self.model_type, best_params)
        final_model.fit(X_train, y_train)

        return final_model, best_params, best_score

    def train(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        self.start_time = time.time()

        model, best_params, best_cv_score = self._search(X_train, y_train)

        y_pred = model.predict(X_val)
        
        # Casting metrics to float for JSON/MLflow compatibility
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, average="weighted")),
            "recall": float(recall_score(y_val, y_pred, average="weighted")),
            "f1": float(f1_score(y_val, y_pred, average="weighted")),
        }

        elapsed_hours = (time.time() - self.start_time) / 3600

        return {
            "model": model,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "metrics": metrics,
            "training_cost_usd": elapsed_hours * self.compute_cost_per_hour,
            "training_time_hours": elapsed_hours,
            "model_type": self.model_type,
        }


# -------------------- MAIN --------------------
def main():
    try:
        os.makedirs("models", exist_ok=True)

        # 1. Load Data
        X_train = np.load("data/processed/X_train.npy")
        y_train = np.load("data/processed/y_train.npy")
        X_val = np.load("data/processed/X_test.npy")
        y_val = np.load("data/processed/y_test.npy")

        # 2. Load Config
        with open("params.yaml") as f:
            params = yaml.safe_load(f)["training"]

        # 3. MLflow Initialization
        mlflow.set_experiment("churn_prediction_experiment")

        if mlflow.active_run():
            mlflow.end_run()

        

        with mlflow.start_run() as run:
            trainer = ModelTrainer(params)
            results = trainer.train(X_train, y_train, X_val, y_val)

            # Log to MLflow
            mlflow.log_params(results["best_params"])
            mlflow.log_metrics(results["metrics"])
            mlflow.log_metric("best_cv_accuracy", results["best_cv_score"])
            mlflow.log_metric("training_cost_usd", results["training_cost_usd"])

            # 4. Registered Model Logic
            if results["model_type"] == "xgboost":
                mlflow.xgboost.log_model(
                    results["model"],
                    artifact_path="model",
                    registered_model_name="churn_prediction"
                )
            else:
                mlflow.sklearn.log_model(
                    results["model"],
                    artifact_path="model",
                    registered_model_name="churn_prediction"
                )

            # 5. Local Artifacts
            with open("models/churn_model.pkl", "wb") as f:
                pickle.dump(results["model"], f)

            with open("models/model_config.json", "w") as f:
                json.dump({
                    "model_type": results["model_type"],
                    "best_params": results["best_params"],
                    "run_id": run.info.run_id,
                    "timestamp": datetime.utcnow().isoformat()
                }, f, indent=2)

            with open("models/training_metrics.json", "w") as f:
                json.dump(results["metrics"], f, indent=2)

            logger.info("✅ Training and MLflow registration completed successfully")
            return 0

    except Exception as e:
        logger.exception("❌ Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())