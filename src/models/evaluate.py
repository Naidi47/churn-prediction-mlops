"""
Model Evaluation Pipeline
Comprehensive model evaluation with fairness metrics, calibration analysis, and business impact assessment
"""

import numpy as np
import pandas as pd
import logging
import json
import sys
import os
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss,
    roc_curve
)
# Fixed: Imported from correct module only
from sklearn.calibration import calibration_curve

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluationError(Exception):
    """Custom exception for model evaluation failures"""
    pass


class ModelEvaluator:
    """Production-grade model evaluator with comprehensive metrics"""
    
    def __init__(
        self,
        model_name: str = "churn_prediction",
        fairness_metrics: bool = True,
        calibration_analysis: bool = True,
        slice_analysis: bool = True,
        business_metrics: bool = True
    ):
        self.model_name = model_name
        self.fairness_metrics = fairness_metrics
        self.calibration_analysis = calibration_analysis
        self.slice_analysis = slice_analysis
        self.business_metrics = business_metrics
        
        # Evaluation results
        self.results = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive model performance metrics"""
        # Using weighted to handle potential class imbalances in slices
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba))
                metrics['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {e}")
                metrics['auc'] = 0.0
                metrics['brier_score'] = 0.0
        
        return metrics
    
    def analyze_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Analyze model calibration"""
        if y_pred_proba is None:
            return {}
        
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=n_bins
            )
            
            # Calculate Expected Calibration Error (ECE)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Calculate Maximum Calibration Error (MCE)
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'expected_calibration_error': float(ece),
                'maximum_calibration_error': float(mce),
                'calibration_curve': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            return {}
    
    def analyze_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze model fairness across sensitive attributes"""
        fairness_results = {}
        
        for attr_name, attr_values in sensitive_attributes.items():
            unique_values = np.unique(attr_values)
            group_metrics = {}
            
            for value in unique_values:
                mask = attr_values == value
                if mask.sum() > 0:
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    
                    group_metrics[str(value)] = {
                        'accuracy': float(accuracy_score(y_true_group, y_pred_group)),
                        'precision': float(precision_score(y_true_group, y_pred_group, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y_true_group, y_pred_group, average='weighted', zero_division=0)),
                        'f1': float(f1_score(y_true_group, y_pred_group, average='weighted', zero_division=0)),
                        'sample_size': int(len(y_true_group))
                    }
            
            fairness_results[attr_name] = {
                'group_metrics': group_metrics,
                'demographic_parity': float(self._calculate_demographic_parity(y_pred, attr_values)),
                'equalized_odds': float(self._calculate_equalized_odds(y_true, y_pred, attr_values))
            }
        
        return fairness_results
    
    def _calculate_demographic_parity(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        unique_values = np.unique(sensitive_attr)
        positive_rates = []
        for value in unique_values:
            mask = sensitive_attr == value
            if mask.sum() > 0:
                positive_rates.append(np.mean(y_pred[mask]))
        return float(np.std(positive_rates)) if positive_rates else 0.0
    
    def _calculate_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        unique_values = np.unique(sensitive_attr)
        tprs, fprs = [], []
        for value in unique_values:
            mask = sensitive_attr == value
            if mask.sum() > 0:
                y_t, y_p = y_true[mask], y_pred[mask]
                tn = np.sum((y_t == 0) & (y_p == 0))
                fp = np.sum((y_t == 0) & (y_p == 1))
                fn = np.sum((y_t == 1) & (y_p == 0))
                tp = np.sum((y_t == 1) & (y_p == 1))
                tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        return float((np.std(tprs) + np.std(fprs)) / 2) if tprs else 0.0
    
    def analyze_slices(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, slice_columns: List[str]) -> Dict[str, Any]:
        slice_results = {}
        for col in slice_columns:
            if col not in df.columns: continue
            unique_values = df[col].unique()
            col_results = {}
            for value in unique_values:
                mask = df[col] == value
                if mask.sum() > 10:
                    col_results[str(value)] = {
                        'metrics': self.calculate_metrics(y_true[mask], y_pred[mask]),
                        'sample_size': int(mask.sum()),
                        'churn_rate': float(np.mean(y_true[mask]))
                    }
            slice_results[col] = col_results
        return slice_results
    
    def calculate_business_impact(self, y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float = 100.0, cost_fn: float = 500.0) -> Dict[str, float]:
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        baseline_cost = np.sum(y_true) * cost_fn
        savings = baseline_cost - total_cost
        campaign_cost = (tp + fp) * cost_fp
        
        return {
            'total_cost': float(total_cost),
            'baseline_cost': float(baseline_cost),
            'savings': float(savings),
            'roi': float(savings / campaign_cost) if campaign_cost > 0 else 0.0,
            'campaigns_run': int(tp + fp),
            'churners_caught': int(tp),
            'missed_churners': int(fn)
        }
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray], output_dir: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if y_pred_proba is not None:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_pred_proba):.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calibration
            f_pos, m_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
            plt.figure(figsize=(8, 6))
            plt.plot(m_pred, f_pos, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:")
            plt.title('Calibration Plot')
            plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
            plt.close()

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, df_test: Optional[pd.DataFrame] = None, sensitive_attributes: Optional[Dict[str, np.ndarray]] = None, output_dir: str = "models/evaluation") -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.results = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'metrics': self.calculate_metrics(y_test, y_pred, y_pred_proba),
                'calibration': self.analyze_calibration(y_test, y_pred_proba) if self.calibration_analysis and y_pred_proba is not None else {},
                'fairness': self.analyze_fairness(y_test, y_pred, sensitive_attributes) if self.fairness_metrics and sensitive_attributes else {},
                'business_impact': self.calculate_business_impact(y_test, y_pred) if self.business_metrics else {},
                'sample_size': int(len(y_test)),
                'churn_rate': float(np.mean(y_test))
            }
            
            self.create_visualizations(y_test, y_pred, y_pred_proba, str(output_dir))
            
            with open(output_dir / "evaluation_results.json", 'w') as f:
                json.dump(self.results, f, indent=2)
            
            return self.results
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise ModelEvaluationError(f"Evaluation failed: {str(e)}")


def main():
    model_path = "models/churn_model.pkl"
    X_test_path = "data/processed/X_test.npy"
    y_test_path = "data/processed/y_test.npy"
    output_dir = "models/evaluation"
    
    if len(sys.argv) > 1: model_path = sys.argv[1]
    if len(sys.argv) > 2: X_test_path = sys.argv[2]
    
    try:
        model = joblib.load(model_path)
        X_test, y_test = np.load(X_test_path), np.load(y_test_path)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model, X_test, y_test, output_dir=output_dir)
        
        with open("models/evaluation_metrics.json", 'w') as f:
            json.dump({
                "metrics": results['metrics'],
                "business_impact": results['business_impact'],
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
            
        logger.info(f"✅ Evaluation complete. Accuracy: {results['metrics']['accuracy']:.4f}")
        return 0
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())