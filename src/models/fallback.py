"""
Fallback Rule-Based Model
Provides predictions when ML model fails, ensuring service continuity
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

structlog = logging.getLogger(__name__)


class RuleBasedModel:
    """
    Rule-based fallback model for churn prediction.
    Uses business logic when ML model is unavailable.
    """
    
    def __init__(self, rules_path: str = "src/models/fallback_rules.json"):
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load business rules for churn prediction"""
        default_rules = {
            "tenure_rules": {
                "new_customer_threshold": 90,  # days
                "new_customer_churn_probability": 0.7,
                "loyal_customer_threshold": 730,  # days (2 years)
                "loyal_customer_churn_probability": 0.1
            },
            "payment_rules": {
                "high_churn_methods": ["Check", "Bank Transfer"],
                "high_churn_probability": 0.6,
                "low_churn_methods": ["Credit Card", "PayPal"],
                "low_churn_probability": 0.3
            },
            "contract_rules": {
                "month_to_month_probability": 0.6,
                "one_year_probability": 0.3,
                "two_year_probability": 0.1
            },
            "service_rules": {
                "multiple_services_bonus": -0.1,  # Reduce churn probability
                "no_services_penalty": 0.2
            },
            "billing_rules": {
                "high_charges_threshold": 100,  # USD
                "high_charges_penalty": 0.15,
                "unpaid_bills_penalty": 0.3
            },
            "default_probability": 0.4
        }
        
        try:
            if self.rules_path.exists():
                with open(self.rules_path, 'r') as f:
                    rules = json.load(f)
                structlog.info("Loaded fallback rules from file")
                return rules
            else:
                # Create default rules file
                self.rules_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.rules_path, 'w') as f:
                    json.dump(default_rules, f, indent=2)
                structlog.info("Created default fallback rules")
                return default_rules
        
        except Exception as e:
            structlog.warning("Failed to load fallback rules, using defaults", error=str(e))
            return default_rules
    
    def predict(self, features: np.ndarray) -> int:
        """
        Make prediction using business rules
        
        Expected feature indices (based on typical churn model):
        0: monthly_charges
        1: tenure_days
        2: number_of_services
        3: number_of_dependents
        4: total_charges
        Additional features may be present but are ignored for rule-based prediction
        """
        
        try:
            # Extract key features with fallbacks
            monthly_charges = features[0][0] if features.shape[1] > 0 else 50.0
            tenure_days = features[0][1] if features.shape[1] > 1 else 365.0
            number_of_services = features[0][2] if features.shape[1] > 2 else 2.0
            total_charges = features[0][4] if features.shape[1] > 4 else monthly_charges * 12
            
            # Start with base probability
            churn_probability = self.rules["default_probability"]
            
            # Apply tenure rules
            if tenure_days < self.rules["tenure_rules"]["new_customer_threshold"]:
                churn_probability = max(
                    churn_probability,
                    self.rules["tenure_rules"]["new_customer_churn_probability"]
                )
            elif tenure_days > self.rules["tenure_rules"]["loyal_customer_threshold"]:
                churn_probability = min(
                    churn_probability,
                    self.rules["tenure_rules"]["loyal_customer_churn_probability"]
                )
            
            # Apply payment method rules (mock - in production, this would be a feature)
            # For now, assume medium risk
            
            # Apply contract rules (mock - assume month-to-month)
            contract_multiplier = 1.0
            # Simulate contract type based on tenure
            if tenure_days < 365:
                contract_multiplier = self.rules["contract_rules"]["month_to_month_probability"]
            elif tenure_days < 730:
                contract_multiplier = self.rules["contract_rules"]["one_year_probability"]
            else:
                contract_multiplier = self.rules["contract_rules"]["two_year_probability"]
            
            churn_probability *= contract_multiplier
            
            # Apply service rules
            if number_of_services >= 3:
                churn_probability += self.rules["service_rules"]["multiple_services_bonus"]
            elif number_of_services == 0:
                churn_probability += self.rules["service_rules"]["no_services_penalty"]
            
            # Apply billing rules
            if monthly_charges > self.rules["billing_rules"]["high_charges_threshold"]:
                churn_probability += self.rules["billing_rules"]["high_charges_penalty"]
            
            # Check for unusual billing patterns (mock)
            avg_monthly = total_charges / max(tenure_days / 30, 1)
            if avg_monthly > monthly_charges * 1.5:
                churn_probability += self.rules["billing_rules"]["unpaid_bills_penalty"]
            
            # Ensure probability is between 0 and 1
            churn_probability = max(0.0, min(1.0, churn_probability))
            
            # Make binary prediction
            prediction = 1 if churn_probability > 0.5 else 0
            
            structlog.info(
                "Fallback prediction made",
                churn_probability=churn_probability,
                prediction=prediction,
                tenure_days=tenure_days,
                monthly_charges=monthly_charges,
                number_of_services=number_of_services
            )
            
            return prediction
            
        except Exception as e:
            structlog.error("Fallback prediction failed", error=str(e))
            # Return random but deterministic prediction
            return int(features[0][0] % 2) if features.shape[1] > 0 else 0
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (mock implementation)"""
        predictions = []
        
        for feature_row in features:
            # Use same logic as predict but return probability
            prediction = self.predict(feature_row.reshape(1, -1))
            
            # Mock probability based on prediction
            if prediction == 1:
                probability = 0.6 + np.random.random() * 0.3  # 0.6 - 0.9
            else:
                probability = 0.1 + np.random.random() * 0.3  # 0.1 - 0.4
            
            predictions.append([1 - probability, probability])
        
        return np.array(predictions)
    
    def explain_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain the reasoning behind a prediction"""
        
        try:
            monthly_charges = features[0][0] if features.shape[1] > 0 else 50.0
            tenure_days = features[0][1] if features.shape[1] > 1 else 365.0
            number_of_services = features[0][2] if features.shape[1] > 2 else 2.0
            
            reasons = []
            risk_score = 0.0
            
            # Check tenure
            if tenure_days < 90:
                reasons.append("New customer (less than 90 days)")
                risk_score += 0.3
            elif tenure_days > 730:
                reasons.append("Loyal customer (more than 2 years)")
                risk_score -= 0.2
            
            # Check services
            if number_of_services >= 3:
                reasons.append(f"Multiple services ({number_of_services})")
                risk_score -= 0.1
            elif number_of_services == 0:
                reasons.append("No active services")
                risk_score += 0.2
            
            # Check charges
            if monthly_charges > 100:
                reasons.append(f"High monthly charges (${monthly_charges:.2f})")
                risk_score += 0.15
            
            # Determine risk level
            if risk_score > 0.3:
                risk_level = "high"
            elif risk_score > 0.1:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "prediction_method": "rule_based_fallback",
                "risk_level": risk_level,
                "risk_score": risk_score,
                "reasons": reasons,
                "key_factors": {
                    "tenure_days": tenure_days,
                    "monthly_charges": monthly_charges,
                    "number_of_services": number_of_services
                }
            }
            
        except Exception as e:
            structlog.error("Failed to explain prediction", error=str(e))
            return {
                "prediction_method": "rule_based_fallback",
                "risk_level": "unknown",
                "risk_score": 0.0,
                "reasons": ["Analysis failed"],
                "error": str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    # Create fallback model
    fallback = RuleBasedModel()
    
    # Test with different customer profiles
    test_cases = [
        # New customer, high charges, few services
        np.array([[120.0, 30, 1, 0, 120.0]] * 4 + [[120.0, 30, 1, 0, 120.0]]),
        
        # Loyal customer, reasonable charges, many services
        np.array([[60.0, 1000, 4, 2, 2400.0]] * 4 + [[60.0, 1000, 4, 2, 2400.0]]),
        
        # Medium tenure, medium charges
        np.array([[80.0, 500, 2, 1, 1200.0]] * 4 + [[80.0, 500, 2, 1, 1200.0]])
    ]
    
    for i, features in enumerate(test_cases):
        prediction = fallback.predict(features.reshape(1, -1))
        explanation = fallback.explain_prediction(features.reshape(1, -1))
        
        print(f"\nTest Case {i + 1}:")
        print(f"Prediction: {prediction}")
        print(f"Risk Level: {explanation['risk_level']}")
        print(f"Risk Score: {explanation['risk_score']:.2f}")
        print(f"Reasons: {', '.join(explanation['reasons'])}")
        print(f"Key Factors: {explanation['key_factors']}")