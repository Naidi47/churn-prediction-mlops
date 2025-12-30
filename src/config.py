"""
Production MLOps Configuration
Handles environment-specific settings with security and performance in mind
"""

import os
from functools import lru_cache
from typing import Optional, List
import json

# Correct Pydantic V2 imports
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and environment-specific defaults"""
    
    # Application
    APP_NAME: str = "churn-prediction-api"
    APP_ENV: str = Field(default="development", validation_alias="APP_ENV")
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 5
    API_RELOAD: bool = False
    
    # Model Configuration
    MODEL_NAME: str = "churn_prediction"
    MODEL_STAGE: str = "Production"
    MODEL_CACHE_TTL: int = 300  # 5 minutes
    FALLBACK_MODEL_PATH: str = "src/models/fallback_rules.json"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field(..., validation_alias="MLFLOW_TRACKING_URI")
    MLFLOW_REGISTRY_URI: str = Field(..., validation_alias="MLFLOW_REGISTRY_URI")
    MLFLOW_EXPERIMENT_NAME: str = "churn_prediction_experiment"
    
    # Feature Store - PostgreSQL
    POSTGRES_HOST: str = Field(..., validation_alias="POSTGRES_HOST")
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = Field(..., validation_alias="POSTGRES_DB")
    POSTGRES_USER: str = Field(..., validation_alias="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., validation_alias="POSTGRES_PASSWORD")
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 30
    
    # Redis Cache
    REDIS_HOST: str = Field(..., validation_alias="REDIS_HOST")
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = Field(None, validation_alias="REDIS_PASSWORD")
    REDIS_POOL_SIZE: int = 50
    REDIS_SSL: bool = True
    
    # Security
    JWT_SECRET_KEY: str = Field(..., validation_alias="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60
    API_KEY_HEADER: str = "X-API-Key"
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Monitoring
    PROMETHEUS_METRICS_PORT: int = 9090
    PROMETHEUS_METRICS_PATH: str = "/metrics"
    HEALTH_CHECK_TIMEOUT: int = 5
    
    # Performance
    INFERENCE_TIMEOUT_SECONDS: float = 0.05  # 50ms target
    FEATURE_STORE_TIMEOUT_SECONDS: float = 0.03  # 30ms target
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 60
    
    # Cost Tracking
    COST_PER_INFERENCE_USD: float = 0.0001
    ENABLE_COST_TRACKING: bool = True
    
    # A/B Testing
    A_B_TEST_ENABLED: bool = False
    A_B_TEST_VARIANT_TRAFFIC_PERCENT: float = 10.0
    
    # Circuit Breaker & Fallback
    ENABLE_CIRCUIT_BREAKER: bool = True
    ENABLE_GRACEFUL_DEGRADATION: bool = True
    STALE_FEATURE_TTL_SECONDS: int = 3600  # 1 hour
    
    # Data Pipeline
    DVC_REMOTE_STORAGE: str = Field(..., validation_alias="DVC_REMOTE_STORAGE")
    DATA_VALIDATION_THRESHOLD: float = 0.05  # 5% failure threshold
    
    # Compliance & Audit
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_RETENTION_DAYS: int = 2555  # 7 years for compliance
    GDPR_COMPLIANT: bool = True
    
    # Region Failover
    PRIMARY_REGION: str = "us-east-1"
    FAILOVER_REGION: str = "us-west-2"
    ENABLE_CROSS_REGION_REPLICATION: bool = True

    # Pydantic V2 field_validator syntax
    @field_validator("APP_ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in {"development", "staging", "production"}:
            raise ValueError("APP_ENV must be one of: development, staging, production")
        return v
    
    @field_validator("POSTGRES_PORT")
    @classmethod
    def validate_postgres_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("POSTGRES_PORT must be between 1 and 65535")
        return v
    
    @property
    def database_url(self) -> str:
        """PostgreSQL connection URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        """Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"
    
    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"
    
    # Updated Config for Pydantic V2
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Environment-specific configurations
ENV_CONFIGS = {
    "development": {
        "DEBUG": True,
        "API_RELOAD": True,
        "LOG_LEVEL": "DEBUG",
        "RATE_LIMIT_PER_MINUTE": 1000,
    },
    "staging": {
        "DEBUG": False,
        "API_RELOAD": False,
        "LOG_LEVEL": "INFO",
        "RATE_LIMIT_PER_MINUTE": 500,
        "A_B_TEST_ENABLED": True,
    },
    "production": {
        "DEBUG": False,
        "API_RELOAD": False,
        "LOG_LEVEL": "WARNING",
        "RATE_LIMIT_PER_MINUTE": 100,
        "A_B_TEST_ENABLED": True,
        "ENABLE_COST_TRACKING": True,
    }
}


def get_env_config() -> dict:
    """Get environment-specific configuration"""
    settings = get_settings()
    return ENV_CONFIGS.get(settings.APP_ENV, ENV_CONFIGS["development"])