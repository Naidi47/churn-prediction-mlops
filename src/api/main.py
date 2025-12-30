"""
Production FastAPI Inference Service
Production-safe, MLflow-backed, observable inference API
Compatible with FastAPI + Pydantic v2
"""

import asyncio
import logging
import time
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from prometheus_client import REGISTRY, Histogram, Counter, Gauge, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("inference")

# -------------------------------------------------
# PROMETHEUS METRICS (FIXED FOR RELOAD)
# -------------------------------------------------
# Unregister existing collectors to allow for Uvicorn hot-reloads
for collector in list(REGISTRY._collector_to_names.keys()):
    REGISTRY.unregister(collector)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency",
    ["status"]
)
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total predictions",
    ["is_fallback"]
)
MODEL_VERSION = Gauge(
    "model_version_gauge",
    "Currently loaded model version"
)
CIRCUIT_BREAKER_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state"
)

security = HTTPBearer(auto_error=False)

# -------------------------------------------------
# Pydantic Models
# -------------------------------------------------
# CONSTANT: The model expects 38 features
EXPECTED_FEATURES = 38

class PredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "CUST_12345678",
            "feature_vector": [0.1] * EXPECTED_FEATURES
        }
    })

    customer_id: str = Field(..., pattern=r"^CUST_\d{8}$")
    feature_vector: Optional[List[float]] = None

    @field_validator("feature_vector")
    @classmethod
    def validate_vector(cls, v):
        if v is not None and len(v) != EXPECTED_FEATURES:
            raise ValueError(f"feature_vector must contain exactly {EXPECTED_FEATURES} values")
        return v


class PredictionResponse(BaseModel):
    # Fix: Pydantic V2 protected namespace warning
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: int
    probability: float
    model_version: str
    is_fallback: bool
    latency_ms: float
    feature_source: str


# -------------------------------------------------
# MODEL MANAGER (CLOUD READY FIX)
# -------------------------------------------------
class ModelManager:
    def __init__(self, model_name: str, tracking_uri: Optional[str] = "http://127.0.0.1:5000"):
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri)
        self.model = None
        self.version = "cloud-v1"
        self.lock = asyncio.Lock()

    async def load_model(self):
        async with self.lock:
            try:
                # OPTION 1: Try Loading from Local File (Best for Cloud)
                # We look for src/model_files/model because that is where we downloaded it
                local_path = Path("src/model_files/model")
                
                if local_path.exists():
                    logger.info(f"Loading model from local path: {local_path}")
                    self.model = await asyncio.to_thread(mlflow.pyfunc.load_model, str(local_path))
                    return

                # OPTION 2: Fallback to MLflow Server (Best for Local Dev)
                # This only runs if the folder is missing
                mlflow.set_tracking_uri(self.tracking_uri)
                versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
                if versions:
                    latest = versions[0]
                    logger.info(f"Loading model version {latest.version} from MLflow")
                    model_uri = f"models:/{self.model_name}/Production"
                    self.model = await asyncio.to_thread(mlflow.pyfunc.load_model, model_uri)
                    self.version = latest.version
                    try:
                        MODEL_VERSION.set(float(latest.version))
                    except ValueError:
                        MODEL_VERSION.set(0)
                        
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                # We don't raise an error here so the API can still start (circuit breaker will handle failures)

    def predict(self, features: np.ndarray) -> tuple:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # ---------------------------------------------------------
        # ROBUST PREDICTION LOGIC
        # ---------------------------------------------------------
        
        # 1. Force float64 (Fixes 'dtype mismatch' error)
        features = features.astype(np.float64)

        # 2. Get Raw Prediction
        result = self.model.predict(features)

        # 3. Handle result extraction safely
        if isinstance(result, np.ndarray):
            result = result.flatten() # Flatten [[0.8]] to [0.8]
            val = result[0]
        elif hasattr(result, "iloc"): # Pandas Series/DataFrame
            val = result.iloc[0]
        else: # Standard list
            val = result[0]

        # 4. Determine Class vs Probability
        # If val is between 0 and 1, treat as probability
        if 0.0 <= float(val) <= 1.0 and isinstance(val, (float, np.floating)):
            probability = float(val)
            prediction = 1 if probability > 0.5 else 0
        else:
            # It's already a class label (0 or 1)
            prediction = int(val)
            probability = 1.0 if prediction == 1 else 0.0

        return prediction, probability


# -------------------------------------------------
# APP LIFESPAN
# -------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_manager = ModelManager(
        model_name="churn_prediction",
        tracking_uri="http://127.0.0.1:5000"
    )
    await app.state.model_manager.load_model()
    yield

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
async def predict(
    request: Request, # Fix: Required for SlowAPI
    payload: PredictionRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    start = time.time()
    is_fallback = False
    debug_error_msg = None

    try:
        # Prepare feature vector (using EXPECTED_FEATURES=38)
        features = (
            np.array(payload.feature_vector)
            if payload.feature_vector
            else np.zeros(EXPECTED_FEATURES)
        ).reshape(1, -1)

        try:
            prediction, probability = app.state.model_manager.predict(features)
            CIRCUIT_BREAKER_STATE.set(0)
        except Exception as e:
            # Log the EXACT error so we know why fallback happened
            logger.error(f"Inference error: {e}") 
            prediction, probability = 0, 0.5
            is_fallback = True
            debug_error_msg = str(e)
            CIRCUIT_BREAKER_STATE.set(1)

        latency_ms = (time.time() - start) * 1000
        INFERENCE_LATENCY.labels(status="success").observe(latency_ms / 1000)
        PREDICTION_COUNTER.labels(is_fallback=str(is_fallback)).inc()

        # Logic: If fallback happened, show the Error Message in 'feature_source'
        source_info = f"ERROR: {debug_error_msg}" if is_fallback else ("provided" if payload.feature_vector else "default")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=app.state.model_manager.version,
            is_fallback=is_fallback,
            latency_ms=latency_ms,
            feature_source=source_info
        )

    except Exception as e:
        logger.exception("Final endpoint failure")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.get("/health/live")
async def health():
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


# -------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------
if __name__ == "__main__":
    # Ensure this script is named main.py
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )