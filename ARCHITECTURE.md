# Production MLOps Architecture

## ASCII Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  [Mobile App] → [Web App] → [B2B API] → [Internal Services]      │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                     API GATEWAY LAYER                              │
│  [AWS API Gateway] → [WAF] → [Rate Limiter] → [Auth Service]      │
│  Logging: Every request gets trace-id, user-id, model-version      │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                    INFERENCE SERVICE (FastAPI)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ Model Runner │→ │ Feature Cache│→│  Prometheus  │            │
│  │ (5 workers)  │  │  (Redis)     │  │  Metrics     │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│  Health: /health/live, /health/ready, /health/dependencies        │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                    FEATURE STORE LAYER                             │
│  [Redis Cache] ←→ [PostgreSQL] ←→ [Feature Pipeline]             │
│  Features versioned like models; TTL enforcement                   │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                    MODEL REGISTRY (MLflow)                         │
│  S3 Backend → RDS Metadata → Lifecycle Policies → Audit Log       │
│  Tags: owner, cost_center, p95_latency, error_rate                │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│                    DATA PIPELINE (DVC + Airflow)                   │
│  [S3 Raw Data] → [Great Expectations] → [Featurization] → [Train]│
│  DAG has retry logic, circuit breakers, data quality gates         │
└─────────────────────────────────────────────────────────────────┘
```

## Component Explanations

### Circuit Breaker Pattern
When the feature store is down, the system serves stale features from Redis cache with a metric alert. This ensures service continuity while maintaining observability.

### Graceful Degradation
If the model fails, the system falls back to a rule-based system that uses business logic for predictions. This prevents complete service failure.

### Request Hedging
After 30ms, the system sends a duplicate request to the backup region. This ensures low latency even when primary region is slow.

### Cost-Aware Routing
10% of traffic is routed to a cheaper but slightly worse model variant to optimize costs while maintaining performance.

## Key Design Decisions

1. **FastAPI over Flask**: FastAPI's async capabilities provide 5x throughput for high-concurrency scenarios
2. **Redis + PostgreSQL**: Redis for hot feature caching, PostgreSQL for persistent feature storage
3. **MLflow Model Registry**: Centralized model versioning with lifecycle management
4. **DVC + Airflow**: Reproducible data pipelines with automated orchestration
5. **Prometheus + Grafana**: Comprehensive monitoring and alerting
6. **Blue-Green Deployment**: Zero-downtime deployments with instant rollback capability