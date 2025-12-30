"""
Feature Store Client
Production-grade feature store with Redis caching, PostgreSQL persistence, and circuit breaker pattern
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import asyncpg
import aioredis
from asyncpg import Pool
from enum import Enum
import backoff
from src.feature_store.circuit_breaker import CircuitBreaker

# Configure logging
structlog = logging.getLogger(__name__)


class FeatureStoreError(Exception):
    """Base exception for feature store errors"""
    pass


class FeatureStoreConnectionError(FeatureStoreError):
    """Connection-related errors"""
    pass


class FeatureNotFoundError(FeatureStoreError):
    """Feature not found in store"""
    pass


class CircuitBreakerOpenError(FeatureStoreError):
    """Circuit breaker is open"""
    pass


@dataclass
class Feature:
    """Feature data structure"""
    entity_id: str
    feature_name: str
    feature_value: float
    event_timestamp: datetime
    ttl_seconds: int
    version: str
    created_at: Optional[datetime] = None


@dataclass
class FeatureStoreMetrics:
    """Feature store performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    database_queries: int = 0
    circuit_breaker_trips: int = 0
    average_latency_ms: float = 0.0
    errors: int = 0


class FeatureStoreClient:
    """Production-grade feature store client with caching and circuit breaker"""
    
    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_timeout_seconds: int = 60,
        cache_ttl_seconds: int = 3600,
        hot_feature_threshold: int = 1000,
        enable_circuit_breaker: bool = True
    ):
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.cache_ttl_seconds = cache_ttl_seconds
        self.hot_feature_threshold = hot_feature_threshold
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Connection pools
        self.pg_pool: Optional[Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Circuit breaker
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_breaker_failure_threshold,
                timeout_seconds=circuit_breaker_timeout_seconds
            )
        else:
            self.circuit_breaker = None
        
        # Metrics
        self.metrics = FeatureStoreMetrics()
        self.start_time = datetime.now()
        
        # Feature cache for hot features (in-memory optimization)
        self.hot_feature_cache: Dict[str, Feature] = {}
    
    async def initialize(self) -> None:
        """Initialize connection pools"""
        try:
            # Initialize PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=10,
                max_size=20,
                command_timeout=5.0  # 5 second timeout
            )
            
            # Initialize Redis client
            self.redis_client = aioredis.from_url(
                self.redis_url,
                max_connections=50,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            # Test connections
            await self._test_connections()
            
            structlog.info("Feature store client initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize feature store client: {e}")
            raise FeatureStoreConnectionError(f"Initialization failed: {str(e)}")
    
    async def _test_connections(self) -> None:
        """Test database and cache connections"""
        # Test PostgreSQL
        async with self.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Test Redis
        await self.redis_client.ping()
    
    async def close(self) -> None:
        """Close connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        structlog.info("Feature store client closed")
    
    def _update_metrics(self, cache_hit: bool, latency_ms: float, error: bool = False) -> None:
        """Update performance metrics"""
        self.metrics.total_requests += 1
        
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
            self.metrics.database_queries += 1
        
        if error:
            self.metrics.errors += 1
        
        # Update average latency
        total_requests = self.metrics.total_requests
        self.metrics.average_latency_ms = (
            (self.metrics.average_latency_ms * (total_requests - 1) + latency_ms) / total_requests
        )
    
    @backoff.on_exception(
        backoff.expo,
        (asyncpg.PostgresError, aioredis.RedisError),
        max_tries=3,
        max_time=5.0
    )
    async def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        version: str = "latest"
    ) -> Optional[Feature]:
        """Get feature value with caching and circuit breaker"""
        
        start_time = time.time()
        cache_hit = False
        error = False
        
        try:
            # Check circuit breaker
            if self.circuit_breaker and self.circuit_breaker.is_open():
                raise CircuitBreakerOpenError("Circuit breaker is open")
            
            # Create cache key
            cache_key = f"feature:{entity_id}:{feature_name}:{version}"
            
            # Try hot feature cache first (in-memory)
            if cache_key in self.hot_feature_cache:
                feature = self.hot_feature_cache[cache_key]
                if feature.event_timestamp + timedelta(seconds=feature.ttl_seconds) > datetime.now():
                    cache_hit = True
                    return feature
                else:
                    # Expired, remove from hot cache
                    del self.hot_feature_cache[cache_key]
            
            # Try Redis cache
            if self.redis_client:
                try:
                    cached_data = await self.redis_client.get(cache_key)
                    if cached_data:
                        feature_data = json.loads(cached_data)
                        feature = Feature(**feature_data)
                        
                        # Check TTL
                        if feature.event_timestamp + timedelta(seconds=feature.ttl_seconds) > datetime.now():
                            cache_hit = True
                            
                            # Promote to hot cache if frequently accessed
                            if await self._is_hot_feature(entity_id, feature_name):
                                self.hot_feature_cache[cache_key] = feature
                            
                            return feature
                        else:
                            # Expired, remove from Redis
                            await self.redis_client.delete(cache_key)
                
                except Exception as e:
                    structlog.warning("Redis cache error", error=str(e))
            
            # Fetch from PostgreSQL
            feature = await self._get_feature_from_db(entity_id, feature_name, version)
            
            if feature:
                # Cache in Redis
                if self.redis_client:
                    try:
                        feature_data = json.dumps(asdict(feature), default=str)
                        await self.redis_client.setex(
                            cache_key,
                            self.cache_ttl_seconds,
                            feature_data
                        )
                    except Exception as e:
                        structlog.warning("Failed to cache feature in Redis", error=str(e))
                
                # Check if this should be in hot cache
                if await self._is_hot_feature(entity_id, feature_name):
                    self.hot_feature_cache[cache_key] = feature
            
            return feature
            
        except Exception as e:
            error = True
            
            # Trip circuit breaker on failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
                self.metrics.circuit_breaker_trips += 1
            
            # If circuit breaker is open, try to serve stale features
            if isinstance(e, CircuitBreakerOpenError):
                structlog.warning("Circuit breaker open, attempting to serve stale features")
                stale_feature = await self._get_stale_feature(entity_id, feature_name)
                if stale_feature:
                    return stale_feature
            
            structlog.error("Failed to get feature", entity_id=entity_id, feature_name=feature_name, error=str(e))
            raise
            
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(cache_hit, latency_ms, error)
    
    async def _get_feature_from_db(
        self,
        entity_id: str,
        feature_name: str,
        version: str
    ) -> Optional[Feature]:
        """Get feature from PostgreSQL database"""
        
        query = """
            SELECT entity_id, feature_name, feature_value, 
                   event_timestamp, ttl_seconds, version, created_at
            FROM feature_store 
            WHERE entity_id = $1 
            AND feature_name = $2 
            AND ($3 = 'latest' OR version = $3)
            ORDER BY event_timestamp DESC 
            LIMIT 1
        """
        
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(query, entity_id, feature_name, version)
            
            if row:
                return Feature(
                    entity_id=row['entity_id'],
                    feature_name=row['feature_name'],
                    feature_value=row['feature_value'],
                    event_timestamp=row['event_timestamp'],
                    ttl_seconds=row['ttl_seconds'],
                    version=row['version'],
                    created_at=row['created_at']
                )
            
            return None
    
    async def _get_stale_feature(self, entity_id: str, feature_name: str) -> Optional[Feature]:
        """Get stale feature when circuit breaker is open"""
        # Try to get the most recent feature even if expired
        query = """
            SELECT entity_id, feature_name, feature_value, 
                   event_timestamp, ttl_seconds, version, created_at
            FROM feature_store 
            WHERE entity_id = $1 
            AND feature_name = $2 
            ORDER BY event_timestamp DESC 
            LIMIT 1
        """
        
        try:
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(query, entity_id, feature_name)
                
                if row:
                    return Feature(
                        entity_id=row['entity_id'],
                        feature_name=row['feature_name'],
                        feature_value=row['feature_value'],
                        event_timestamp=row['event_timestamp'],
                        ttl_seconds=row['ttl_seconds'],
                        version=row['version'],
                        created_at=row['created_at']
                    )
        
        except Exception as e:
            structlog.warning("Failed to get stale feature", error=str(e))
        
        return None
    
    async def _is_hot_feature(self, entity_id: str, feature_name: str) -> bool:
        """Check if feature should be in hot cache"""
        # In production, this would track access patterns
        # For now, use a simple heuristic based on entity_id
        return hash(entity_id) % 10 == 0  # 10% of features are "hot"
    
    async def get_features_batch(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        version: str = "latest"
    ) -> Dict[str, Dict[str, Optional[Feature]]]:
        """Get multiple features in batch"""
        
        results = {}
        
        # Create tasks for concurrent execution
        tasks = []
        for entity_id in entity_ids:
            entity_features = {}
            for feature_name in feature_names:
                task = self.get_feature(entity_id, feature_name, version)
                tasks.append((entity_id, feature_name, task))
            
            results[entity_id] = {}
        
        # Execute tasks concurrently
        task_results = await asyncio.gather(
            *[task for _, _, task in tasks],
            return_exceptions=True
        )
        
        # Collect results
        for i, (entity_id, feature_name, _) in enumerate(tasks):
            result = task_results[i]
            if isinstance(result, Exception):
                structlog.error(
                    "Failed to get feature in batch",
                    entity_id=entity_id,
                    feature_name=feature_name,
                    error=str(result)
                )
                results[entity_id][feature_name] = None
            else:
                results[entity_id][feature_name] = result
        
        return results
    
    async def set_feature(
        self,
        entity_id: str,
        feature_name: str,
        feature_value: float,
        ttl_seconds: int,
        version: str = "latest"
    ) -> None:
        """Set feature value in store"""
        
        try:
            # Check circuit breaker
            if self.circuit_breaker and self.circuit_breaker.is_open():
                raise CircuitBreakerOpenError("Circuit breaker is open")
            
            # Create feature object
            feature = Feature(
                entity_id=entity_id,
                feature_name=feature_name,
                feature_value=feature_value,
                event_timestamp=datetime.now(),
                ttl_seconds=ttl_seconds,
                version=version
            )
            
            # Insert into PostgreSQL
            await self._set_feature_in_db(feature)
            
            # Cache in Redis
            cache_key = f"feature:{entity_id}:{feature_name}:{version}"
            feature_data = json.dumps(asdict(feature), default=str)
            
            if self.redis_client:
                await self.redis_client.setex(cache_key, ttl_seconds, feature_data)
            
            structlog.info(
                "Feature set successfully",
                entity_id=entity_id,
                feature_name=feature_name,
                version=version
            )
            
        except Exception as e:
            # Trip circuit breaker on failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            structlog.error(
                "Failed to set feature",
                entity_id=entity_id,
                feature_name=feature_name,
                error=str(e)
            )
            raise
    
    async def _set_feature_in_db(self, feature: Feature) -> None:
        """Insert feature into PostgreSQL"""
        
        query = """
            INSERT INTO feature_store 
            (entity_id, feature_name, feature_value, event_timestamp, ttl_seconds, version, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                query,
                feature.entity_id,
                feature.feature_name,
                feature.feature_value,
                feature.event_timestamp,
                feature.ttl_seconds,
                feature.version,
                datetime.now()
            )
    
    async def get_metrics(self) -> FeatureStoreMetrics:
        """Get current performance metrics"""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for feature store"""
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "metrics": asdict(self.metrics),
            "circuit_breaker": {
                "enabled": self.enable_circuit_breaker,
                "is_open": self.circuit_breaker.is_open() if self.circuit_breaker else False,
                "failure_count": self.circuit_breaker.failure_count if self.circuit_breaker else 0
            },
            "connections": {
                "postgres": False,
                "redis": False
            }
        }
        
        # Test PostgreSQL connection
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["connections"]["postgres"] = True
        except:
            health_status["status"] = "unhealthy"
        
        # Test Redis connection
        try:
            await self.redis_client.ping()
            health_status["connections"]["redis"] = True
        except:
            health_status["status"] = "unhealthy"
        
        return health_status


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_failure(self) -> None:
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            structlog.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )
    
    def record_success(self) -> None:
        """Record a success"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            structlog.info("Circuit breaker closed")


async def create_feature_store_table(postgres_dsn: str) -> None:
    """Create feature store table in PostgreSQL"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS feature_store (
        entity_id VARCHAR(64) NOT NULL,
        feature_name VARCHAR(128) NOT NULL,
        feature_value FLOAT NOT NULL,
        event_timestamp TIMESTAMP NOT NULL,
        ttl_seconds INT NOT NULL,
        version VARCHAR(32) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (entity_id, feature_name, version)
    );
    
    CREATE INDEX IF NOT EXISTS idx_feature_store_entity 
    ON feature_store(entity_id);
    
    CREATE INDEX IF NOT EXISTS idx_feature_store_feature 
    ON feature_store(feature_name);
    
    CREATE INDEX IF NOT EXISTS idx_feature_store_timestamp 
    ON feature_store(event_timestamp);
    """
    
    try:
        pool = await asyncpg.create_pool(postgres_dsn)
        async with pool.acquire() as conn:
            await conn.execute(create_table_sql)
        
        await pool.close()
        structlog.info("Feature store table created successfully")
        
    except Exception as e:
        structlog.error("Failed to create feature store table", error=str(e))
        raise


async def main():
    """Example usage"""
    # Configuration
    postgres_dsn = "postgresql://user:password@localhost:5432/featurestore"
    redis_url = "redis://localhost:6379/0"
    
    # Create feature store client
    client = FeatureStoreClient(
        postgres_dsn=postgres_dsn,
        redis_url=redis_url,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_timeout_seconds=60,
        cache_ttl_seconds=3600
    )
    
    try:
        # Initialize client
        await client.initialize()
        
        # Example: Set a feature
        await client.set_feature(
            entity_id="CUST_12345678",
            feature_name="monthly_charges",
            feature_value=79.99,
            ttl_seconds=3600,
            version="v1.0"
        )
        
        # Example: Get a feature
        feature = await client.get_feature(
            entity_id="CUST_12345678",
            feature_name="monthly_charges",
            version="v1.0"
        )
        
        if feature:
            print(f"Feature value: {feature.feature_value}")
        
        # Get metrics
        metrics = await client.get_metrics()
        print(f"Cache hit rate: {metrics.cache_hits / max(metrics.total_requests, 1):.2%}")
        
        # Health check
        health = await client.health_check()
        print(f"Health status: {health['status']}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())