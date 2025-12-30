"""
Locust Load Testing Suite
Tests the inference service under realistic load conditions
"""

import json
import random
import time
from typing import Dict, Any
from locust import HttpUser, task, between, events
from locust.stats import stats_printer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictionUser(HttpUser):
    """Simulates realistic user behavior for churn prediction API"""
    
    # Wait time between requests (realistic user behavior)
    wait_time = between(0.1, 2.0)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Customer ID pool for realistic testing
        self.customer_ids = [
            f"CUST_{str(i).zfill(8)}" for i in range(1, 10001)
        ]
        
        # API key for authentication
        self.api_key = "test-api-key-123"
        
        # Track user session
        self.request_count = 0
        self.start_time = time.time()
    
    def on_start(self):
        """Called when a user starts"""
        logger.info(f"User started: {self.client.base_url}")
    
    def on_stop(self):
        """Called when a user stops"""
        logger.info(f"User stopped after {self.request_count} requests")
    
    @task(70)
    def predict_churn(self):
        """Main prediction task - 70% of traffic"""
        
        # Select random customer
        customer_id = random.choice(self.customer_ids)
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": f"load-test-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        }
        
        payload = {
            "customer_id": customer_id
        }
        
        # Make prediction request
        with self.client.post(
            "/predict",
            json=payload,
            headers=headers,
            name="/predict",
            catch_response=True
        ) as response:
            
            self.request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Log interesting metrics
                if data.get("is_fallback"):
                    response.failure("Fallback model used")
                elif data.get("latency_ms", 0) > 100:
                    response.failure(f"High latency: {data.get('latency_ms')}ms")
                else:
                    response.success()
                    
                    # Track additional metrics
                    if hasattr(self.environment, 'stats'):
                        self.environment.stats.log_request(
                            "POST", "/predict", 
                            response.elapsed.total_seconds() * 1000, 
                            response.content_length or 0
                        )
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(15)
    def predict_with_features(self):
        """Prediction with pre-computed features - 15% of traffic"""
        
        customer_id = random.choice(self.customer_ids)
        
        # Generate realistic feature vector
        feature_vector = self._generate_feature_vector()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "customer_id": customer_id,
            "feature_vector": feature_vector
        }
        
        with self.client.post(
            "/predict",
            json=payload,
            headers=headers,
            name="/predict (with features)",
            catch_response=True
        ) as response:
            
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(10)
    def health_check(self):
        """Health check task - 10% of traffic"""
        
        with self.client.get("/health/ready", name="/health/ready") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    def get_model_info(self):
        """Model info task - 5% of traffic"""
        
        with self.client.get("/model/info", name="/model/info") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")
    
    def _generate_feature_vector(self) -> list:
        """Generate realistic feature vector for testing"""
        
        # Simulate realistic customer data
        monthly_charges = random.uniform(20, 200)
        tenure_days = random.uniform(1, 3650)
        number_of_services = random.randint(1, 8)
        number_of_dependents = random.randint(0, 5)
        total_charges = monthly_charges * (tenure_days / 30) * random.uniform(0.8, 1.2)
        
        # Create feature vector (20 features)
        features = [
            monthly_charges,  # monthly_charges
            tenure_days,      # tenure_days
            number_of_services,  # number_of_services
            number_of_dependents,  # number_of_dependents
            total_charges,    # total_charges
            random.uniform(0, 1),  # normalized feature 5
            random.uniform(0, 1),  # normalized feature 6
            random.uniform(0, 1),  # normalized feature 7
            random.uniform(0, 1),  # normalized feature 8
            random.uniform(0, 1),  # normalized feature 9
            random.uniform(0, 1),  # normalized feature 10
            random.uniform(0, 1),  # normalized feature 11
            random.uniform(0, 1),  # normalized feature 12
            random.uniform(0, 1),  # normalized feature 13
            random.uniform(0, 1),  # normalized feature 14
            random.uniform(0, 1),  # normalized feature 15
            random.uniform(0, 1),  # normalized feature 16
            random.uniform(0, 1),  # normalized feature 17
            random.uniform(0, 1),  # normalized feature 18
            random.uniform(0, 1),  # normalized feature 19
        ]
        
        return features


class BurstLoadUser(HttpUser):
    """Simulates burst traffic patterns"""
    
    wait_time = between(0.01, 0.1)  # Very short wait times for burst
    
    @task
    def burst_predictions(self):
        """High-frequency prediction requests"""
        
        customer_id = f"CUST_{random.randint(1, 1000):08d}"
        
        headers = {
            "Authorization": "Bearer test-api-key-123",
            "Content-Type": "application/json"
        }
        
        payload = {"customer_id": customer_id}
        
        self.client.post("/predict", json=payload, headers=headers)


class SustainedLoadUser(HttpUser):
    """Simulates sustained high load"""
    
    wait_time = between(0.05, 0.2)  # Sustained high throughput
    
    @task(90)
    def sustained_predictions(self):
        """Sustained prediction load"""
        
        customer_id = f"CUST_{random.randint(1, 5000):08d}"
        
        headers = {
            "Authorization": "Bearer test-api-key-123",
            "Content-Type": "application/json"
        }
        
        payload = {"customer_id": customer_id}
        
        with self.client.post(
            "/predict",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            
            if response.status_code != 200:
                # Log error but don't fail the test
                logger.warning(f"Request failed: {response.status_code}")
    
    @task(10)
    def health_checks(self):
        """Periodic health checks during sustained load"""
        
        self.client.get("/health/ready")


# Custom event handlers
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    logger.info(f"Load test started: {environment.host}")
    logger.info(f"Users: {environment.runner.target_user_count}")
    logger.info(f"Spawn rate: {environment.runner.spawn_rate}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    logger.info("Load test stopped")
    
    # Print final statistics
    if environment.stats.total.num_requests > 0:
        logger.info(f"Total requests: {environment.stats.total.num_requests}")
        logger.info(f"Failed requests: {environment.stats.total.num_failures}")
        logger.info(f"Average response time: {environment.stats.total.avg_response_time:.2f}ms")
        logger.info(f"95th percentile: {environment.stats.total.get_response_time_percentile(0.95):.2f}ms")
        
        # Calculate requests per second
        total_time = environment.stats.total.last_request_timestamp - environment.stats.total.start_time
        if total_time > 0:
            rps = environment.stats.total.num_requests / total_time
            logger.info(f"Requests per second: {rps:.2f}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Log individual request details"""
    # This can be used for detailed request logging
    pass


# Custom command line arguments for Locust
@events.init_command_line_parser.add_listener
def _(parser):
    """Add custom command line arguments"""
    parser.add_argument(
        "--api-key",
        type=str,
        default="test-api-key-123",
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--ramp-up-time",
        type=int,
        default=60,
        help="Time to ramp up users in seconds"
    )
    
    parser.add_argument(
        "--sustain-time",
        type=int,
        default=300,
        help="Time to sustain load in seconds"
    )


if __name__ == "__main__":
    # Run locust locally for testing
    import sys
    
    # Default settings for local testing
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    users = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    spawn_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run_time = sys.argv[4] if len(sys.argv) > 4 else "60s"
    
    # Configure and run
    import locust
    
    # This would typically be run with: locust -f locustfile.py --host=http://localhost:8000
    print(f"To run load test:")
    print(f"locust -f tests/load/locustfile.py --host={host} --users={users} --spawn-rate={spawn_rate} --run-time={run_time}")
    print(f"")
    print(f"Or for web interface:")
    print(f"locust -f tests/load/locustfile.py --host={host}")
    print(f"Then open http://localhost:8089")