import time
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Implementation of the Circuit Breaker pattern for service resilience"""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker state changed to OPEN")

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    @property
    def is_available(self):
        if self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF-OPEN"
                return True
            return False
        return True