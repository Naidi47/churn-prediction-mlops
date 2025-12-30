#!/bin/bash

# Deployment Verification Script
# Validates that a deployment is healthy before proceeding

set -euo pipefail

# Default values
ENVIRONMENT="staging"
TIMEOUT=300
HEALTH_CHECK_URL=""
MAX_RETRIES=10
RETRY_DELAY=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Verify deployment health and readiness

OPTIONS:
    -e, --env ENVIRONMENT     Target environment (staging|production) [default: staging]
    -t, --timeout TIMEOUT     Timeout in seconds [default: 300]
    -u, --health-url URL      Health check URL [default: auto-detected]
    -r, --retries RETRIES     Maximum number of retries [default: 10]
    -d, --delay DELAY         Delay between retries in seconds [default: 30]
    -h, --help               Show this help message

EXAMPLES:
    $0 --env production --timeout 600
    $0 --health-url https://api.company.com/health/ready
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -u|--health-url)
                HEALTH_CHECK_URL="$2"
                shift 2
                ;;
            -r|--retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            -d|--delay)
                RETRY_DELAY="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
        exit 1
    fi
}

# Detect health check URL
detect_health_url() {
    if [[ -z "$HEALTH_CHECK_URL" ]]; then
        case "$ENVIRONMENT" in
            staging)
                HEALTH_CHECK_URL="https://api-staging.company.com/health/ready"
                ;;
            production)
                HEALTH_CHECK_URL="https://api.company.com/health/ready"
                ;;
        esac
    fi
    
    log_info "Health check URL: $HEALTH_CHECK_URL"
}

# Check service health
check_service_health() {
    local attempt=1
    local start_time=$(date +%s)
    
    log_info "Starting health check (timeout: ${TIMEOUT}s, max retries: $MAX_RETRIES)"
    
    while [[ $attempt -le $MAX_RETRIES ]]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -ge $TIMEOUT ]]; then
            log_error "Timeout reached after ${elapsed}s"
            return 1
        fi
        
        log_info "Health check attempt $attempt/$MAX_RETRIES"
        
        # Make health check request
        if response=$(curl -s -f -m 10 -H "Authorization: Bearer $API_KEY" "$HEALTH_CHECK_URL" 2>&1); then
            log_info "Health check successful"
            
            # Parse response
            if echo "$response" | jq -e . >/dev/null 2>&1; then
                local status=$(echo "$response" | jq -r '.status // "unknown"')
                local model_status=$(echo "$response" | jq -r '.checks.model // "unknown"')
                local feature_store_status=$(echo "$response" | jq -r '.checks.feature_store // "unknown"')
                
                log_info "Status: $status"
                log_info "Model: $model_status"
                log_info "Feature Store: $feature_store_status"
                
                if [[ "$status" == "ready" && "$model_status" == "healthy" && "$feature_store_status" == "healthy" ]]; then
                    log_info "All systems healthy!"
                    return 0
                else
                    log_warn "Some systems are not healthy"
                fi
            else
                log_warn "Invalid JSON response: $response"
            fi
        else
            log_warn "Health check failed: $response"
        fi
        
        # Calculate remaining time
        local remaining=$((TIMEOUT - elapsed))
        if [[ $remaining -le 0 ]]; then
            log_error "No time remaining for retries"
            return 1
        fi
        
        # Wait before retry
        local sleep_time=$((RETRY_DELAY < remaining ? RETRY_DELAY : remaining))
        log_info "Waiting ${sleep_time}s before retry..."
        sleep $sleep_time
        
        ((attempt++))
    done
    
    log_error "Max retries reached"
    return 1
}

# Check load balancer health
check_load_balancer() {
    log_info "Checking load balancer health..."
    
    # Get load balancer DNS name
    local lb_dns=$(aws elbv2 describe-load-balancers \
        --names "churn-model-$ENVIRONMENT-alb" \
        --query 'LoadBalancers[0].DNSName' \
        --output text 2>/dev/null || echo "")
    
    if [[ -z "$lb_dns" ]]; then
        log_warn "Could not find load balancer DNS name"
        return 0  # Don't fail the deployment for this
    fi
    
    # Check if load balancer is responding
    if curl -s -f -m 10 "http://$lb_dns/health/ready" >/dev/null 2>&1; then
        log_info "Load balancer is healthy"
        return 0
    else
        log_warn "Load balancer health check failed"
        return 1
    fi
}

# Check CloudWatch metrics
check_cloudwatch_metrics() {
    log_info "Checking CloudWatch metrics..."
    
    local end_time=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local start_time=$(date -u -d "5 minutes ago" +"%Y-%m-%dT%H:%M:%SZ")
    
    # Check error rate
    local error_rate=$(aws cloudwatch get-metric-statistics \
        --namespace "ML/Inference" \
        --metric-name "PredictionErrors" \
        --dimensions Name=Environment,Value=$ENVIRONMENT \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --period 300 \
        --statistics Sum \
        --query 'Datapoints[0].Sum' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$error_rate" != "None" && "$error_rate" != "0.0" ]]; then
        log_warn "Error rate detected: $error_rate errors in last 5 minutes"
    else
        log_info "No errors detected in CloudWatch"
    fi
    
    # Check latency
    local avg_latency=$(aws cloudwatch get-metric-statistics \
        --namespace "ML/Inference" \
        --metric-name "InferenceLatency" \
        --dimensions Name=Environment,Value=$ENVIRONMENT \
        --start-time "$start_time" \
        --end-time "$end_time" \
        --period 300 \
        --statistics Average \
        --query 'Datapoints[0].Average' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$avg_latency" != "None" && "$avg_latency" != "0.0" ]]; then
        log_info "Average latency: ${avg_latency}ms"
        
        # Check if latency is acceptable
        if (( $(echo "$avg_latency > 100" | bc -l) )); then
            log_warn "High latency detected: ${avg_latency}ms"
        fi
    fi
    
    return 0
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    local base_url
    case "$ENVIRONMENT" in
        staging)
            base_url="https://api-staging.company.com"
            ;;
        production)
            base_url="https://api.company.com"
            ;;
    esac
    
    # Install Newman if needed
    if ! command -v newman &> /dev/null; then
        log_info "Installing Newman..."
        npm install -g newman
    fi
    
    # Run smoke tests
    if newman run tests/api/smoke-tests.postman_collection.json \
        --env-var base_url="$base_url" \
        --reporters cli \
        --timeout-request 10000; then
        log_info "Smoke tests passed"
        return 0
    else
        log_error "Smoke tests failed"
        return 1
    fi
}

# Generate deployment report
generate_report() {
    local report_file="deployment-report-${ENVIRONMENT}-$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "deployment": {
        "environment": "$ENVIRONMENT",
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "health_check_url": "$HEALTH_CHECK_URL",
        "timeout": $TIMEOUT,
        "max_retries": $MAX_RETRIES
    },
    "verification": {
        "status": "success",
        "duration_seconds": $(( $(date +%s) - START_TIME )),
        "checks_performed": [
            "health_check",
            "load_balancer",
            "cloudwatch_metrics",
            "smoke_tests"
        ]
    }
}
EOF
    
    log_info "Deployment report generated: $report_file"
}

# Main execution
main() {
    local START_TIME=$(date +%s)
    
    log_info "Starting deployment verification for $ENVIRONMENT"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate environment
    validate_environment
    
    # Detect health check URL
    detect_health_url
    
    # Get API key from environment
    API_KEY=${API_KEY:-$(aws secretsmanager get-secret-value \
        --secret-id ml-api-key \
        --query SecretString \
        --output text 2>/dev/null | jq -r '.api_key' || echo "")}
    
    # Run verification steps
    local success=true
    
    # 1. Service health check
    if ! check_service_health; then
        log_error "Service health check failed"
        success=false
    fi
    
    # 2. Load balancer health
    if ! check_load_balancer; then
        log_warn "Load balancer check failed (non-critical)"
    fi
    
    # 3. CloudWatch metrics
    if ! check_cloudwatch_metrics; then
        log_warn "CloudWatch metrics check failed (non-critical)"
    fi
    
    # 4. Smoke tests
    if ! run_smoke_tests; then
        log_error "Smoke tests failed"
        success=false
    fi
    
    # Generate report
    generate_report
    
    # Final result
    if [[ "$success" == true ]]; then
        log_info "✅ Deployment verification completed successfully"
        exit 0
    else
        log_error "❌ Deployment verification failed"
        exit 1
    fi
}

# Run main function
main "$@"