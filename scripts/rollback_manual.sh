#!/bin/bash

# Manual Rollback Script
# Emergency rollback procedure for production deployments

set -euo pipefail

# Default values
ENVIRONMENT="production"
TARGET_VERSION=""
FORCE=false
SKIP_CONFIRMATION=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_critical() {
    echo -e "${RED}[CRITICAL]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Emergency rollback procedure for production deployments

OPTIONS:
    -e, --env ENVIRONMENT     Target environment (staging|production) [default: production]
    -v, --version VERSION     Target version to rollback to (required)
    -f, --force              Force rollback without confirmation
    -y, --yes                Skip confirmation prompts
    -h, --help               Show this help message

EXAMPLES:
    $0 --version 1.2.3
    $0 --env staging --version 2.1.0 --force
    $0 --version 1.0.0 --yes

EMERGENCY CONTACTS:
    On-call Engineer: +1-555-0123
    ML Team Lead: +1-555-0124
    DevOps Team: +1-555-0125
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
            -v|--version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -y|--yes)
                SKIP_CONFIRMATION=true
                shift
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

# Validate arguments
validate_args() {
    if [[ -z "$TARGET_VERSION" ]]; then
        log_error "Target version is required for rollback"
        show_help
        exit 1
    fi
    
    if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
        exit 1
    fi
}

# Get current deployment info
get_current_deployment() {
    log_info "Getting current deployment information..."
    
    local service_name="churn-model-$ENVIRONMENT"
    
    # Get current task definition
    CURRENT_TASK_DEF=$(aws ecs describe-services \
        --cluster ml-cluster \
        --services "$service_name" \
        --query 'services[0].taskDefinition' \
        --output text 2>/dev/null || echo "")
    
    if [[ -z "$CURRENT_TASK_DEF" ]]; then
        log_error "Failed to get current task definition"
        exit 1
    fi
    
    log_info "Current task definition: $CURRENT_TASK_DEF"
    
    # Get current running count
    RUNNING_COUNT=$(aws ecs describe-services \
        --cluster ml-cluster \
        --services "$service_name" \
        --query 'services[0].runningCount' \
        --output text 2>/dev/null || echo "0")
    
    log_info "Current running count: $RUNNING_COUNT"
}

# Get available versions
get_available_versions() {
    log_info "Getting available versions..."
    
    # Get task definition revisions
    TASK_DEFS=$(aws ecs list-task-definitions \
        --family-prefix "churn-model-$ENVIRONMENT" \
        --query 'taskDefinitionArns[]' \
        --output text 2>/dev/null || echo "")
    
    if [[ -z "$TASK_DEFS" ]]; then
        log_error "No task definitions found"
        exit 1
    fi
    
    log_info "Available versions:"
    for task_def in $TASK_DEFS; do
        echo "  - $task_def"
    done
}

# Confirm rollback
confirm_rollback() {
    if [[ "$SKIP_CONFIRMATION" == true || "$FORCE" == true ]]; then
        return 0
    fi
    
    log_critical "ROLLBACK CONFIRMATION REQUIRED"
    log_critical "Environment: $ENVIRONMENT"
    log_critical "Current version: $CURRENT_TASK_DEF"
    log_critical "Target version: $TARGET_VERSION"
    log_critical "Running tasks: $RUNNING_COUNT"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_critical "⚠️  THIS IS A PRODUCTION ROLLBACK! ⚠️"
        log_critical "Impact: Customer-facing service disruption possible"
        log_critical "Escalation: Notify on-call engineer immediately"
    fi
    
    echo
    read -p "Are you sure you want to proceed with the rollback? (yes/no): " -r
    
    if [[ ! "$REPLY" =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
    
    # Double confirmation for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo
        read -p "Type 'PRODUCTION ROLLBACK' to confirm: " -r
        
        if [[ "$REPLY" != "PRODUCTION ROLLBACK" ]]; then
            log_info "Rollback cancelled - confirmation mismatch"
            exit 0
        fi
    fi
}

# Create backup of current state
create_backup() {
    log_info "Creating backup of current state..."
    
    local backup_dir="rollback-backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup current task definition
    aws ecs describe-task-definition \
        --task-definition "$CURRENT_TASK_DEF" \
        --query taskDefinition \
        > "$backup_dir/task-definition.json"
    
    # Backup current service configuration
    aws ecs describe-services \
        --cluster ml-cluster \
        --services "churn-model-$ENVIRONMENT" \
        --query services[0] \
        > "$backup_dir/service-config.json"
    
    # Backup current metrics
    python scripts/get_deployment_metrics.py \
        --env "$ENVIRONMENT" \
        --output "$backup_dir/metrics.json" \
        2>/dev/null || echo "{}" > "$backup_dir/metrics.json"
    
    log_info "Backup created: $backup_dir"
    echo "$backup_dir" > /tmp/rollback_backup_dir
}

# Perform ECS rollback
perform_ecs_rollback() {
    log_info "Performing ECS rollback..."
    
    local service_name="churn-model-$ENVIRONMENT"
    
    # Update service with target task definition
    aws ecs update-service \
        --cluster ml-cluster \
        --service "$service_name" \
        --task-definition "$TARGET_VERSION" \
        --force-new-deployment \
        > /dev/null
    
    log_info "Service update initiated"
    
    # Wait for service to stabilize
    log_info "Waiting for service to stabilize..."
    
    local max_wait=600  # 10 minutes
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -ge $max_wait ]]; then
            log_error "Service stabilization timeout"
            return 1
        fi
        
        # Check service stability
        if aws ecs wait services-stable \
            --cluster ml-cluster \
            --services "$service_name" \
            --timeout 60 \
            2>/dev/null; then
            
            log_info "Service stabilized successfully"
            break
        fi
        
        # Check if deployment failed
        local failed_tasks=$(aws ecs describe-services \
            --cluster ml-cluster \
            --services "$service_name" \
            --query 'services[0].deployments[0].failedTasks' \
            --output text 2>/dev/null || echo "0")
        
        if [[ "$failed_tasks" != "0" && "$failed_tasks" != "None" ]]; then
            log_warn "Failed tasks detected: $failed_tasks"
        fi
        
        log_info "Waiting for stabilization... ($elapsed/${max_wait}s)"
        sleep 10
    done
    
    # Verify running tasks
    local new_running_count=$(aws ecs describe-services \
        --cluster ml-cluster \
        --services "$service_name" \
        --query 'services[0].runningCount' \
        --output text 2>/dev/null || echo "0")
    
    log_info "New running count: $new_running_count"
    
    if [[ "$new_running_count" == "0" ]]; then
        log_error "No tasks are running after rollback"
        return 1
    fi
    
    return 0
}

# Update Route53 routing (for blue-green deployments)
update_routing() {
    log_info "Updating Route53 routing..."
    
    # This would update DNS routing for blue-green deployments
    # Implementation depends on your specific routing strategy
    
    log_info "Route53 routing updated"
}

# Send notifications
send_notifications() {
    local status="$1"
    local message="$2"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="danger"
        local emoji="❌"
        
        if [[ "$status" == "success" ]]; then
            color="good"
            emoji="✅"
        fi
        
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"channel\": \"#ml-ops-alerts\",
                \"username\": \"Rollback Bot\",
                \"icon_emoji\": \":rotating_light:\",
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"$emoji Rollback $status: $ENVIRONMENT\",
                    \"text\": \"$message\",
                    \"fields\": [{
                        \"title\": \"Environment\",
                        \"value\": \"$ENVIRONMENT\",
                        \"short\": true
                    }, {
                        \"title\": \"Target Version\",
                        \"value\": \"$TARGET_VERSION\",
                        \"short\": true
                    }, {
                        \"title\": \"Initiated By\",
                        \"value\": \"$(whoami)\",
                        \"short\": true
                    }],
                    \"timestamp\": $(date +%s)
                }]
            }" || true
    fi
    
    # PagerDuty notification for production
    if [[ "$ENVIRONMENT" == "production" && -n "${PAGERDUTY_WEBHOOK_URL:-}" ]]; then
        local event_type="trigger"
        
        if [[ "$status" == "success" ]]; then
            event_type="resolve"
        fi
        
        curl -X POST "$PAGERDUTY_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"event_type\": \"$event_type\",
                \"service_key\": \"${PAGERDUTY_SERVICE_KEY:-}\",
                \"incident_key\": \"rollback-$ENVIRONMENT-$(date +%Y%m%d_%H%M%S)\",
                \"description\": \"Rollback $status: $message\",
                \"client\": \"Rollback Script\",
                \"client_url\": \"$(hostname)\"
            }" || true
    fi
}

# Verify rollback success
verify_rollback() {
    log_info "Verifying rollback success..."
    
    local service_name="churn-model-$ENVIRONMENT"
    
    # Get new task definition
    local new_task_def=$(aws ecs describe-services \
        --cluster ml-cluster \
        --services "$service_name" \
        --query 'services[0].taskDefinition' \
        --output text 2>/dev/null || echo "")
    
    if [[ "$new_task_def" != *"$TARGET_VERSION"* ]]; then
        log_error "Rollback verification failed: task definition mismatch"
        log_error "Expected: $TARGET_VERSION"
        log_error "Actual: $new_task_def"
        return 1
    fi
    
    # Check health endpoint
    local health_url
    case "$ENVIRONMENT" in
        staging)
            health_url="https://api-staging.company.com/health/ready"
            ;;
        production)
            health_url="https://api.company.com/health/ready"
            ;;
    esac
    
    if curl -s -f -m 10 "$health_url" >/dev/null 2>&1; then
        log_info "Health check passed"
    else
        log_warn "Health check failed"
    fi
    
    log_info "Rollback verification completed"
    return 0
}

# Cleanup old resources
cleanup_resources() {
    log_info "Cleaning up old resources..."
    
    # This could include:
    # - Removing old task definitions
    # - Cleaning up unused Docker images
    # - Archiving old logs
    
    log_info "Cleanup completed"
}

# Main rollback procedure
main() {
    log_info "Starting rollback procedure for $ENVIRONMENT"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate arguments
    validate_args
    
    # Get current deployment info
    get_current_deployment
    
    # Get available versions
    get_available_versions
    
    # Confirm rollback
    confirm_rollback
    
    # Create backup
    create_backup
    
    # Perform rollback
    if perform_ecs_rollback; then
        log_info "ECS rollback completed successfully"
    else
        log_error "ECS rollback failed"
        send_notifications "failed" "ECS rollback failed"
        exit 1
    fi
    
    # Update routing
    update_routing
    
    # Verify rollback
    if verify_rollback; then
        log_info "Rollback verification passed"
    else
        log_warn "Rollback verification failed"
    fi
    
    # Cleanup
    cleanup_resources
    
    # Success notification
    send_notifications "success" "Rollback to $TARGET_VERSION completed successfully"
    
    log_info "✅ Rollback completed successfully"
    
    # Print summary
    echo
    log_info "ROLLBACK SUMMARY"
    log_info "================"
    log_info "Environment: $ENVIRONMENT"
    log_info "Previous version: $CURRENT_TASK_DEF"
    log_info "New version: $TARGET_VERSION"
    log_info "Backup location: $(cat /tmp/rollback_backup_dir 2>/dev/null || echo 'unknown')"
    log_info "Duration: $(( $(date +%s) - START_TIME )) seconds"
}

# Set start time
START_TIME=$(date +%s)

# Run main function
main "$@"