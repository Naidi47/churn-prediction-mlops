-- Feature Store Database Schema
-- Production-grade schema for feature storage with TTL and versioning

-- Create database (run manually)
-- CREATE DATABASE featurestore;

-- Connect to database
-- \c featurestore;

-- Create feature store table with TTL support
CREATE TABLE IF NOT EXISTS feature_store (
    entity_id VARCHAR(64) NOT NULL,
    feature_name VARCHAR(128) NOT NULL,
    feature_value FLOAT NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    ttl_seconds INT NOT NULL DEFAULT 3600,
    version VARCHAR(32) NOT NULL DEFAULT 'latest',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Primary key for uniqueness
    PRIMARY KEY (entity_id, feature_name, version),
    
    -- Constraints
    CHECK (ttl_seconds > 0),
    CHECK (feature_value IS NOT NULL),
    CHECK (event_timestamp IS NOT NULL)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_store_entity 
ON feature_store(entity_id);

CREATE INDEX IF NOT EXISTS idx_feature_store_feature 
ON feature_store(feature_name);

CREATE INDEX IF NOT EXISTS idx_feature_store_timestamp 
ON feature_store(event_timestamp);

CREATE INDEX IF NOT EXISTS idx_feature_store_created_at 
ON feature_store(created_at);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_feature_store_entity_feature 
ON feature_store(entity_id, feature_name);

-- Index for version-based queries
CREATE INDEX IF NOT EXISTS idx_feature_store_version 
ON feature_store(version);

-- Create audit table for feature access tracking (GDPR compliance)
CREATE TABLE IF NOT EXISTS feature_access_audit (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(64) NOT NULL,
    feature_name VARCHAR(128) NOT NULL,
    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_method VARCHAR(50) NOT NULL, -- 'cache', 'database', 'stale'
    user_id VARCHAR(64), -- For audit trail
    request_id VARCHAR(64), -- For request correlation
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    
    -- Indexes for audit queries
    INDEX idx_audit_entity (entity_id),
    INDEX idx_audit_timestamp (access_timestamp),
    INDEX idx_audit_user (user_id),
    INDEX idx_audit_request (request_id)
);

-- Create feature metadata table for governance
CREATE TABLE IF NOT EXISTS feature_metadata (
    feature_name VARCHAR(128) PRIMARY KEY,
    description TEXT,
    data_type VARCHAR(50) NOT NULL,
    feature_category VARCHAR(50) NOT NULL, -- 'numerical', 'categorical', 'datetime'
    owner VARCHAR(64),
    data_source VARCHAR(128),
    transformation_logic TEXT,
    privacy_level VARCHAR(20) DEFAULT 'public', -- 'public', 'internal', 'restricted'
    pii_flag BOOLEAN DEFAULT FALSE,
    compliance_required BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (privacy_level IN ('public', 'internal', 'restricted')),
    CHECK (data_type IN ('float', 'integer', 'string', 'boolean', 'datetime'))
);

-- Create feature lineage table for data lineage tracking
CREATE TABLE IF NOT EXISTS feature_lineage (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(128) NOT NULL,
    source_table VARCHAR(128) NOT NULL,
    source_column VARCHAR(128) NOT NULL,
    transformation_sql TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key to feature metadata
    FOREIGN KEY (feature_name) REFERENCES feature_metadata(feature_name) ON DELETE CASCADE,
    
    -- Index for lineage queries
    INDEX idx_lineage_feature (feature_name),
    INDEX idx_lineage_source (source_table)
);

-- Create model feature mapping table
CREATE TABLE IF NOT EXISTS model_feature_mapping (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    feature_name VARCHAR(128) NOT NULL,
    feature_position INT NOT NULL,
    is_required BOOLEAN DEFAULT TRUE,
    default_value FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite unique constraint
    UNIQUE (model_name, model_version, feature_name),
    
    -- Foreign key to feature metadata
    FOREIGN KEY (feature_name) REFERENCES feature_metadata(feature_name) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_model_feature_model (model_name, model_version),
    INDEX idx_model_feature_feature (feature_name)
);

-- Create feature drift monitoring table
CREATE TABLE IF NOT EXISTS feature_drift_monitor (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(128) NOT NULL,
    drift_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    drift_score FLOAT NOT NULL,
    drift_threshold FLOAT NOT NULL DEFAULT 0.1,
    distribution_stats JSONB,
    alert_sent BOOLEAN DEFAULT FALSE,
    
    -- Foreign key to feature metadata
    FOREIGN KEY (feature_name) REFERENCES feature_metadata(feature_name) ON DELETE CASCADE,
    
    -- Indexes
    INDEX idx_drift_feature (feature_name),
    INDEX idx_drift_timestamp (drift_timestamp),
    INDEX idx_drift_score (drift_score)
);

-- Insert sample feature metadata
INSERT INTO feature_metadata (
    feature_name, description, data_type, feature_category, 
    owner, data_source, transformation_logic, privacy_level, pii_flag
) VALUES 
('monthly_charges', 'Customer monthly charges in USD', 'float', 'numerical', 'billing-team', 'subscriptions_table', 'monthly_recurring_charges + usage_charges', 'internal', FALSE),
('tenure_days', 'Customer tenure in days', 'integer', 'numerical', 'ops-team', 'customers_table', 'DATEDIFF(current_date, signup_date)', 'public', FALSE),
('subscription_type', 'Type of subscription plan', 'string', 'categorical', 'product-team', 'subscriptions_table', 'subscription_plan_name', 'public', FALSE),
('payment_method', 'Customer payment method', 'string', 'categorical', 'billing-team', 'billing_table', 'primary_payment_method', 'internal', FALSE),
('region', 'Customer geographical region', 'string', 'categorical', 'ops-team', 'customers_table', 'customer_region', 'public', FALSE),
('contract_type', 'Contract type (monthly/yearly)', 'string', 'categorical', 'sales-team', 'contracts_table', 'contract_type', 'public', FALSE),
('number_of_services', 'Number of active services', 'integer', 'numerical', 'ops-team', 'services_table', 'COUNT(active_services)', 'public', FALSE),
('number_of_dependents', 'Number of dependent family members', 'integer', 'numerical', 'ops-team', 'customers_table', 'dependent_count', 'restricted', TRUE),
('internet_service', 'Type of internet service', 'string', 'categorical', 'network-team', 'services_table', 'internet_service_type', 'public', FALSE),
('phone_service', 'Phone service flag', 'string', 'categorical', 'network-team', 'services_table', 'phone_service_flag', 'public', FALSE),
('total_charges', 'Total charges to date', 'float', 'numerical', 'billing-team', 'billing_table', 'SUM(all_charges)', 'internal', FALSE),
('gender', 'Customer gender', 'string', 'categorical', 'ops-team', 'customers_table', 'gender', 'restricted', TRUE);

-- Create stored procedure for TTL cleanup
CREATE OR REPLACE FUNCTION cleanup_expired_features()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM feature_store 
    WHERE event_timestamp + (ttl_seconds || ' seconds')::INTERVAL < CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create stored procedure for feature access logging
CREATE OR REPLACE FUNCTION log_feature_access(
    p_entity_id VARCHAR(64),
    p_feature_name VARCHAR(128),
    p_access_method VARCHAR(50),
    p_user_id VARCHAR(64) DEFAULT NULL,
    p_request_id VARCHAR(64) DEFAULT NULL,
    p_success BOOLEAN DEFAULT TRUE,
    p_error_message TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO feature_access_audit (
        entity_id, feature_name, access_method, user_id, request_id, success, error_message
    ) VALUES (
        p_entity_id, p_feature_name, p_access_method, p_user_id, p_request_id, p_success, p_error_message
    );
END;
$$ LANGUAGE plpgsql;

-- Create view for feature store statistics
CREATE OR REPLACE VIEW feature_store_stats AS
SELECT 
    feature_name,
    COUNT(*) as total_records,
    COUNT(DISTINCT entity_id) as unique_entities,
    AVG(feature_value) as avg_value,
    MIN(feature_value) as min_value,
    MAX(feature_value) as max_value,
    MIN(event_timestamp) as earliest_timestamp,
    MAX(event_timestamp) as latest_timestamp,
    AVG(ttl_seconds) as avg_ttl
FROM feature_store
GROUP BY feature_name;

-- Create view for model lineage
CREATE OR REPLACE VIEW model_feature_lineage AS
SELECT 
    mfm.model_name,
    mfm.model_version,
    mfm.feature_name,
    mfm.feature_position,
    mfm.is_required,
    mfm.default_value,
    fm.description,
    fm.data_type,
    fm.feature_category,
    fm.owner,
    fm.privacy_level,
    fm.pii_flag
FROM model_feature_mapping mfm
JOIN feature_metadata fm ON mfm.feature_name = fm.feature_name;

-- Create scheduled job for TTL cleanup (PostgreSQL 11+)
-- This would typically be handled by an external scheduler like pg_cron
-- or application-level cleanup tasks

-- Grant permissions (adjust roles as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ml_app;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO feature_store_user;

-- Add comments for documentation
COMMENT ON TABLE feature_store IS 'Main feature store table with TTL support';
COMMENT ON TABLE feature_access_audit IS 'Audit trail for GDPR compliance';
COMMENT ON TABLE feature_metadata IS 'Feature metadata for governance';
COMMENT ON TABLE feature_lineage IS 'Feature lineage tracking for data quality';
COMMENT ON TABLE model_feature_mapping IS 'Mapping between models and their required features';
COMMENT ON TABLE feature_drift_monitor IS 'Feature drift monitoring and alerts';

-- Set table-level TTL (for automatic cleanup)
-- Note: This is a placeholder - actual TTL would be implemented via:
-- 1. Application logic
-- 2. Database triggers
-- 3. External cleanup jobs
-- 4. Partitioning with automatic drop