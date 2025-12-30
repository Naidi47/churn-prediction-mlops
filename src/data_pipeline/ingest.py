"""
Data Ingestion Pipeline
Handles data ingestion from various sources with schema validation and error handling
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import backoff
from dataclasses import dataclass

# Configure logging
structlog = logging.getLogger(__name__)


@dataclass
class IngestionMetrics:
    """Metrics for data ingestion process"""
    records_ingested: int
    records_failed: int
    ingestion_time_seconds: float
    data_freshness_hours: float
    source_reliability: float


class DataIngestionError(Exception):
    """Custom exception for data ingestion failures"""
    pass


class DataIngestor:
    """Handles data ingestion with production-grade error handling and monitoring"""
    
    def __init__(
        self,
        source_url: str,
        output_path: str,
        sample_size: Optional[int] = None,
        random_seed: int = 42
    ):
        self.source_url = source_url
        self.output_path = Path(output_path)
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.s3_client = boto3.client('s3')
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    @backoff.on_exception(
        backoff.expo,
        (ClientError, BotoCoreError),
        max_tries=3,
        max_time=300
    )
    def _download_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """Download data from S3 with exponential backoff retry"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            if key.endswith('.parquet'):
                df = pd.read_parquet(obj['Body'])
            elif key.endswith('.csv'):
                df = pd.read_csv(obj['Body'])
            elif key.endswith('.json'):
                df = pd.read_json(obj['Body'])
            else:
                raise DataIngestionError(f"Unsupported file format: {key}")
            
            return df
            
        except Exception as e:
            structlog.error("Failed to download from S3", bucket=bucket, key=key, error=str(e))
            raise
    
    def _generate_synthetic_data(self, num_records: int = 100000) -> pd.DataFrame:
        """Generate synthetic customer churn data for demonstration"""
        np.random.seed(self.random_seed)
        
        # Customer demographics
        n_customers = num_records
        
        # Generate customer IDs
        customer_ids = [f"CUST_{str(i).zfill(8)}" for i in range(1, n_customers + 1)]
        
        # Demographics
        genders = np.random.choice(['Male', 'Female'], n_customers, p=[0.5, 0.5])
        regions = np.random.choice(['North', 'South', 'East', 'West'], n_customers, p=[0.25, 0.25, 0.25, 0.25])
        
        # Service details
        subscription_types = np.random.choice(['Basic', 'Premium', 'Enterprise'], n_customers, p=[0.4, 0.4, 0.2])
        payment_methods = np.random.choice(['Credit Card', 'Bank Transfer', 'PayPal', 'Check'], n_customers, p=[0.4, 0.3, 0.2, 0.1])
        
        # Contract and services
        contract_types = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers, p=[0.5, 0.3, 0.2])
        internet_services = np.random.choice(['DSL', 'Fiber Optic', 'No'], n_customers, p=[0.4, 0.4, 0.2])
        phone_services = np.random.choice(['Yes', 'No'], n_customers, p=[0.7, 0.3])
        
        # Financial metrics
        monthly_charges = np.random.exponential(60, n_customers)
        monthly_charges = np.clip(monthly_charges, 20, 200)  # Clip to realistic range
        
        # Tenure (days)
        tenure_days = np.random.exponential(730, n_customers)  # ~2 years average
        tenure_days = np.clip(tenure_days, 1, 3650)  # 1 day to 10 years
        
        # Calculate total charges (with some noise)
        total_charges = monthly_charges * (tenure_days / 30) + np.random.normal(0, 50, n_customers)
        total_charges = np.maximum(total_charges, monthly_charges)  # Ensure total >= monthly
        
        # Number of services
        number_of_services = np.random.poisson(2, n_customers) + 1
        number_of_services = np.clip(number_of_services, 1, 10)
        
        # Number of dependents
        number_of_dependents = np.random.poisson(0.5, n_customers)
        number_of_dependents = np.clip(number_of_dependents, 0, 5)
        
        # Create churn target (higher probability for certain conditions)
        churn_probability = np.zeros(n_customers)
        
        # Higher churn for month-to-month contracts
        month_to_month_idx = contract_types == 'Month-to-Month'
        churn_probability[month_to_month_idx] += 0.3
        
        # Higher churn for high monthly charges
        high_charges_idx = monthly_charges > 100
        churn_probability[high_charges_idx] += 0.2
        
        # Lower churn for longer tenure
        long_tenure_idx = tenure_days > 1000
        churn_probability[long_tenure_idx] -= 0.2
        
        # Lower churn for more services
        many_services_idx = number_of_services > 3
        churn_probability[many_services_idx] -= 0.15
        
        # Clip probabilities to [0, 1]
        churn_probability = np.clip(churn_probability, 0.05, 0.95)
        
        # Generate churn outcomes
        churn = np.random.binomial(1, churn_probability, n_customers)
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'gender': genders,
            'region': regions,
            'subscription_type': subscription_types,
            'payment_method': payment_methods,
            'contract_type': contract_types,
            'internet_service': internet_services,
            'phone_service': phone_services,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'tenure_days': tenure_days,
            'number_of_services': number_of_services,
            'number_of_dependents': number_of_dependents,
            'churn': churn
        })
        
        # Add some missing values for realism (5% null rate)
        for col in ['monthly_charges', 'total_charges', 'tenure_days']:
            missing_count = int(0.05 * len(df))
            missing_indices = np.random.choice(df.index, missing_count, replace=False)
            df.loc[missing_indices, col] = np.nan
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate data schema and quality"""
        required_columns = [
            'customer_id', 'gender', 'region', 'subscription_type',
            'payment_method', 'contract_type', 'monthly_charges',
            'total_charges', 'tenure_days', 'churn'
        ]
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataIngestionError(f"Missing required columns: {missing_columns}")
        
        # Check customer_id format
        if not df['customer_id'].str.match(r'^CUST_\d{8}$').all():
            raise DataIngestionError("Customer ID format validation failed")
        
        # Check for reasonable value ranges
        if (df['monthly_charges'] < 0).any() or (df['monthly_charges'] > 1000).any():
            raise DataIngestionError("Monthly charges out of expected range")
        
        if (df['tenure_days'] < 0).any():
            raise DataIngestionError("Negative tenure days found")
        
        # Check churn is binary
        if not set(df['churn'].unique()).issubset({0, 1}):
            raise DataIngestionError("Churn column must be binary (0 or 1)")
        
        return True
    
    def _calculate_metrics(self, df: pd.DataFrame, start_time: datetime) -> IngestionMetrics:
        """Calculate ingestion metrics"""
        end_time = datetime.now()
        ingestion_time = (end_time - start_time).total_seconds()
        
        # Calculate data freshness (assuming data is from yesterday)
        data_freshness_hours = 24  # Synthetic data is always "fresh"
        
        # Source reliability (mock value based on successful ingestion)
        source_reliability = 0.99 if len(df) > 0 else 0.0
        
        return IngestionMetrics(
            records_ingested=len(df),
            records_failed=0,  # No failures in this implementation
            ingestion_time_seconds=ingestion_time,
            data_freshness_hours=data_freshness_hours,
            source_reliability=source_reliability
        )
    
    def ingest(self) -> IngestionMetrics:
        """Main ingestion method"""
        start_time = datetime.now()
        structlog.info("Starting data ingestion", source=self.source_url)
        
        try:
            # For demonstration, generate synthetic data
            # In production, this would download from S3, API, etc.
            if self.sample_size:
                df = self._generate_synthetic_data(self.sample_size)
            else:
                df = self._generate_synthetic_data(100000)  # Default size
            
            # Validate schema
            self._validate_schema(df)
            
            # Save to output path
            df.to_csv(self.output_path, index=False)
            
            # Calculate metrics
            metrics = self._calculate_metrics(df, start_time)
            
            # Save metrics
            metrics_path = self.output_path.parent / "ingest_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    "records_ingested": metrics.records_ingested,
                    "records_failed": metrics.records_failed,
                    "ingestion_time_seconds": metrics.ingestion_time_seconds,
                    "data_freshness_hours": metrics.data_freshness_hours,
                    "source_reliability": metrics.source_reliability,
                    "timestamp": start_time.isoformat()
                }, f, indent=2)
            
            structlog.info(
                "Data ingestion completed successfully",
                records_ingested=metrics.records_ingested,
                output_path=str(self.output_path),
                ingestion_time_seconds=metrics.ingestion_time_seconds
            )
            
            return metrics
            
        except Exception as e:
            structlog.error("Data ingestion failed", error=str(e), source=self.source_url)
            raise DataIngestionError(f"Ingestion failed: {str(e)}")


def main():
    """Main entry point for DVC pipeline"""
    import sys
    
    # Default configuration
    source_url = "s3://customer-data-lake/churn/"
    output_path = "data/raw/customer_data.csv"
    sample_size = None
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    
    # Create and run ingestor
    ingestor = DataIngestor(
        source_url=source_url,
        output_path=output_path,
        sample_size=sample_size
    )
    
    try:
        metrics = ingestor.ingest()
        print(f"✅ Ingestion successful: {metrics.records_ingested} records")
        return 0
    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())