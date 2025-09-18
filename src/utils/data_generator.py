"""
Synthetic Data Generator
Generates realistic pipeline events and error scenarios for testing and simulation
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

class DataGenerator:
    """Generates synthetic data for testing the monitoring system"""
    
    def __init__(self):
        self.job_types = [
            'dataflow_streaming',
            'dataflow_batch',
            'composer_dag',
            'dataproc_job',
            'bigquery_load',
            'cloud_function',
            'vertex_pipeline'
        ]
        
        self.job_name_prefixes = [
            'user_analytics',
            'sales_etl',
            'inventory_sync',
            'recommendation_ml',
            'fraud_detection',
            'reporting_batch',
            'real_time_alerts',
            'data_quality_check',
            'customer_segmentation',
            'financial_processing'
        ]
        
        self.error_scenarios = {
            'permission_denied': [
                'Access denied for service account data-pipeline@project.iam.gserviceaccount.com',
                'Permission denied: User does not have bigquery.datasets.create access',
                'IAM policy violation: Insufficient privileges to access Cloud Storage bucket',
                'Service account lacks required roles for Dataflow job execution',
                'Access forbidden for user: Missing storage.objects.write permission'
            ],
            'network_timeout': [
                'Connection timeout after 30 seconds to BigQuery API',
                'Network timeout occurred while connecting to Cloud SQL instance',
                'Request timed out: Unable to reach Pub/Sub endpoint',
                'Socket timeout exception during data transfer to Cloud Storage',
                'Connection reset by peer: Network unreachable error'
            ],
            'schema_mismatch': [
                'Schema validation failed: Column "user_id" not found in target table',
                'Data type mismatch: Expected INTEGER, got STRING for field "amount"',
                'Table schema incompatibility: Missing required field "timestamp"',
                'Column count mismatch: Source has 15 columns, target expects 12',
                'Schema evolution error: Cannot add non-nullable column without default value'
            ],
            'resource_exhaustion': [
                'Out of memory error: Cannot allocate 2.5GB for worker process',
                'CPU limit exceeded: Job requires more than allocated 4 vCPUs',
                'Disk space insufficient: Available 500MB, required 2GB',
                'Memory allocation failed: Heap space exhausted during data processing',
                'Worker nodes unavailable: All instances in zone us-central1-a are full'
            ],
            'data_duplication': [
                'Duplicate key violation: Primary key constraint failed on user_events table',
                'Unique constraint violation: Duplicate entry for composite key (date, user_id)',
                'Integrity constraint violation: Duplicate records detected in source data',
                'Primary key conflict: Record with ID 12345 already exists',
                'Data uniqueness check failed: Found 2,456 duplicate rows in batch'
            ],
            'missing_dependency': [
                'Module not found: No module named "apache_beam.transforms.custom"',
                'Package import failed: pandas version 1.5.0 required, found 1.3.0',
                'Dependency not installed: Missing google-cloud-bigquery library',
                'Library not available: tensorflow 2.8.0 not found in environment',
                'Required package missing: Unable to import pyarrow for Parquet processing'
            ],
            'configuration_error': [
                'Configuration file not found: /config/pipeline.yaml does not exist',
                'Invalid configuration parameter: batch_size must be positive integer',
                'Missing required environment variable: BIGQUERY_DATASET not set',
                'Configuration syntax error: Invalid YAML format in line 23',
                'Parameter validation failed: max_workers cannot exceed 100'
            ],
            'quota_exceeded': [
                'BigQuery quota exceeded: Query limit of 1000 queries/day reached',
                'API rate limit exceeded: 100 requests per 100 seconds to Dataflow API',
                'Storage quota exhausted: Bucket has reached 5TB limit',
                'Compute Engine quota exceeded: Cannot create more than 24 instances',
                'Cloud Function invocation limit reached: 1M invocations per month'
            ],
            'authentication_failure': [
                'Authentication failed: Invalid service account key file',
                'Token expired: OAuth2 access token is no longer valid',
                'Unauthorized access attempt: API key authentication failed',
                'Service account authentication error: Key file corrupted or invalid',
                'Login credentials invalid: Unable to authenticate with provided credentials'
            ],
            'data_corruption': [
                'Data integrity check failed: Checksum mismatch in file batch_001.parquet',
                'Corrupt data detected: Invalid UTF-8 encoding in CSV file',
                'File corruption detected: Unexpected EOF while reading Avro file',
                'Data validation failed: 15% of records have null values in required fields',
                'Checksum validation failed: Expected SHA256 does not match actual'
            ]
        }
        
        self.success_scenarios = [
            'Job completed successfully in {duration:.1f} minutes',
            'Data processing completed: {records} records processed successfully',
            'Pipeline execution finished: All {tasks} tasks completed without errors',
            'Batch job successful: {files} files processed and loaded to BigQuery',
            'Streaming job running: Processing {rate} events per second'
        ]
    
    def generate_pipeline_events(self, count: int = 10, 
                                failure_rate: float = 0.15,
                                time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Generate synthetic pipeline events
        
        Args:
            count: Number of events to generate
            failure_rate: Probability of generating failed events (0.0 to 1.0)
            time_range_hours: Time range for event timestamps
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        for _ in range(count):
            # Generate random timestamp within time range
            hours_ago = random.uniform(0, time_range_hours)
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            
            # Determine if this should be a failure
            is_failure = random.random() < failure_rate
            
            # Generate job details
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            job_type = random.choice(self.job_types)
            job_name_prefix = random.choice(self.job_name_prefixes)
            job_name = f"{job_name_prefix}_{job_type}_{timestamp.strftime('%Y%m%d_%H%M')}"
            
            # Generate event
            event = {
                'job_id': job_id,
                'job_name': job_name,
                'job_type': job_type,
                'status': 'failed' if is_failure else random.choice(['completed', 'running']),
                'timestamp': timestamp,
                'context': {
                    'region': random.choice(['us-central1', 'us-east1', 'europe-west1']),
                    'environment': random.choice(['production', 'staging', 'development']),
                    'priority': random.choice(['high', 'medium', 'low']),
                    'estimated_duration': random.randint(5, 120)  # minutes
                }
            }
            
            # Add error details for failed jobs
            if is_failure:
                error_type = random.choice(list(self.error_scenarios.keys()))
                error_message = random.choice(self.error_scenarios[error_type])
                
                event['error_message'] = error_message
                event['context']['error_category'] = error_type
                event['context']['severity'] = random.choice(['critical', 'high', 'medium'])
            else:
                # Add success details
                if event['status'] == 'completed':
                    duration = random.uniform(2, 60)
                    records = random.randint(1000, 1000000)
                    tasks = random.randint(3, 15)
                    files = random.randint(1, 50)
                    rate = random.randint(100, 5000)
                    
                    success_message = random.choice(self.success_scenarios).format(
                        duration=duration,
                        records=records,
                        tasks=tasks,
                        files=files,
                        rate=rate
                    )
                    
                    event['error_message'] = success_message
                    event['context']['duration_minutes'] = duration
                    event['context']['records_processed'] = records
            
            events.append(event)
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return events
    
    def generate_error_batch(self, error_type: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generate a batch of errors of a specific type"""
        if error_type not in self.error_scenarios:
            raise ValueError(f"Unknown error type: {error_type}")
        
        events = []
        error_messages = self.error_scenarios[error_type]
        
        for i in range(count):
            job_id = f"job_{error_type}_{i+1}_{uuid.uuid4().hex[:8]}"
            job_name = f"test_{error_type}_{i+1}"
            
            event = {
                'job_id': job_id,
                'job_name': job_name,
                'job_type': random.choice(self.job_types),
                'status': 'failed',
                'timestamp': datetime.now() - timedelta(minutes=random.randint(1, 60)),
                'error_message': random.choice(error_messages),
                'context': {
                    'simulated': True,
                    'error_type': error_type,
                    'batch_id': f"batch_{error_type}_{count}",
                    'severity': 'high'
                }
            }
            
            events.append(event)
        
        return events
    
    def generate_ml_training_data(self, samples_per_category: int = 100) -> List[Dict[str, Any]]:
        """Generate labeled training data for ML model"""
        training_data = []
        
        for error_type, messages in self.error_scenarios.items():
            for _ in range(samples_per_category):
                # Use base message or create variation
                base_message = random.choice(messages)
                
                # Add random variations
                variations = [
                    f"Error: {base_message}",
                    f"FAILED: {base_message} (retry 1/3)",
                    f"Pipeline failure: {base_message} at step {random.randint(1, 10)}",
                    f"{base_message} in job {uuid.uuid4().hex[:8]}",
                    f"Worker node error: {base_message}"
                ]
                
                message = random.choice([base_message] + variations)
                
                training_data.append({
                    'error_message': message,
                    'error_type': error_type,
                    'confidence': 1.0,  # Ground truth
                    'source': 'synthetic_training_data'
                })
        
        # Shuffle the data
        random.shuffle(training_data)
        
        return training_data
    
    def generate_remediation_feedback(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate synthetic remediation feedback data"""
        feedback_data = []
        
        strategies = [
            'retry_with_exponential_backoff',
            'restart_with_elevated_permissions', 
            'update_schema_and_retry',
            'scale_compute_resources',
            'run_deduplication_script',
            'install_dependencies_and_retry',
            'update_configuration_and_retry',
            'request_quota_increase',
            'refresh_credentials_and_retry',
            'run_data_validation_and_cleanup'
        ]
        
        for _ in range(count):
            error_type = random.choice(list(self.error_scenarios.keys()))
            strategy = random.choice(strategies)
            
            # Some strategies work better for certain error types
            success_probability = self._get_strategy_success_rate(error_type, strategy)
            success = random.random() < success_probability
            
            feedback = {
                'job_id': f"feedback_job_{uuid.uuid4().hex[:8]}",
                'error_type': error_type,
                'error_message': random.choice(self.error_scenarios[error_type]),
                'strategy': strategy,
                'success': success,
                'duration': random.uniform(30, 300),  # seconds
                'confidence': random.uniform(0.6, 0.95),
                'timestamp': datetime.now() - timedelta(
                    hours=random.uniform(0, 168)  # Last week
                ),
                'manual_override': random.random() < 0.1  # 10% manual overrides
            }
            
            feedback_data.append(feedback)
        
        return feedback_data
    
    def _get_strategy_success_rate(self, error_type: str, strategy: str) -> float:
        """Get realistic success rate for error type and strategy combination"""
        
        # Define strategy effectiveness for different error types
        effectiveness_map = {
            'permission_denied': {
                'restart_with_elevated_permissions': 0.9,
                'refresh_credentials_and_retry': 0.85,
                'retry_with_exponential_backoff': 0.3
            },
            'network_timeout': {
                'retry_with_exponential_backoff': 0.8,
                'scale_compute_resources': 0.4,
                'restart_with_elevated_permissions': 0.2
            },
            'schema_mismatch': {
                'update_schema_and_retry': 0.95,
                'run_data_validation_and_cleanup': 0.7,
                'retry_with_exponential_backoff': 0.1
            },
            'resource_exhaustion': {
                'scale_compute_resources': 0.9,
                'restart_with_elevated_permissions': 0.6,
                'retry_with_exponential_backoff': 0.3
            },
            'data_duplication': {
                'run_deduplication_script': 0.95,
                'run_data_validation_and_cleanup': 0.8,
                'update_schema_and_retry': 0.4
            },
            'missing_dependency': {
                'install_dependencies_and_retry': 0.9,
                'update_configuration_and_retry': 0.7,
                'restart_with_elevated_permissions': 0.5
            },
            'configuration_error': {
                'update_configuration_and_retry': 0.9,
                'restart_with_elevated_permissions': 0.6,
                'retry_with_exponential_backoff': 0.4
            },
            'quota_exceeded': {
                'request_quota_increase': 0.8,
                'scale_compute_resources': 0.6,
                'retry_with_exponential_backoff': 0.2
            },
            'authentication_failure': {
                'refresh_credentials_and_retry': 0.9,
                'restart_with_elevated_permissions': 0.7,
                'update_configuration_and_retry': 0.5
            },
            'data_corruption': {
                'run_data_validation_and_cleanup': 0.9,
                'run_deduplication_script': 0.6,
                'update_schema_and_retry': 0.4
            }
        }
        
        if error_type in effectiveness_map and strategy in effectiveness_map[error_type]:
            return effectiveness_map[error_type][strategy]
        
        # Default success rate for unmapped combinations
        return 0.5
    
    def generate_realistic_pipeline_scenario(self, scenario_type: str = 'mixed') -> List[Dict[str, Any]]:
        """Generate a realistic pipeline scenario with related events"""
        
        if scenario_type == 'cascade_failure':
            return self._generate_cascade_failure_scenario()
        elif scenario_type == 'schema_migration':
            return self._generate_schema_migration_scenario()
        elif scenario_type == 'resource_spike':
            return self._generate_resource_spike_scenario()
        elif scenario_type == 'dependency_update':
            return self._generate_dependency_update_scenario()
        else:
            # Mixed scenario
            scenarios = [
                self._generate_cascade_failure_scenario(),
                self._generate_schema_migration_scenario(),
                self._generate_resource_spike_scenario()
            ]
            return random.choice(scenarios)
    
    def _generate_cascade_failure_scenario(self) -> List[Dict[str, Any]]:
        """Generate a cascade failure scenario"""
        events = []
        base_time = datetime.now() - timedelta(hours=2)
        
        # Initial failure triggers cascade
        pipeline_id = f"cascade_{uuid.uuid4().hex[:8]}"
        
        # Stage 1: Network timeout in data ingestion
        events.append({
            'job_id': f"{pipeline_id}_ingestion",
            'job_name': f"data_ingestion_{pipeline_id}",
            'job_type': 'dataflow_streaming',
            'status': 'failed',
            'timestamp': base_time,
            'error_message': 'Connection timeout after 30 seconds to source database',
            'context': {'stage': 'ingestion', 'cascade_id': pipeline_id}
        })
        
        # Stage 2: Dependent transformation jobs fail
        for i in range(3):
            events.append({
                'job_id': f"{pipeline_id}_transform_{i}",
                'job_name': f"data_transform_{i}_{pipeline_id}",
                'job_type': 'dataflow_batch',
                'status': 'failed',
                'timestamp': base_time + timedelta(minutes=10 + i*5),
                'error_message': f'Missing input data: Expected file transform_input_{i}.parquet not found',
                'context': {'stage': 'transformation', 'cascade_id': pipeline_id, 'depends_on': f"{pipeline_id}_ingestion"}
            })
        
        # Stage 3: Final aggregation fails
        events.append({
            'job_id': f"{pipeline_id}_aggregation",
            'job_name': f"daily_aggregation_{pipeline_id}",
            'job_type': 'bigquery_load',
            'status': 'failed',
            'timestamp': base_time + timedelta(minutes=30),
            'error_message': 'Schema validation failed: Required columns missing from source tables',
            'context': {'stage': 'aggregation', 'cascade_id': pipeline_id}
        })
        
        return events
    
    def _generate_schema_migration_scenario(self) -> List[Dict[str, Any]]:
        """Generate a schema migration scenario"""
        events = []
        base_time = datetime.now() - timedelta(hours=1)
        migration_id = f"migration_{uuid.uuid4().hex[:8]}"
        
        # Multiple jobs fail due to schema changes
        tables = ['users', 'orders', 'products', 'analytics']
        
        for i, table in enumerate(tables):
            events.append({
                'job_id': f"{migration_id}_{table}",
                'job_name': f"etl_{table}_{migration_id}",
                'job_type': 'composer_dag',
                'status': 'failed',
                'timestamp': base_time + timedelta(minutes=i*15),
                'error_message': f'Column "updated_at" not found in table {table}. Schema migration may be incomplete.',
                'context': {
                    'table': table,
                    'migration_id': migration_id,
                    'schema_version': 'v2.1.0'
                }
            })
        
        return events
    
    def _generate_resource_spike_scenario(self) -> List[Dict[str, Any]]:
        """Generate a resource exhaustion scenario during peak load"""
        events = []
        base_time = datetime.now() - timedelta(minutes=30)
        spike_id = f"spike_{uuid.uuid4().hex[:8]}"
        
        # Multiple jobs fail due to resource exhaustion
        for i in range(5):
            events.append({
                'job_id': f"{spike_id}_worker_{i}",
                'job_name': f"peak_processing_worker_{i}",
                'job_type': 'dataproc_job',
                'status': 'failed',
                'timestamp': base_time + timedelta(minutes=i*3),
                'error_message': f'Out of memory error: Cannot allocate {2.0 + i*0.5:.1f}GB for Spark executor',
                'context': {
                    'worker_id': i,
                    'spike_id': spike_id,
                    'peak_load': True,
                    'cpu_usage': random.randint(85, 98)
                }
            })
        
        return events
    
    def _generate_dependency_update_scenario(self) -> List[Dict[str, Any]]:
        """Generate a scenario where dependency updates break pipelines"""
        events = []
        base_time = datetime.now() - timedelta(hours=6)
        update_id = f"update_{uuid.uuid4().hex[:8]}"
        
        # Jobs fail after dependency update
        libraries = ['pandas', 'apache-beam', 'google-cloud-bigquery', 'tensorflow']
        
        for i, library in enumerate(libraries):
            events.append({
                'job_id': f"{update_id}_{library.replace('-', '_')}",
                'job_name': f"ml_pipeline_{library.replace('-', '_')}",
                'job_type': 'vertex_pipeline',
                'status': 'failed',
                'timestamp': base_time + timedelta(hours=i),
                'error_message': f'ImportError: cannot import name "deprecated_function" from {library}. Library updated to incompatible version.',
                'context': {
                    'library': library,
                    'update_id': update_id,
                    'old_version': f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
                    'new_version': f"{random.randint(2, 4)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
                }
            })
        
        return events
