"""
Configuration Management
Handles application configuration and environment variables
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration management"""
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            # GCP Configuration
            'gcp_project_id': os.getenv('GCP_PROJECT_ID', 'ai-pipeline-monitor'),
            'gcp_region': os.getenv('GCP_REGION', 'us-central1'),
            'google_application_credentials': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            
            # BigQuery Configuration
            'bigquery_dataset': os.getenv('BIGQUERY_DATASET', 'pipeline_monitoring'),
            'bigquery_location': os.getenv('BIGQUERY_LOCATION', 'US'),
            
            # Pub/Sub Configuration
            'pubsub_topic_events': os.getenv('PUBSUB_TOPIC_EVENTS', 'pipeline-events'),
            'pubsub_topic_errors': os.getenv('PUBSUB_TOPIC_ERRORS', 'error-notifications'),
            'pubsub_subscription_monitor': os.getenv('PUBSUB_SUBSCRIPTION_MONITOR', 'monitor-events'),
            
            # ML Model Configuration
            'model_confidence_threshold': float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.75')),
            'model_retrain_threshold': int(os.getenv('MODEL_RETRAIN_THRESHOLD', '100')),
            'model_path': os.getenv('MODEL_PATH', 'models/error_classifier.joblib'),
            
            # Monitoring Configuration
            'monitoring_interval': int(os.getenv('MONITORING_INTERVAL', '30')),  # seconds
            'max_retry_attempts': int(os.getenv('MAX_RETRY_ATTEMPTS', '3')),
            'healing_timeout': int(os.getenv('HEALING_TIMEOUT', '300')),  # seconds
            
            # Database Configuration
            'database_path': os.getenv('DATABASE_PATH', 'pipeline_monitor.db'),
            'database_backup_enabled': os.getenv('DATABASE_BACKUP_ENABLED', 'true').lower() == 'true',
            
            # Alerting Configuration
            'alerting_enabled': os.getenv('ALERTING_ENABLED', 'true').lower() == 'true',
            'alert_slack_webhook': os.getenv('ALERT_SLACK_WEBHOOK'),
            'alert_email_enabled': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
            'alert_email_smtp_server': os.getenv('ALERT_EMAIL_SMTP_SERVER'),
            'alert_email_from': os.getenv('ALERT_EMAIL_FROM'),
            'alert_email_to': os.getenv('ALERT_EMAIL_TO'),
            
            # Security Configuration
            'session_secret': os.getenv('SESSION_SECRET', 'default-secret-key'),
            'api_key_required': os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true',
            'api_key': os.getenv('API_KEY'),
            
            # Feature Flags
            'auto_healing_enabled': os.getenv('AUTO_HEALING_ENABLED', 'true').lower() == 'true',
            'ml_classification_enabled': os.getenv('ML_CLASSIFICATION_ENABLED', 'true').lower() == 'true',
            'real_time_monitoring_enabled': os.getenv('REAL_TIME_MONITORING_ENABLED', 'true').lower() == 'true',
            'feedback_loop_enabled': os.getenv('FEEDBACK_LOOP_ENABLED', 'true').lower() == 'true',
            
            # Development Configuration
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            'mock_gcp_services': os.getenv('MOCK_GCP_SERVICES', 'true').lower() == 'true',
            'generate_synthetic_data': os.getenv('GENERATE_SYNTHETIC_DATA', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            
            # Performance Configuration
            'max_concurrent_remediations': int(os.getenv('MAX_CONCURRENT_REMEDIATIONS', '5')),
            'event_processing_batch_size': int(os.getenv('EVENT_PROCESSING_BATCH_SIZE', '10')),
            'metrics_retention_days': int(os.getenv('METRICS_RETENTION_DAYS', '30')),
            
            # UI Configuration
            'dashboard_refresh_interval': int(os.getenv('DASHBOARD_REFRESH_INTERVAL', '30')),  # seconds
            'max_dashboard_events': int(os.getenv('MAX_DASHBOARD_EVENTS', '1000')),
            'chart_animation_enabled': os.getenv('CHART_ANIMATION_ENABLED', 'true').lower() == 'true'
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self.config.get(key, default)
    
    def get_gcp_project_id(self) -> str:
        """Get GCP project ID"""
        return self.config['gcp_project_id']
    
    def get_gcp_region(self) -> str:
        """Get GCP region"""
        return self.config['gcp_region']
    
    def get_bigquery_dataset(self) -> str:
        """Get BigQuery dataset name"""
        return self.config['bigquery_dataset']
    
    def get_confidence_threshold(self) -> float:
        """Get ML model confidence threshold"""
        return self.config['model_confidence_threshold']
    
    def get_max_retry_attempts(self) -> int:
        """Get maximum retry attempts for failed jobs"""
        return self.config['max_retry_attempts']
    
    def get_monitoring_interval(self) -> int:
        """Get monitoring interval in seconds"""
        return self.config['monitoring_interval']
    
    def is_auto_healing_enabled(self) -> bool:
        """Check if auto-healing is enabled"""
        return self.config['auto_healing_enabled']
    
    def is_ml_classification_enabled(self) -> bool:
        """Check if ML classification is enabled"""
        return self.config['ml_classification_enabled']
    
    def is_real_time_monitoring_enabled(self) -> bool:
        """Check if real-time monitoring is enabled"""
        return self.config['real_time_monitoring_enabled']
    
    def is_feedback_loop_enabled(self) -> bool:
        """Check if feedback loop is enabled"""
        return self.config['feedback_loop_enabled']
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.config['debug_mode']
    
    def is_mock_gcp_services(self) -> bool:
        """Check if mock GCP services should be used"""
        return self.config['mock_gcp_services']
    
    def get_database_path(self) -> str:
        """Get database file path"""
        return self.config['database_path']
    
    def get_model_path(self) -> str:
        """Get ML model file path"""
        return self.config['model_path']
    
    def get_dashboard_refresh_interval(self) -> int:
        """Get dashboard refresh interval in seconds"""
        return self.config['dashboard_refresh_interval']
    
    def get_max_dashboard_events(self) -> int:
        """Get maximum events to display in dashboard"""
        return self.config['max_dashboard_events']
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration values"""
        self.config.update(updates)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = []
        warnings = []
        
        # Check required GCP configuration
        if not self.config['gcp_project_id']:
            issues.append("GCP_PROJECT_ID is required")
        
        # Check confidence threshold range
        threshold = self.config['model_confidence_threshold']
        if not 0.0 <= threshold <= 1.0:
            issues.append(f"MODEL_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {threshold}")
        
        # Check retry attempts
        max_retries = self.config['max_retry_attempts']
        if max_retries < 1 or max_retries > 10:
            warnings.append(f"MAX_RETRY_ATTEMPTS of {max_retries} may be outside recommended range (1-10)")
        
        # Check monitoring interval
        interval = self.config['monitoring_interval']
        if interval < 5 or interval > 300:
            warnings.append(f"MONITORING_INTERVAL of {interval}s may be outside recommended range (5-300)")
        
        # Check if using production credentials in debug mode
        if (self.config['debug_mode'] and 
            self.config['google_application_credentials'] and 
            not self.config['mock_gcp_services']):
            warnings.append("Using real GCP credentials in debug mode. Consider using mock services.")
        
        # Check alerting configuration
        if self.config['alerting_enabled'] and not self.config['alert_slack_webhook']:
            warnings.append("Alerting is enabled but no Slack webhook URL provided")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get a summary of the current environment configuration"""
        return {
            'environment': 'development' if self.config['debug_mode'] else 'production',
            'gcp_project': self.config['gcp_project_id'],
            'region': self.config['gcp_region'],
            'mock_services': self.config['mock_gcp_services'],
            'auto_healing': self.config['auto_healing_enabled'],
            'ml_classification': self.config['ml_classification_enabled'],
            'real_time_monitoring': self.config['real_time_monitoring_enabled'],
            'confidence_threshold': self.config['model_confidence_threshold'],
            'max_retries': self.config['max_retry_attempts'],
            'monitoring_interval': self.config['monitoring_interval']
        }
