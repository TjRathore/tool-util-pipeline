"""
Real-time Monitoring Service
Handles pipeline monitoring, error detection, and event processing
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from dataclasses import dataclass
import uuid

from models.error_classifier import ErrorClassifier
from src.rag.rag_classifier import RAGErrorClassifier
from services.bigquery_service import BigQueryService
from services.remediation_service import RemediationService
from utils.data_generator import DataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineEvent:
    """Data class for pipeline events"""
    job_id: str
    job_name: str
    job_type: str
    status: str
    timestamp: datetime
    error_message: Optional[str] = None
    context: Optional[Dict] = None

class MonitoringService:
    """Real-time pipeline monitoring and error detection service"""
    
    def __init__(self, config):
        self.config = config
        # Initialize both traditional and RAG-enhanced classifiers
        self.traditional_classifier = ErrorClassifier()
        self.error_classifier = RAGErrorClassifier(
            traditional_classifier=self.traditional_classifier,
            config=config.get('rag', {})
        )
        
        # Feature flag for RAG enhancement
        self.use_rag_classification = config.get('use_rag_classification', True)
        
        logger.info("RAG-Enhanced Error Classifier initialized")
        self.bigquery_service = None
        self.remediation_service = None
        self.data_generator = DataGenerator()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.event_callbacks = []
        
        # Performance metrics
        self.processed_events = 0
        self.detected_errors = 0
        self.healing_attempts = 0
        
        # Event queue for real-time processing
        self.event_queue = []
        self.queue_lock = threading.Lock()
    
    def set_services(self, bigquery_service: BigQueryService, remediation_service: RemediationService):
        """Set dependent services"""
        self.bigquery_service = bigquery_service
        self.remediation_service = remediation_service
    
    def start_monitoring(self):
        """Start the monitoring service"""
        if self.is_monitoring:
            logger.warning("Monitoring service is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Monitoring service started")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop - simulates real-time event processing"""
        while self.is_monitoring:
            try:
                # Generate synthetic pipeline events
                events = self.data_generator.generate_pipeline_events(count=3)
                
                for event_data in events:
                    event = PipelineEvent(**event_data)
                    self._process_event(event)
                
                # Process queued events
                self._process_event_queue()
                
                # Wait before next cycle
                time.sleep(5)  # 5-second monitoring cycle
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _process_event(self, event: PipelineEvent):
        """Process a single pipeline event"""
        try:
            self.processed_events += 1
            
            # Store job information
            if self.bigquery_service:
                job_data = {
                    'job_id': event.job_id,
                    'job_name': event.job_name,
                    'job_type': event.job_type,
                    'status': event.status,
                    'error_message': event.error_message
                }
                
                self.bigquery_service.insert_job(job_data)
                
                # Log the event
                log_data = {
                    'job_id': event.job_id,
                    'level': 'ERROR' if event.status == 'failed' else 'INFO',
                    'message': event.error_message or f"Job {event.status}",
                    'context': event.context or {}
                }
                
                self.bigquery_service.insert_log(log_data)
            
            # Check for errors and trigger healing
            if event.status == 'failed' and event.error_message:
                self._handle_error_event(event)
            
            # Notify callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing event {event.job_id}: {e}")
    
    def _handle_error_event(self, event: PipelineEvent):
        """Handle error events - classify and trigger remediation"""
        try:
            self.detected_errors += 1
            
            # Classify the error using RAG-enhanced classifier
            if self.use_rag_classification:
                classification = self.error_classifier.classify_with_rag(
                    error_message=event.error_message,
                    context={
                        'job_id': event.job_id,
                        'job_name': event.job_name,
                        'job_type': event.job_type,
                        'timestamp': event.timestamp.isoformat(),
                        'pipeline_context': event.context
                    }
                )
                
                # Log RAG insights
                if classification.get('rag_enhanced'):
                    logger.info(f"RAG found {classification.get('similar_cases', 0)} similar cases")
                    if classification.get('recommended_strategy'):
                        logger.info(f"RAG recommends: {classification['recommended_strategy']}")
                    if classification.get('historical_success_rate'):
                        logger.info(f"Historical success rate: {classification['historical_success_rate']:.1%}")
                
            else:
                # Fallback to traditional classification
                classification = self.error_classifier.classify_error(event.error_message)
            
            # Extract error type from RAG or traditional classification
            error_type = classification.get('classification', classification.get('error_type', 'unknown'))
            confidence = classification.get('confidence', 0.0)
            
            logger.info(f"Error classified: {error_type} "
                       f"(confidence: {confidence:.3f})")
            
            # Update job with classification results
            if self.bigquery_service:
                updates = {
                    'error_type': error_type,
                    'confidence_score': confidence
                }
                self.bigquery_service.update_job(event.job_id, updates)
                
                # Update error pattern
                remediation_strategy = classification.get('recommended_strategy', 
                                                        classification.get('remediation_strategy', 'default'))
                pattern_data = {
                    'error_type': error_type,
                    'resolution_strategy': remediation_strategy
                }
                self.bigquery_service.update_error_pattern(
                    event.error_message[:100],  # Use first 100 chars as signature
                    pattern_data
                )
            
            # Create remediation recommendation for manual approval
            remediation_classification = {
                'error_type': error_type,
                'confidence': confidence,
                'remediation_strategy': classification.get('recommended_strategy', 
                                                        classification.get('remediation_strategy', 'default'))
            }
            self._create_recommendation(event, remediation_classification)
                    
        except Exception as e:
            logger.error(f"Error handling error event {event.job_id}: {e}")
    
    def _map_rag_strategy_to_predefined(self, rag_strategy: str, error_type: str) -> str:
        """Map RAG-recommended strategy to predefined strategy keys"""
        
        # Strategy mapping based on keywords and error types
        strategy_mapping = {
            # Permission/Authentication strategies
            'permission': 'restart_with_elevated_permissions',
            'role': 'restart_with_elevated_permissions', 
            'access denied': 'restart_with_elevated_permissions',
            'credentials': 'refresh_credentials_and_retry',
            'authentication': 'refresh_credentials_and_retry',
            
            # Resource strategies
            'resource': 'scale_compute_resources',
            'memory': 'scale_compute_resources',
            'cpu': 'scale_compute_resources',
            'machine type': 'scale_compute_resources',
            'quota': 'request_quota_increase',
            
            # Schema/Data strategies  
            'schema': 'update_schema_and_retry',
            'table': 'update_schema_and_retry',
            'column': 'update_schema_and_retry',
            'duplicate': 'run_deduplication_script',
            'deduplication': 'run_deduplication_script',
            
            # Network strategies
            'timeout': 'retry_with_exponential_backoff',
            'network': 'retry_with_exponential_backoff',
            'connection': 'retry_with_exponential_backoff',
            
            # Dependency strategies
            'dependency': 'install_dependencies_and_retry',
            'package': 'install_dependencies_and_retry',
            'missing': 'install_dependencies_and_retry',
            
            # Configuration strategies
            'configuration': 'update_configuration_and_retry',
            'config': 'update_configuration_and_retry',
            'parameter': 'update_configuration_and_retry',
            
            # Data quality strategies
            'validation': 'run_data_validation_and_cleanup',
            'cleanup': 'run_data_validation_and_cleanup',
            'corrupt': 'run_data_validation_and_cleanup'
        }
        
        # Convert to lowercase for matching
        rag_strategy_lower = rag_strategy.lower() if rag_strategy else ""
        
        # Try to find matching strategy based on keywords
        for keyword, predefined_strategy in strategy_mapping.items():
            if keyword in rag_strategy_lower:
                return predefined_strategy
        
        # Fallback based on error type
        error_type_fallbacks = {
            'permission_denied': 'restart_with_elevated_permissions',
            'authentication_failure': 'refresh_credentials_and_retry',
            'network_timeout': 'retry_with_exponential_backoff',
            'schema_mismatch': 'update_schema_and_retry',
            'resource_exhaustion': 'scale_compute_resources',
            'data_duplication': 'run_deduplication_script',
            'missing_dependency': 'install_dependencies_and_retry',
            'configuration_error': 'update_configuration_and_retry',
            'data_corruption': 'run_data_validation_and_cleanup',
            'quota_exceeded': 'request_quota_increase'
        }
        
        return error_type_fallbacks.get(error_type, 'retry_with_exponential_backoff')

    def _trigger_remediation(self, event: PipelineEvent, classification: Dict):
        """Trigger automated remediation"""
        try:
            self.healing_attempts += 1
            
            if self.remediation_service:
                # Map RAG strategy to predefined strategy
                rag_strategy = classification['remediation_strategy']
                predefined_strategy = self._map_rag_strategy_to_predefined(rag_strategy, classification['error_type'])
                
                logger.info(f"Mapped RAG strategy '{rag_strategy}' to predefined strategy '{predefined_strategy}'")
                
                # Start remediation with predefined strategy
                remediation_result = self.remediation_service.execute_remediation(
                    job_id=event.job_id,
                    strategy=predefined_strategy,
                    error_type=classification['error_type'],
                    confidence=classification['confidence']
                )
                
                # Update job with remediation results
                if self.bigquery_service and remediation_result:
                    healing_duration = remediation_result.get('duration', 0)
                    success = remediation_result.get('success', False)
                    
                    updates = {
                        'auto_healed': success,
                        'healing_duration': healing_duration,
                        'resolution': remediation_result.get('strategy'),
                        'status': 'completed' if success else 'failed'
                    }
                    
                    self.bigquery_service.update_job(event.job_id, updates)
                    
                    # Record healing action
                    action_data = {
                        'job_id': event.job_id,
                        'action_type': predefined_strategy,  # Use predefined strategy for database
                        'status': 'completed' if success else 'failed',
                        'completed_at': datetime.now().isoformat(),
                        'success': success
                    }
                    
                    self.bigquery_service.insert_healing_action(action_data)
                    
                    # Update error pattern with success feedback
                    pattern_data = {
                        'error_type': classification['error_type'],
                        'resolution_strategy': predefined_strategy,  # Use predefined strategy
                        'success': success
                    }
                    
                    self.bigquery_service.update_error_pattern(
                        event.error_message[:100],
                        pattern_data
                    )
                    
                    logger.info(f"Remediation {'successful' if success else 'failed'} "
                               f"for job {event.job_id} in {healing_duration:.2f}s")
                    
        except Exception as e:
            logger.error(f"Error triggering remediation for job {event.job_id}: {e}")
    
    def _create_recommendation(self, event: PipelineEvent, classification: Dict):
        """Create a remediation recommendation for manual approval"""
        try:
            # Map RAG strategy to predefined strategy
            rag_strategy = classification['remediation_strategy']
            predefined_strategy = self._map_rag_strategy_to_predefined(rag_strategy, classification['error_type'])
            
            logger.info(f"Creating recommendation for job {event.job_id}: {predefined_strategy}")
            
            # Generate unique recommendation ID
            recommendation_id = f"rec_{uuid.uuid4().hex[:12]}"
            
            # Create recommendation data
            recommendation_data = {
                'recommendation_id': recommendation_id,
                'job_id': event.job_id,
                'error_type': classification['error_type'],
                'rag_recommendation': rag_strategy,
                'mapped_strategy': predefined_strategy,
                'confidence_score': classification['confidence'],
                'status': 'pending'
            }
            
            # Save to database
            if self.bigquery_service:
                success = self.bigquery_service.insert_pending_recommendation(recommendation_data)
                if success:
                    logger.info(f"Recommendation {recommendation_id} created for job {event.job_id}")
                    
                    # Update job status to indicate recommendation is pending
                    job_updates = {
                        'status': 'recommendation_pending',
                        'resolution': f"Recommendation pending: {predefined_strategy}"
                    }
                    self.bigquery_service.update_job(event.job_id, job_updates)
                else:
                    logger.error(f"Failed to save recommendation for job {event.job_id}")
                    
        except Exception as e:
            logger.error(f"Error creating recommendation for job {event.job_id}: {e}")
    
    def execute_approved_recommendation(self, recommendation_id: str, approved_by: str = "user") -> Dict[str, Any]:
        """Execute an approved recommendation"""
        try:
            # Get recommendation details
            if not self.bigquery_service:
                return {'success': False, 'error': 'Database service not available'}
                
            recommendation = self.bigquery_service.get_recommendation(recommendation_id)
            if not recommendation:
                return {'success': False, 'error': 'Recommendation not found'}
                
            if recommendation['status'] != 'pending':
                return {'success': False, 'error': f'Recommendation status is {recommendation["status"]}, not pending'}
            
            # Update recommendation status to approved
            self.bigquery_service.update_recommendation_status(recommendation_id, {
                'status': 'approved',
                'approved_by': approved_by,
                'approved_at': datetime.now().isoformat()
            })
            
            # Execute the remediation
            if self.remediation_service:
                logger.info(f"Executing approved recommendation {recommendation_id}: {recommendation['mapped_strategy']}")
                
                remediation_result = self.remediation_service.execute_remediation(
                    job_id=recommendation['job_id'],
                    strategy=recommendation['mapped_strategy'],
                    error_type=recommendation['error_type'],
                    confidence=recommendation['confidence_score']
                )
                
                # Update recommendation with execution results
                execution_updates = {
                    'executed_at': datetime.now().isoformat(),
                    'execution_result': 'success' if remediation_result.get('success', False) else 'failed',
                    'status': 'completed'
                }
                self.bigquery_service.update_recommendation_status(recommendation_id, execution_updates)
                
                # Update job with remediation results
                if remediation_result:
                    healing_duration = remediation_result.get('duration', 0)
                    success = remediation_result.get('success', False)
                    
                    job_updates = {
                        'auto_healed': success,
                        'healing_duration': healing_duration,
                        'resolution': recommendation['mapped_strategy'],
                        'status': 'completed' if success else 'failed'
                    }
                    
                    self.bigquery_service.update_job(recommendation['job_id'], job_updates)
                    
                    logger.info(f"Manual remediation {'successful' if success else 'failed'} "
                               f"for job {recommendation['job_id']} in {healing_duration:.2f}s")
                    
                    return {
                        'success': True,
                        'remediation_result': remediation_result,
                        'message': f"Remediation {'completed successfully' if success else 'failed'}"
                    }
                else:
                    return {'success': False, 'error': 'Remediation service failed to execute'}
            else:
                return {'success': False, 'error': 'Remediation service not available'}
                
        except Exception as e:
            logger.error(f"Error executing approved recommendation {recommendation_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def reject_recommendation(self, recommendation_id: str, rejected_by: str = "user") -> Dict[str, Any]:
        """Reject a pending recommendation"""
        try:
            if not self.bigquery_service:
                return {'success': False, 'error': 'Database service not available'}
                
            recommendation = self.bigquery_service.get_recommendation(recommendation_id)
            if not recommendation:
                return {'success': False, 'error': 'Recommendation not found'}
                
            if recommendation['status'] != 'pending':
                return {'success': False, 'error': f'Recommendation status is {recommendation["status"]}, not pending'}
            
            # Update recommendation status to rejected
            self.bigquery_service.update_recommendation_status(recommendation_id, {
                'status': 'rejected',
                'approved_by': rejected_by,
                'approved_at': datetime.now().isoformat()
            })
            
            # Update job status
            self.bigquery_service.update_job(recommendation['job_id'], {
                'status': 'failed',
                'resolution': 'Remediation rejected by user'
            })
            
            logger.info(f"Recommendation {recommendation_id} rejected by {rejected_by}")
            return {'success': True, 'message': 'Recommendation rejected successfully'}
                
        except Exception as e:
            logger.error(f"Error rejecting recommendation {recommendation_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_event_queue(self):
        """Process events from the queue"""
        with self.queue_lock:
            while self.event_queue:
                event = self.event_queue.pop(0)
                self._process_event(event)
    
    def add_event_callback(self, callback: Callable[[PipelineEvent], None]):
        """Add a callback for event notifications"""
        self.event_callbacks.append(callback)
    
    def inject_event(self, event_data: Dict):
        """Manually inject an event for testing/simulation"""
        event = PipelineEvent(**event_data)
        
        with self.queue_lock:
            self.event_queue.append(event)
        
        logger.info(f"Event injected: {event.job_id}")
    
    def get_monitoring_stats(self) -> Dict[str, any]:
        """Get current monitoring statistics"""
        return {
            'is_monitoring': self.is_monitoring,
            'processed_events': self.processed_events,
            'detected_errors': self.detected_errors,
            'healing_attempts': self.healing_attempts,
            'queue_size': len(self.event_queue),
            'error_detection_rate': (self.detected_errors / max(self.processed_events, 1)) * 100,
            'healing_rate': (self.healing_attempts / max(self.detected_errors, 1)) * 100
        }
    
    def simulate_pipeline_failure(self, job_type: str = "dataflow", error_type: str = "network_timeout"):
        """Simulate a pipeline failure for testing"""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        error_messages = {
            'network_timeout': 'Connection timeout after 30 seconds',
            'permission_denied': 'Access denied for user service-account@project.iam.gserviceaccount.com',
            'schema_mismatch': 'Schema validation failed: column "user_id" not found in table',
            'resource_exhaustion': 'Out of memory error: cannot allocate 2.5GB',
            'data_duplication': 'Duplicate key violation: primary key constraint failed'
        }
        
        event_data = {
            'job_id': job_id,
            'job_name': f"pipeline_{job_type}_{datetime.now().strftime('%H%M%S')}",
            'job_type': job_type,
            'status': 'failed',
            'timestamp': datetime.now(),
            'error_message': error_messages.get(error_type, 'Unknown error occurred'),
            'context': {
                'simulated': True,
                'error_type': error_type,
                'severity': 'high'
            }
        }
        
        self.inject_event(event_data)
        return job_id
    
    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent pipeline events from database"""
        if not self.bigquery_service:
            return []
        
        jobs = self.bigquery_service.get_jobs(limit=limit)
        
        events = []
        for job in jobs:
            events.append({
                'job_id': job['job_id'],
                'job_name': job['job_name'],
                'job_type': job['job_type'],
                'status': job['status'],
                'timestamp': job['created_at'],
                'error_message': job['error_message'],
                'error_type': job['error_type'],
                'auto_healed': job['auto_healed'],
                'confidence_score': job['confidence_score']
            })
        
        return events
