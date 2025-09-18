"""
Unit Tests for Monitoring Service
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.monitoring_service import MonitoringService, PipelineEvent
from services.bigquery_service import BigQueryService
from services.remediation_service import RemediationService
from utils.config import Config

class TestMonitoringService:
    """Test suite for MonitoringService"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return Config()
    
    @pytest.fixture
    def bigquery_service(self, config):
        """Create a test BigQuery service"""
        # Use in-memory database for testing
        service = BigQueryService(config)
        service.initialize_tables()
        return service
    
    @pytest.fixture
    def remediation_service(self, config):
        """Create a test remediation service"""
        return RemediationService(config)
    
    @pytest.fixture
    def monitoring_service(self, config):
        """Create a test monitoring service"""
        return MonitoringService(config)
    
    def test_service_initialization(self, monitoring_service):
        """Test monitoring service initialization"""
        assert monitoring_service is not None
        assert monitoring_service.error_classifier is not None
        assert monitoring_service.data_generator is not None
        assert monitoring_service.is_monitoring is False
        assert monitoring_service.processed_events == 0
        assert monitoring_service.detected_errors == 0
        assert monitoring_service.healing_attempts == 0
        assert isinstance(monitoring_service.event_queue, list)
    
    def test_service_dependencies(self, monitoring_service, bigquery_service, remediation_service):
        """Test setting service dependencies"""
        monitoring_service.set_services(bigquery_service, remediation_service)
        
        assert monitoring_service.bigquery_service is bigquery_service
        assert monitoring_service.remediation_service is remediation_service
    
    def test_start_stop_monitoring(self, monitoring_service):
        """Test starting and stopping monitoring service"""
        # Initially should not be monitoring
        assert monitoring_service.is_monitoring is False
        
        # Start monitoring
        monitoring_service.start_monitoring()
        time.sleep(0.1)  # Allow thread to start
        assert monitoring_service.is_monitoring is True
        assert monitoring_service.monitoring_thread is not None
        assert monitoring_service.monitoring_thread.is_alive()
        
        # Stop monitoring
        monitoring_service.stop_monitoring()
        assert monitoring_service.is_monitoring is False
        
        # Thread should terminate
        if monitoring_service.monitoring_thread:
            monitoring_service.monitoring_thread.join(timeout=1)
            assert not monitoring_service.monitoring_thread.is_alive()
    
    def test_event_injection(self, monitoring_service):
        """Test manual event injection"""
        initial_queue_size = len(monitoring_service.event_queue)
        
        test_event = {
            'job_id': 'test_job_123',
            'job_name': 'test_pipeline',
            'job_type': 'dataflow',
            'status': 'failed',
            'timestamp': datetime.now(),
            'error_message': 'Test error message'
        }
        
        monitoring_service.inject_event(test_event)
        
        assert len(monitoring_service.event_queue) == initial_queue_size + 1
    
    def test_event_processing(self, monitoring_service, bigquery_service, remediation_service):
        """Test event processing functionality"""
        # Set up services
        monitoring_service.set_services(bigquery_service, remediation_service)
        
        # Create test event
        test_event = PipelineEvent(
            job_id='test_job_456',
            job_name='test_processing_job',
            job_type='composer_dag',
            status='completed',
            timestamp=datetime.now()
        )
        
        initial_processed = monitoring_service.processed_events
        
        # Process event
        monitoring_service._process_event(test_event)
        
        assert monitoring_service.processed_events == initial_processed + 1
        
        # Check that job was stored in database
        stored_job = bigquery_service.get_job('test_job_456')
        assert stored_job is not None
        assert stored_job['job_name'] == 'test_processing_job'
        assert stored_job['status'] == 'completed'
    
    def test_error_event_processing(self, monitoring_service, bigquery_service, remediation_service):
        """Test processing of error events"""
        # Set up services
        monitoring_service.set_services(bigquery_service, remediation_service)
        
        # Create error event
        error_event = PipelineEvent(
            job_id='error_job_789',
            job_name='failing_pipeline',
            job_type='dataflow',
            status='failed',
            timestamp=datetime.now(),
            error_message='Access denied for service account'
        )
        
        initial_errors = monitoring_service.detected_errors
        initial_healing = monitoring_service.healing_attempts
        
        # Process error event
        monitoring_service._process_event(error_event)
        
        # Should increment error count
        assert monitoring_service.detected_errors == initial_errors + 1
        
        # Check job was stored with error classification
        stored_job = bigquery_service.get_job('error_job_789')
        assert stored_job is not None
        assert stored_job['status'] == 'failed'
        assert stored_job['error_message'] == 'Access denied for service account'
        assert stored_job['error_type'] is not None
        assert stored_job['confidence_score'] is not None
        
        # If confidence was high enough, should trigger healing
        if stored_job['confidence_score'] >= 0.75:  # Default threshold
            assert monitoring_service.healing_attempts > initial_healing
    
    def test_monitoring_stats(self, monitoring_service):
        """Test monitoring statistics functionality"""
        stats = monitoring_service.get_monitoring_stats()
        
        required_fields = [
            'is_monitoring', 'processed_events', 'detected_errors',
            'healing_attempts', 'queue_size', 'error_detection_rate', 'healing_rate'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check data types
        assert isinstance(stats['is_monitoring'], bool)
        assert isinstance(stats['processed_events'], int)
        assert isinstance(stats['detected_errors'], int)
        assert isinstance(stats['healing_attempts'], int)
        assert isinstance(stats['queue_size'], int)
        assert isinstance(stats['error_detection_rate'], float)
        assert isinstance(stats['healing_rate'], float)
        
        # Check ranges
        assert stats['processed_events'] >= 0
        assert stats['detected_errors'] >= 0
        assert stats['healing_attempts'] >= 0
        assert stats['queue_size'] >= 0
        assert 0.0 <= stats['error_detection_rate'] <= 100.0
        assert 0.0 <= stats['healing_rate'] <= 100.0
    
    def test_event_callbacks(self, monitoring_service):
        """Test event callback functionality"""
        callback_events = []
        
        def test_callback(event):
            callback_events.append(event)
        
        # Add callback
        monitoring_service.add_event_callback(test_callback)
        
        # Create and process test event
        test_event = PipelineEvent(
            job_id='callback_test_job',
            job_name='callback_test',
            job_type='test_job',
            status='completed',
            timestamp=datetime.now()
        )
        
        monitoring_service._process_event(test_event)
        
        # Check callback was called
        assert len(callback_events) == 1
        assert callback_events[0].job_id == 'callback_test_job'
    
    def test_simulate_pipeline_failure(self, monitoring_service):
        """Test pipeline failure simulation"""
        initial_queue_size = len(monitoring_service.event_queue)
        
        job_id = monitoring_service.simulate_pipeline_failure(
            job_type='dataflow',
            error_type='network_timeout'
        )
        
        # Should add event to queue
        assert len(monitoring_service.event_queue) == initial_queue_size + 1
        
        # Check job_id format
        assert job_id.startswith('job_')
        assert len(job_id) > 4
    
    def test_recent_events_retrieval(self, monitoring_service, bigquery_service):
        """Test retrieval of recent events"""
        # Set up service
        monitoring_service.bigquery_service = bigquery_service
        
        # Add some test jobs
        test_jobs = [
            {
                'job_id': f'recent_job_{i}',
                'job_name': f'recent_pipeline_{i}',
                'job_type': 'test_job',
                'status': 'completed' if i % 2 == 0 else 'failed',
                'error_message': 'Test error' if i % 2 == 1 else None
            }
            for i in range(5)
        ]
        
        for job in test_jobs:
            bigquery_service.insert_job(job)
        
        # Get recent events
        events = monitoring_service.get_recent_events(limit=10)
        
        assert isinstance(events, list)
        assert len(events) >= len(test_jobs)
        
        # Check event structure
        for event in events:
            required_fields = ['job_id', 'job_name', 'job_type', 'status', 'timestamp']
            for field in required_fields:
                assert field in event
    
    def test_confidence_threshold_filtering(self, monitoring_service, bigquery_service, remediation_service):
        """Test that low confidence errors don't trigger healing"""
        # Set up services
        monitoring_service.set_services(bigquery_service, remediation_service)
        
        # Mock a low confidence classification
        original_classify = monitoring_service.error_classifier.classify_error
        
        def mock_low_confidence(message):
            result = original_classify(message)
            result['confidence'] = 0.3  # Below default threshold of 0.75
            return result
        
        monitoring_service.error_classifier.classify_error = mock_low_confidence
        
        # Create error event
        low_confidence_event = PipelineEvent(
            job_id='low_confidence_job',
            job_name='low_confidence_test',
            job_type='test_job',
            status='failed',
            timestamp=datetime.now(),
            error_message='Ambiguous error message'
        )
        
        initial_healing = monitoring_service.healing_attempts
        
        # Process event
        monitoring_service._process_event(low_confidence_event)
        
        # Should not trigger healing due to low confidence
        assert monitoring_service.healing_attempts == initial_healing
        
        # Restore original method
        monitoring_service.error_classifier.classify_error = original_classify
    
    def test_concurrent_event_processing(self, monitoring_service, bigquery_service, remediation_service):
        """Test concurrent event processing"""
        # Set up services
        monitoring_service.set_services(bigquery_service, remediation_service)
        
        # Create multiple events
        events = []
        for i in range(5):
            event_data = {
                'job_id': f'concurrent_job_{i}',
                'job_name': f'concurrent_test_{i}',
                'job_type': 'test_job',
                'status': 'completed',
                'timestamp': datetime.now()
            }
            events.append(event_data)
        
        # Inject events concurrently
        threads = []
        for event_data in events:
            thread = threading.Thread(target=monitoring_service.inject_event, args=(event_data,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All events should be in queue
        assert len(monitoring_service.event_queue) >= len(events)
    
    def test_monitoring_loop_integration(self, monitoring_service, bigquery_service, remediation_service):
        """Test monitoring loop integration"""
        # Set up services
        monitoring_service.set_services(bigquery_service, remediation_service)
        
        initial_processed = monitoring_service.processed_events
        
        # Start monitoring for a short period
        monitoring_service.start_monitoring()
        time.sleep(2)  # Let it process some synthetic events
        monitoring_service.stop_monitoring()
        
        # Should have processed some events
        assert monitoring_service.processed_events > initial_processed
    
    @pytest.mark.parametrize("error_type,expected_in_message", [
        ("network_timeout", "timeout"),
        ("permission_denied", "denied"),
        ("schema_mismatch", "schema"),
        ("resource_exhaustion", "memory"),
        ("data_duplication", "duplicate")
    ])
    def test_error_type_simulation(self, monitoring_service, error_type, expected_in_message):
        """Test simulation of different error types"""
        job_id = monitoring_service.simulate_pipeline_failure(
            job_type='test_job',
            error_type=error_type
        )
        
        # Check that event was added to queue
        assert len(monitoring_service.event_queue) > 0
        
        # Find the event in queue
        event = None
        with monitoring_service.queue_lock:
            for e in monitoring_service.event_queue:
                if hasattr(e, 'job_id') and getattr(e, 'job_id', None) == job_id:
                    event = e
                    break
                elif isinstance(e, dict) and e.get('job_id') == job_id:
                    event = e
                    break
        
        # Event should contain expected error message content
        if event:
            error_message = event.get('error_message') if isinstance(event, dict) else getattr(event, 'error_message', '')
            assert expected_in_message.lower() in error_message.lower()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
