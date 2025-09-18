"""
Unit Tests for Remediation Service
"""

import pytest
import time
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.remediation_service import RemediationService
from utils.config import Config

class TestRemediationService:
    """Test suite for RemediationService"""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration"""
        return Config()
    
    @pytest.fixture
    def remediation_service(self, config):
        """Create a test remediation service"""
        return RemediationService(config)
    
    def test_service_initialization(self, remediation_service):
        """Test remediation service initialization"""
        assert remediation_service is not None
        assert isinstance(remediation_service.strategies, dict)
        assert len(remediation_service.strategies) > 0
        assert isinstance(remediation_service.active_remediations, dict)
        assert isinstance(remediation_service.remediation_history, list)
        
        # Check that all required strategies are present
        required_strategies = [
            'retry_with_exponential_backoff',
            'restart_with_elevated_permissions',
            'update_schema_and_retry',
            'scale_compute_resources',
            'run_deduplication_script',
            'escalate_to_human'
        ]
        
        for strategy in required_strategies:
            assert strategy in remediation_service.strategies
    
    def test_strategy_configuration(self, remediation_service):
        """Test strategy configuration structure"""
        for strategy_name, config in remediation_service.strategies.items():
            # Each strategy should have required configuration
            assert 'max_attempts' in config
            assert 'base_delay' in config
            assert 'success_rate' in config
            
            # Check data types and ranges
            assert isinstance(config['max_attempts'], int)
            assert config['max_attempts'] > 0
            assert isinstance(config['base_delay'], (int, float))
            assert config['base_delay'] >= 0
            assert isinstance(config['success_rate'], float)
            assert 0.0 <= config['success_rate'] <= 1.0
    
    def test_execute_remediation_success(self, remediation_service):
        """Test successful remediation execution"""
        # Force success for testing by temporarily modifying success rate
        strategy = 'retry_with_exponential_backoff'
        original_rate = remediation_service.strategies[strategy]['success_rate']
        remediation_service.strategies[strategy]['success_rate'] = 1.0  # Force success
        
        result = remediation_service.execute_remediation(
            job_id='test_job_success',
            strategy=strategy,
            error_type='network_timeout',
            confidence=0.95
        )
        
        # Restore original success rate
        remediation_service.strategies[strategy]['success_rate'] = original_rate
        
        # Check result structure
        assert isinstance(result, dict)
        required_fields = ['success', 'duration', 'strategy', 'attempts', 'details', 'timestamp']
        for field in required_fields:
            assert field in result
        
        # Check success case
        assert result['success'] is True
        assert result['strategy'] == strategy
        assert result['attempts'] >= 1
        assert isinstance(result['duration'], float)
        assert result['duration'] > 0
        
        # Check remediation history
        assert len(remediation_service.remediation_history) > 0
        last_record = remediation_service.remediation_history[-1]
        assert last_record['job_id'] == 'test_job_success'
        assert last_record['success'] is True
    
    def test_execute_remediation_failure(self, remediation_service):
        """Test failed remediation execution"""
        # Force failure for testing
        strategy = 'retry_with_exponential_backoff'
        original_rate = remediation_service.strategies[strategy]['success_rate']
        remediation_service.strategies[strategy]['success_rate'] = 0.0  # Force failure
        
        result = remediation_service.execute_remediation(
            job_id='test_job_failure',
            strategy=strategy,
            error_type='network_timeout',
            confidence=0.85
        )
        
        # Restore original success rate
        remediation_service.strategies[strategy]['success_rate'] = original_rate
        
        # Check failure case
        assert result['success'] is False
        assert result['strategy'] == strategy
        assert result['attempts'] >= 1
        
        # Check remediation history
        last_record = remediation_service.remediation_history[-1]
        assert last_record['job_id'] == 'test_job_failure'
        assert last_record['success'] is False
    
    def test_active_remediation_tracking(self, remediation_service):
        """Test tracking of active remediations"""
        # Check initial state
        assert len(remediation_service.get_active_remediations()) == 0
        
        # Start a remediation in background (simulate by adding to active)
        job_id = 'active_test_job'
        remediation_service.active_remediations[job_id] = {
            'strategy': 'retry_with_exponential_backoff',
            'error_type': 'network_timeout',
            'confidence': 0.9,
            'start_time': time.time(),
            'status': 'in_progress'
        }
        
        # Check active remediations
        active = remediation_service.get_active_remediations()
        assert len(active) == 1
        assert job_id in active
        
        # Execute remediation (should remove from active)
        remediation_service.execute_remediation(
            job_id=job_id,
            strategy='retry_with_exponential_backoff',
            error_type='network_timeout',
            confidence=0.9
        )
        
        # Should no longer be active
        assert len(remediation_service.get_active_remediations()) == 0
    
    def test_remediation_history(self, remediation_service):
        """Test remediation history tracking"""
        initial_history_length = len(remediation_service.remediation_history)
        
        # Execute multiple remediations
        test_jobs = ['history_job_1', 'history_job_2', 'history_job_3']
        
        for job_id in test_jobs:
            remediation_service.execute_remediation(
                job_id=job_id,
                strategy='retry_with_exponential_backoff',
                error_type='test_error',
                confidence=0.8
            )
        
        # Check history length
        history = remediation_service.get_remediation_history()
        assert len(history) == initial_history_length + len(test_jobs)
        
        # Check history structure
        for record in history[-len(test_jobs):]:
            required_fields = ['job_id', 'strategy', 'error_type', 'confidence', 'success', 'duration', 'timestamp']
            for field in required_fields:
                assert field in record
    
    def test_remediation_statistics(self, remediation_service):
        """Test remediation statistics calculation"""
        # Execute some test remediations
        for i in range(5):
            # Alternate success/failure
            strategy = 'retry_with_exponential_backoff'
            original_rate = remediation_service.strategies[strategy]['success_rate']
            remediation_service.strategies[strategy]['success_rate'] = 1.0 if i % 2 == 0 else 0.0
            
            remediation_service.execute_remediation(
                job_id=f'stats_job_{i}',
                strategy=strategy,
                error_type='test_error',
                confidence=0.8
            )
            
            # Restore original rate
            remediation_service.strategies[strategy]['success_rate'] = original_rate
        
        # Get statistics
        stats = remediation_service.get_remediation_stats()
        
        # Check structure
        required_fields = [
            'total_remediations', 'successful_remediations', 'success_rate',
            'average_duration', 'strategy_stats', 'error_type_stats', 'active_remediations'
        ]
        for field in required_fields:
            assert field in stats
        
        # Check data types and ranges
        assert isinstance(stats['total_remediations'], int)
        assert isinstance(stats['successful_remediations'], int)
        assert isinstance(stats['success_rate'], float)
        assert isinstance(stats['average_duration'], float)
        assert isinstance(stats['strategy_stats'], dict)
        assert isinstance(stats['error_type_stats'], dict)
        assert isinstance(stats['active_remediations'], int)
        
        # Check ranges
        assert stats['total_remediations'] >= 5
        assert 0 <= stats['successful_remediations'] <= stats['total_remediations']
        assert 0.0 <= stats['success_rate'] <= 100.0
        assert stats['average_duration'] >= 0.0
    
    def test_strategy_execution_details(self, remediation_service):
        """Test strategy-specific execution details"""
        strategies_to_test = [
            'retry_with_exponential_backoff',
            'restart_with_elevated_permissions',
            'update_schema_and_retry',
            'scale_compute_resources'
        ]
        
        for strategy in strategies_to_test:
            result = remediation_service.execute_remediation(
                job_id=f'strategy_test_{strategy}',
                strategy=strategy,
                error_type='test_error',
                confidence=0.9
            )
            
            # Should have strategy-specific details
            assert result['strategy'] == strategy
            assert isinstance(result['details'], str)
            assert len(result['details']) > 0
            
            # Details should be strategy-specific
            assert strategy.replace('_', ' ') in result['details'].lower() or result['success']
    
    def test_invalid_strategy_handling(self, remediation_service):
        """Test handling of invalid remediation strategies"""
        with pytest.raises(ValueError):
            remediation_service.execute_remediation(
                job_id='invalid_strategy_job',
                strategy='nonexistent_strategy',
                error_type='test_error',
                confidence=0.9
            )
    
    def test_cancel_remediation(self, remediation_service):
        """Test remediation cancellation"""
        job_id = 'cancel_test_job'
        
        # Add job to active remediations
        remediation_service.active_remediations[job_id] = {
            'strategy': 'retry_with_exponential_backoff',
            'error_type': 'test_error',
            'confidence': 0.8,
            'start_time': time.time(),
            'status': 'in_progress'
        }
        
        # Cancel remediation
        result = remediation_service.cancel_remediation(job_id)
        assert result is True
        
        # Should no longer be active
        assert job_id not in remediation_service.active_remediations
        
        # Should be in history as cancelled
        history = remediation_service.get_remediation_history()
        cancelled_record = next((r for r in history if r['job_id'] == job_id), None)
        assert cancelled_record is not None
        assert cancelled_record['success'] is False
        assert 'cancelled' in cancelled_record['details'].lower()
        
        # Try to cancel non-existent job
        result = remediation_service.cancel_remediation('nonexistent_job')
        assert result is False
    
    def test_simulate_remediation(self, remediation_service):
        """Test remediation simulation functionality"""
        job_id = 'simulation_test_job'
        strategy = 'retry_with_exponential_backoff'
        
        # Test forced success simulation
        result = remediation_service.simulate_remediation(
            job_id=job_id,
            strategy=strategy,
            force_success=True
        )
        
        assert result['success'] is True
        assert result['strategy'] == strategy
        
        # Test normal simulation (should follow strategy success rate)
        result = remediation_service.simulate_remediation(
            job_id=f'{job_id}_normal',
            strategy=strategy,
            force_success=False
        )
        
        assert isinstance(result['success'], bool)
        assert result['strategy'] == strategy
    
    def test_success_details_generation(self, remediation_service):
        """Test generation of success details for different strategies"""
        strategies = list(remediation_service.strategies.keys())
        
        for strategy in strategies:
            details = remediation_service._get_success_details(strategy, 1)
            
            assert isinstance(details, str)
            assert len(details) > 0
            
            # Should contain strategy-related keywords
            strategy_keywords = strategy.split('_')
            # At least one keyword should appear in details (case insensitive)
            assert any(keyword in details.lower() for keyword in strategy_keywords)
    
    def test_concurrent_remediation_execution(self, remediation_service):
        """Test concurrent execution of multiple remediations"""
        import threading
        
        results = []
        threads = []
        
        def execute_remediation(job_id):
            result = remediation_service.execute_remediation(
                job_id=job_id,
                strategy='retry_with_exponential_backoff',
                error_type='test_error',
                confidence=0.8
            )
            results.append(result)
        
        # Start multiple remediation threads
        for i in range(3):
            thread = threading.Thread(target=execute_remediation, args=(f'concurrent_job_{i}',))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # All should have completed
        assert len(results) == 3
        for result in results:
            assert 'success' in result
            assert 'duration' in result
    
    @pytest.mark.parametrize("strategy,expected_max_attempts", [
        ("retry_with_exponential_backoff", 3),
        ("restart_with_elevated_permissions", 2),
        ("update_schema_and_retry", 1),
        ("escalate_to_human", 1)
    ])
    def test_strategy_max_attempts(self, remediation_service, strategy, expected_max_attempts):
        """Test that strategies respect their max attempt configuration"""
        config = remediation_service.strategies[strategy]
        assert config['max_attempts'] == expected_max_attempts
        
        # Force failure to test retry behavior
        original_rate = config['success_rate']
        config['success_rate'] = 0.0  # Force failure
        
        result = remediation_service.execute_remediation(
            job_id=f'max_attempts_test_{strategy}',
            strategy=strategy,
            error_type='test_error',
            confidence=0.9
        )
        
        # Restore original rate
        config['success_rate'] = original_rate
        
        # Should have attempted the maximum number of times
        assert result['attempts'] == expected_max_attempts
        assert result['success'] is False
    
    def test_remediation_timing(self, remediation_service):
        """Test that remediations have reasonable execution times"""
        start_time = time.time()
        
        result = remediation_service.execute_remediation(
            job_id='timing_test_job',
            strategy='retry_with_exponential_backoff',
            error_type='test_error',
            confidence=0.9
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (allowing for test environment)
        assert total_time < 10.0  # 10 seconds max for test
        assert result['duration'] > 0
        assert result['duration'] <= total_time

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
