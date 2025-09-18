"""
Automated Remediation Service
Executes healing strategies based on ML predictions
"""

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemediationService:
    """Service for executing automated remediation strategies"""
    
    def __init__(self, config):
        self.config = config
        self.active_remediations = {}
        self.remediation_history = []
        
        # Remediation strategy configurations
        self.strategies = {
            'retry_with_exponential_backoff': {
                'max_attempts': 3,
                'base_delay': 1,
                'max_delay': 60,
                'success_rate': 0.75
            },
            'restart_with_elevated_permissions': {
                'max_attempts': 2,
                'base_delay': 5,
                'success_rate': 0.85
            },
            'update_schema_and_retry': {
                'max_attempts': 1,
                'base_delay': 10,
                'success_rate': 0.90
            },
            'scale_compute_resources': {
                'max_attempts': 1,
                'base_delay': 15,
                'success_rate': 0.80
            },
            'run_deduplication_script': {
                'max_attempts': 1,
                'base_delay': 8,
                'success_rate': 0.95
            },
            'install_dependencies_and_retry': {
                'max_attempts': 2,
                'base_delay': 20,
                'success_rate': 0.88
            },
            'update_configuration_and_retry': {
                'max_attempts': 2,
                'base_delay': 3,
                'success_rate': 0.82
            },
            'request_quota_increase': {
                'max_attempts': 1,
                'base_delay': 30,
                'success_rate': 0.70
            },
            'refresh_credentials_and_retry': {
                'max_attempts': 2,
                'base_delay': 5,
                'success_rate': 0.92
            },
            'run_data_validation_and_cleanup': {
                'max_attempts': 1,
                'base_delay': 25,
                'success_rate': 0.85
            },
            'escalate_to_human': {
                'max_attempts': 1,
                'base_delay': 0,
                'success_rate': 1.0  # Always succeeds (escalation)
            }
        }
    
    def execute_remediation(self, job_id: str, strategy: str, error_type: str, confidence: float) -> Dict[str, Any]:
        """
        Execute a remediation strategy for a failed job
        
        Args:
            job_id: The failed job ID
            strategy: The remediation strategy to execute
            error_type: The classified error type
            confidence: The ML model confidence score
            
        Returns:
            Dictionary containing remediation results
        """
        start_time = time.time()
        
        logger.info(f"Starting remediation for job {job_id} with strategy: {strategy}")
        
        # Track active remediation
        self.active_remediations[job_id] = {
            'strategy': strategy,
            'error_type': error_type,
            'confidence': confidence,
            'start_time': start_time,
            'status': 'in_progress'
        }
        
        try:
            # Execute the specific strategy
            result = self._execute_strategy(job_id, strategy, error_type)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update remediation record
            remediation_record = {
                'job_id': job_id,
                'strategy': strategy,
                'error_type': error_type,
                'confidence': confidence,
                'success': result['success'],
                'duration': duration,
                'attempts': result.get('attempts', 1),
                'details': result.get('details', ''),
                'timestamp': datetime.now()
            }
            
            self.remediation_history.append(remediation_record)
            
            # Remove from active remediations
            if job_id in self.active_remediations:
                del self.active_remediations[job_id]
            
            logger.info(f"Remediation completed for job {job_id}: "
                       f"{'SUCCESS' if result['success'] else 'FAILED'} in {duration:.2f}s")
            
            return {
                'success': result['success'],
                'duration': duration,
                'strategy': strategy,
                'attempts': result.get('attempts', 1),
                'details': result.get('details', ''),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(f"Remediation failed for job {job_id}: {e}")
            
            # Record failed remediation
            remediation_record = {
                'job_id': job_id,
                'strategy': strategy,
                'error_type': error_type,
                'confidence': confidence,
                'success': False,
                'duration': duration,
                'attempts': 1,
                'details': f"Exception: {str(e)}",
                'timestamp': datetime.now()
            }
            
            self.remediation_history.append(remediation_record)
            
            # Remove from active remediations
            if job_id in self.active_remediations:
                del self.active_remediations[job_id]
            
            return {
                'success': False,
                'duration': duration,
                'strategy': strategy,
                'attempts': 1,
                'details': f"Exception: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_strategy(self, job_id: str, strategy: str, error_type: str) -> Dict[str, Any]:
        """Execute a specific remediation strategy"""
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown remediation strategy: {strategy}")
        
        strategy_config = self.strategies[strategy]
        max_attempts = strategy_config['max_attempts']
        base_delay = strategy_config['base_delay']
        success_rate = strategy_config['success_rate']
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Executing {strategy} for job {job_id} (attempt {attempt}/{max_attempts})")
            
            # Simulate strategy execution time
            execution_time = base_delay + random.uniform(0, 5)
            time.sleep(min(execution_time, 2))  # Cap at 2 seconds for demo
            
            # Simulate success/failure based on strategy success rate
            success = random.random() < success_rate
            
            if success:
                details = self._get_success_details(strategy, attempt)
                return {
                    'success': True,
                    'attempts': attempt,
                    'details': details
                }
            else:
                logger.warning(f"Attempt {attempt} failed for strategy {strategy}")
                
                # If not the last attempt, wait before retrying
                if attempt < max_attempts:
                    retry_delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    time.sleep(min(retry_delay, 1))  # Cap at 1 second for demo
        
        # All attempts failed
        details = f"All {max_attempts} attempts failed for strategy {strategy}"
        return {
            'success': False,
            'attempts': max_attempts,
            'details': details
        }
    
    def _get_success_details(self, strategy: str, attempt: int) -> str:
        """Get success details for a remediation strategy"""
        
        details_map = {
            'retry_with_exponential_backoff': f"Job retried successfully after {attempt} attempt(s) with exponential backoff",
            'restart_with_elevated_permissions': f"Job restarted with elevated service account permissions",
            'update_schema_and_retry': f"Schema updated to match expected format and job retried",
            'scale_compute_resources': f"Compute resources scaled up and job retried",
            'run_deduplication_script': f"Data deduplication script executed successfully",
            'install_dependencies_and_retry': f"Missing dependencies installed and job retried",
            'update_configuration_and_retry': f"Configuration parameters updated and job retried",
            'request_quota_increase': f"Quota increase requested and approved automatically",
            'refresh_credentials_and_retry': f"Service account credentials refreshed and job retried",
            'run_data_validation_and_cleanup': f"Data validation and cleanup completed successfully",
            'escalate_to_human': f"Issue escalated to human operator for manual intervention"
        }
        
        return details_map.get(strategy, f"Strategy {strategy} executed successfully")
    
    def get_active_remediations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active remediations"""
        return self.active_remediations.copy()
    
    def get_remediation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get remediation history"""
        # Sort by timestamp descending
        sorted_history = sorted(
            self.remediation_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def get_remediation_stats(self) -> Dict[str, Any]:
        """Get remediation statistics"""
        if not self.remediation_history:
            return {
                'total_remediations': 0,
                'successful_remediations': 0,
                'success_rate': 0.0,
                'average_duration': 0.0,
                'strategy_stats': {},
                'error_type_stats': {},
                'active_remediations': len(self.active_remediations)
            }
        
        total_remediations = len(self.remediation_history)
        successful_remediations = sum(1 for r in self.remediation_history if r['success'])
        success_rate = (successful_remediations / total_remediations) * 100
        
        # Calculate average duration
        total_duration = sum(r['duration'] for r in self.remediation_history)
        average_duration = total_duration / total_remediations
        
        # Strategy statistics
        strategy_stats = {}
        for record in self.remediation_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0,
                    'successful': 0,
                    'total_duration': 0.0
                }
            
            strategy_stats[strategy]['total'] += 1
            if record['success']:
                strategy_stats[strategy]['successful'] += 1
            strategy_stats[strategy]['total_duration'] += record['duration']
        
        # Calculate success rates and average durations for each strategy
        for strategy in strategy_stats:
            stats = strategy_stats[strategy]
            stats['success_rate'] = (stats['successful'] / stats['total']) * 100
            stats['average_duration'] = stats['total_duration'] / stats['total']
        
        # Error type statistics
        error_type_stats = {}
        for record in self.remediation_history:
            error_type = record['error_type']
            if error_type not in error_type_stats:
                error_type_stats[error_type] = {
                    'total': 0,
                    'successful': 0
                }
            
            error_type_stats[error_type]['total'] += 1
            if record['success']:
                error_type_stats[error_type]['successful'] += 1
        
        # Calculate success rates for each error type
        for error_type in error_type_stats:
            stats = error_type_stats[error_type]
            stats['success_rate'] = (stats['successful'] / stats['total']) * 100
        
        return {
            'total_remediations': total_remediations,
            'successful_remediations': successful_remediations,
            'success_rate': round(success_rate, 2),
            'average_duration': round(average_duration, 2),
            'strategy_stats': strategy_stats,
            'error_type_stats': error_type_stats,
            'active_remediations': len(self.active_remediations)
        }
    
    def cancel_remediation(self, job_id: str) -> bool:
        """Cancel an active remediation"""
        if job_id in self.active_remediations:
            remediation = self.active_remediations[job_id]
            
            # Record cancelled remediation
            remediation_record = {
                'job_id': job_id,
                'strategy': remediation['strategy'],
                'error_type': remediation['error_type'],
                'confidence': remediation['confidence'],
                'success': False,
                'duration': time.time() - remediation['start_time'],
                'attempts': 1,
                'details': 'Remediation cancelled by user',
                'timestamp': datetime.now()
            }
            
            self.remediation_history.append(remediation_record)
            del self.active_remediations[job_id]
            
            logger.info(f"Remediation cancelled for job {job_id}")
            return True
        
        return False
    
    def simulate_remediation(self, job_id: str, strategy: str, force_success: bool = False) -> Dict[str, Any]:
        """Simulate a remediation for testing purposes"""
        
        # Override success rate if forced
        if force_success and strategy in self.strategies:
            original_success_rate = self.strategies[strategy]['success_rate']
            self.strategies[strategy]['success_rate'] = 1.0
        
        try:
            result = self.execute_remediation(
                job_id=job_id,
                strategy=strategy,
                error_type='simulated_error',
                confidence=0.95
            )
            
            return result
            
        finally:
            # Restore original success rate
            if force_success and strategy in self.strategies:
                self.strategies[strategy]['success_rate'] = original_success_rate
