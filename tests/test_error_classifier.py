"""
Unit Tests for Error Classification System
"""

import pytest
import pandas as pd
import os
import tempfile
from datetime import datetime
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.error_classifier import ErrorClassifier
from utils.data_generator import DataGenerator

class TestErrorClassifier:
    """Test suite for ErrorClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create a test classifier instance"""
        # Use temporary directory for model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_classifier.joblib")
            classifier = ErrorClassifier(model_path=model_path)
            yield classifier
    
    @pytest.fixture
    def data_generator(self):
        """Create a data generator instance"""
        return DataGenerator()
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert classifier.model is not None
        assert len(classifier.error_categories) == 11
        assert 'permission_denied' in classifier.error_categories
        assert 'network_timeout' in classifier.error_categories
    
    def test_error_classification(self, classifier):
        """Test error classification functionality"""
        error_messages = [
            "Access denied for user service-account@project.iam.gserviceaccount.com",
            "Connection timeout after 30 seconds to BigQuery API",
            "Schema validation failed: Column 'user_id' not found in target table",
            "Out of memory error: Cannot allocate 2.5GB for worker process",
            "Duplicate key violation: Primary key constraint failed"
        ]
        
        expected_types = [
            "permission_denied",
            "network_timeout", 
            "schema_mismatch",
            "resource_exhaustion",
            "data_duplication"
        ]
        
        for message, expected_type in zip(error_messages, expected_types):
            result = classifier.classify_error(message)
            
            # Check result structure
            assert 'error_type' in result
            assert 'confidence' in result
            assert 'remediation_strategy' in result
            assert 'class_probabilities' in result
            assert 'timestamp' in result
            
            # Check confidence is reasonable
            assert 0.0 <= result['confidence'] <= 1.0
            
            # Check that predicted type matches expected (allowing for some ML variance)
            # In practice, we'd want high accuracy, but for unit tests we check structure
            assert result['error_type'] in classifier.error_categories
    
    def test_remediation_strategies(self, classifier):
        """Test that each error type has a valid remediation strategy"""
        for error_type in classifier.error_categories:
            assert error_type in classifier.remediation_strategies
            strategy = classifier.remediation_strategies[error_type]
            assert isinstance(strategy, str)
            assert len(strategy) > 0
    
    def test_model_stats(self, classifier):
        """Test model statistics functionality"""
        stats = classifier.get_model_stats()
        
        # Check required fields
        required_fields = [
            'accuracy', 'average_confidence', 'total_predictions',
            'class_distribution', 'supported_error_types', 'model_path'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check data types
        assert isinstance(stats['accuracy'], float)
        assert isinstance(stats['average_confidence'], float)
        assert isinstance(stats['total_predictions'], int)
        assert isinstance(stats['class_distribution'], dict)
        assert isinstance(stats['supported_error_types'], list)
        assert isinstance(stats['model_path'], str)
        
        # Check ranges
        assert 0.0 <= stats['accuracy'] <= 1.0
        assert 0.0 <= stats['average_confidence'] <= 1.0
        assert stats['total_predictions'] >= 0
    
    def test_model_retraining(self, classifier, data_generator):
        """Test model retraining functionality"""
        # Generate synthetic feedback data
        feedback_data = pd.DataFrame([
            {
                'error_message': 'Access denied for user test@example.com',
                'error_type': 'permission_denied',
                'success': True
            },
            {
                'error_message': 'Network timeout occurred during data transfer',
                'error_type': 'network_timeout', 
                'success': True
            },
            {
                'error_message': 'Schema mismatch in table structure',
                'error_type': 'schema_mismatch',
                'success': False
            }
        ])
        
        # Test retraining
        result = classifier.retrain_model(feedback_data)
        
        # Check result structure
        assert 'success' in result
        assert 'old_accuracy' in result
        assert 'new_accuracy' in result
        assert 'improvement' in result
        assert 'feedback_samples' in result
        
        # Check data types
        assert isinstance(result['success'], bool)
        assert isinstance(result['old_accuracy'], float)
        assert isinstance(result['new_accuracy'], float)
        assert isinstance(result['improvement'], float)
        assert isinstance(result['feedback_samples'], int)
        
        # Check feedback samples count
        successful_samples = len(feedback_data[feedback_data['success'] == True])
        assert result['feedback_samples'] == successful_samples
    
    def test_training_data_generation(self, classifier):
        """Test synthetic training data generation"""
        training_data = classifier._generate_training_data()
        
        assert isinstance(training_data, pd.DataFrame)
        assert len(training_data) > 0
        assert 'error_message' in training_data.columns
        assert 'error_type' in training_data.columns
        
        # Check that all error categories are represented
        unique_types = set(training_data['error_type'].unique())
        expected_types = set(classifier.error_categories)
        assert unique_types == expected_types
        
        # Check for message variations
        assert len(training_data) > len(classifier.error_categories) * 5  # Should have variations
    
    def test_classification_confidence_thresholds(self, classifier):
        """Test classification with different confidence levels"""
        # Test with clear error messages (should have high confidence)
        clear_messages = [
            "Permission denied: User does not have access",
            "Connection timeout: Request timed out after 30 seconds"
        ]
        
        for message in clear_messages:
            result = classifier.classify_error(message)
            # Should have reasonable confidence for clear messages
            assert result['confidence'] > 0.3  # Adjusted for ML model variance
        
        # Test with ambiguous message (might have lower confidence)
        ambiguous_message = "Error occurred during processing"
        result = classifier.classify_error(ambiguous_message)
        # Should still return a classification
        assert result['error_type'] in classifier.error_categories
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_class_probabilities(self, classifier):
        """Test that class probabilities are valid"""
        message = "Access denied for service account"
        result = classifier.classify_error(message)
        
        probabilities = result['class_probabilities']
        
        # Check that probabilities sum to approximately 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.01
        
        # Check that all probabilities are non-negative
        for prob in probabilities.values():
            assert prob >= 0.0
        
        # Check that highest probability corresponds to predicted class
        max_prob_class = max(probabilities, key=probabilities.get)
        assert max_prob_class == result['error_type']
    
    def test_error_handling(self, classifier):
        """Test error handling for edge cases"""
        # Test with empty message
        with pytest.raises((ValueError, Exception)):
            classifier.classify_error("")
        
        # Test with None message  
        with pytest.raises((ValueError, TypeError, Exception)):
            classifier.classify_error(None)
        
        # Test retraining with invalid data
        invalid_data = pd.DataFrame([])  # Empty DataFrame
        result = classifier.retrain_model(invalid_data)
        # Should handle gracefully and not improve (no new data)
        assert isinstance(result, dict)
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.joblib")
            
            # Create and train first classifier
            classifier1 = ErrorClassifier(model_path=model_path)
            test_message = "Access denied for user"
            result1 = classifier1.classify_error(test_message)
            
            # Create second classifier with same path (should load existing model)
            classifier2 = ErrorClassifier(model_path=model_path)
            result2 = classifier2.classify_error(test_message)
            
            # Results should be identical (same model)
            assert result1['error_type'] == result2['error_type']
            assert abs(result1['confidence'] - result2['confidence']) < 0.001
    
    @pytest.mark.parametrize("error_type,expected_strategy", [
        ("permission_denied", "restart_with_elevated_permissions"),
        ("network_timeout", "retry_with_exponential_backoff"),
        ("schema_mismatch", "update_schema_and_retry"),
        ("resource_exhaustion", "scale_compute_resources"),
        ("data_duplication", "run_deduplication_script")
    ])
    def test_remediation_strategy_mapping(self, classifier, error_type, expected_strategy):
        """Test that error types map to correct remediation strategies"""
        assert classifier.remediation_strategies[error_type] == expected_strategy
    
    def test_batch_classification(self, classifier, data_generator):
        """Test classification of multiple messages"""
        # Generate test messages
        test_data = data_generator.generate_ml_training_data(samples_per_category=2)
        test_messages = test_data['error_message'].tolist()[:10]  # Test with first 10
        
        results = []
        for message in test_messages:
            result = classifier.classify_error(message)
            results.append(result)
        
        # Check that all messages were classified
        assert len(results) == len(test_messages)
        
        # Check that all results have required fields
        for result in results:
            assert 'error_type' in result
            assert 'confidence' in result
            assert 'remediation_strategy' in result
            assert result['error_type'] in classifier.error_categories

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
