"""
ML-based Error Classification System
Classifies pipeline errors into categories and predicts remediation strategies
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorClassifier:
    """ML model for classifying pipeline errors and predicting remediation strategies"""
    
    def __init__(self, model_path: str = "models/error_classifier.joblib"):
        self.model_path = model_path
        self.model = None
        self.error_categories = [
            'permission_denied',
            'network_timeout',
            'schema_mismatch',
            'resource_exhaustion',
            'data_duplication',
            'missing_dependency',
            'configuration_error',
            'quota_exceeded',
            'authentication_failure',
            'data_corruption',
            'unknown_error'
        ]
        
        self.remediation_strategies = {
            'permission_denied': 'restart_with_elevated_permissions',
            'network_timeout': 'retry_with_exponential_backoff',
            'schema_mismatch': 'update_schema_and_retry',
            'resource_exhaustion': 'scale_compute_resources',
            'data_duplication': 'run_deduplication_script',
            'missing_dependency': 'install_dependencies_and_retry',
            'configuration_error': 'update_configuration_and_retry',
            'quota_exceeded': 'request_quota_increase',
            'authentication_failure': 'refresh_credentials_and_retry',
            'data_corruption': 'run_data_validation_and_cleanup',
            'unknown_error': 'escalate_to_human'
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the ML model"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Creating new model.")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create and train a new ML model"""
        # Create pipeline with TF-IDF vectorizer and Random Forest classifier
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                lowercase=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        # Generate synthetic training data
        training_data = self._generate_training_data()
        
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(
            training_data['error_message'],
            training_data['error_type'],
            test_size=0.2,
            random_state=42,
            stratify=training_data['error_type']
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def _generate_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for error classification"""
        
        error_patterns = {
            'permission_denied': [
                'Access denied for user',
                'Permission denied: Cannot access',
                'Insufficient privileges to',
                'User does not have permission',
                'Access forbidden for service account',
                'IAM policy violation',
                'Service account lacks required roles'
            ],
            'network_timeout': [
                'Connection timeout after',
                'Network timeout occurred',
                'Request timed out',
                'Connection reset by peer',
                'Socket timeout exception',
                'Network unreachable',
                'Connection refused'
            ],
            'schema_mismatch': [
                'Schema validation failed',
                'Column not found in table',
                'Data type mismatch',
                'Schema incompatibility detected',
                'Table structure changed',
                'Field type conversion error',
                'Missing required columns'
            ],
            'resource_exhaustion': [
                'Out of memory error',
                'CPU limit exceeded',
                'Disk space insufficient',
                'Memory allocation failed',
                'Resource quota exceeded',
                'Worker nodes unavailable',
                'Insufficient compute resources'
            ],
            'data_duplication': [
                'Duplicate key violation',
                'Primary key constraint failed',
                'Unique constraint violation',
                'Duplicate records detected',
                'Data already exists',
                'Integrity constraint violation',
                'Duplicate entry found'
            ],
            'missing_dependency': [
                'Module not found',
                'Package import failed',
                'Dependency not installed',
                'Library not available',
                'Required package missing',
                'Import error occurred',
                'Missing external dependency'
            ],
            'configuration_error': [
                'Configuration file not found',
                'Invalid configuration parameter',
                'Missing configuration value',
                'Configuration syntax error',
                'Environment variable not set',
                'Config validation failed',
                'Invalid parameter value'
            ],
            'quota_exceeded': [
                'API quota exceeded',
                'Rate limit exceeded',
                'Usage limit reached',
                'Quota exhausted for',
                'Request limit exceeded',
                'Billing quota exceeded',
                'Service quota limit reached'
            ],
            'authentication_failure': [
                'Authentication failed',
                'Invalid credentials provided',
                'Token expired',
                'Unauthorized access attempt',
                'Login credentials invalid',
                'API key authentication failed',
                'Service account authentication error'
            ],
            'data_corruption': [
                'Data integrity check failed',
                'Corrupt data detected',
                'Checksum validation failed',
                'Data consistency error',
                'File corruption detected',
                'Invalid data format',
                'Data validation failed'
            ],
            'unknown_error': [
                'Unexpected error occurred',
                'Internal server error',
                'System error',
                'Unknown exception',
                'Unhandled error',
                'Generic failure',
                'System malfunction'
            ]
        }
        
        training_data = []
        
        # Generate variations of error messages
        for error_type, patterns in error_patterns.items():
            for pattern in patterns:
                # Add base pattern
                training_data.append({
                    'error_message': pattern,
                    'error_type': error_type
                })
                
                # Add variations with additional context
                variations = [
                    f"{pattern} in pipeline job_12345",
                    f"Error: {pattern} at step 3",
                    f"FAILED: {pattern} during execution",
                    f"{pattern} - retry attempt 2 failed",
                    f"Pipeline error: {pattern} in worker node"
                ]
                
                for variation in variations:
                    training_data.append({
                        'error_message': variation,
                        'error_type': error_type
                    })
        
        return pd.DataFrame(training_data)
    
    def classify_error(self, error_message: str) -> Dict[str, any]:
        """
        Classify an error message and return prediction with confidence
        
        Args:
            error_message: The error message to classify
            
        Returns:
            Dictionary containing error type, confidence, and remediation strategy
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        # Get prediction and probability
        predicted_class = self.model.predict([error_message])[0]
        probabilities = self.model.predict_proba([error_message])[0]
        
        # Get confidence score (max probability)
        confidence = float(max(probabilities))
        
        # Get remediation strategy
        remediation_strategy = self.remediation_strategies.get(
            predicted_class, 
            'escalate_to_human'
        )
        
        # Get class probabilities for all categories
        class_probabilities = dict(zip(
            self.model.classes_,
            probabilities
        ))
        
        return {
            'error_type': predicted_class,
            'confidence': confidence,
            'remediation_strategy': remediation_strategy,
            'class_probabilities': class_probabilities,
            'timestamp': datetime.now().isoformat()
        }
    
    def retrain_model(self, feedback_data: pd.DataFrame) -> Dict[str, any]:
        """
        Retrain the model with new feedback data
        
        Args:
            feedback_data: DataFrame with columns 'error_message', 'error_type', 'success'
            
        Returns:
            Dictionary containing retraining results
        """
        logger.info("Starting model retraining with feedback data")
        
        # Combine with existing training data
        existing_data = self._generate_training_data()
        
        # Add feedback data (only successful classifications)
        successful_feedback = feedback_data[feedback_data['success'] == True]
        combined_data = pd.concat([
            existing_data,
            successful_feedback[['error_message', 'error_type']]
        ], ignore_index=True)
        
        # Retrain model
        X_train, X_test, y_train, y_test = train_test_split(
            combined_data['error_message'],
            combined_data['error_type'],
            test_size=0.2,
            random_state=42,
            stratify=combined_data['error_type']
        )
        
        # Create new model instance
        new_model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                lowercase=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        
        new_model.fit(X_train, y_train)
        
        # Evaluate new model
        y_pred = new_model.predict(X_test)
        new_accuracy = accuracy_score(y_test, y_pred)
        
        # Compare with old model
        old_pred = self.model.predict(X_test)
        old_accuracy = accuracy_score(y_test, old_pred)
        
        # Update model if improved
        if new_accuracy > old_accuracy:
            self.model = new_model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model updated. Accuracy improved: {old_accuracy:.3f} -> {new_accuracy:.3f}")
            
            return {
                'success': True,
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'improvement': new_accuracy - old_accuracy,
                'feedback_samples': len(successful_feedback)
            }
        else:
            logger.info(f"Model not updated. No improvement: {old_accuracy:.3f} vs {new_accuracy:.3f}")
            return {
                'success': False,
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'improvement': new_accuracy - old_accuracy,
                'feedback_samples': len(successful_feedback)
            }
    
    def get_model_stats(self) -> Dict[str, any]:
        """Get model performance statistics"""
        if not self.model:
            return {'error': 'Model not initialized'}
        
        # Generate test data for evaluation
        test_data = self._generate_training_data()
        X_test = test_data['error_message']
        y_test = test_data['error_type']
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        avg_confidence = np.mean(np.max(probabilities, axis=1))
        
        # Count predictions per class
        class_counts = pd.Series(y_pred).value_counts().to_dict()
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'total_predictions': len(y_pred),
            'class_distribution': class_counts,
            'supported_error_types': self.error_categories,
            'model_path': self.model_path,
            'last_updated': datetime.now().isoformat()
        }
