"""
RAG-Enhanced Error Classifier
Combines traditional ML classification with retrieval-augmented generation
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.rag.vector_store import RAGKnowledgeBase


class RAGErrorClassifier:
    """
    RAG-enhanced error classifier that combines:
    1. Traditional ML classification (existing model)
    2. Historical knowledge retrieval 
    3. Contextual reasoning for improved accuracy
    """
    
    def __init__(self, traditional_classifier, config=None):
        self.traditional_classifier = traditional_classifier
        self.config = config or {}
        
        # Initialize RAG knowledge base
        self.knowledge_base = RAGKnowledgeBase()
        
        # RAG configuration parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.3)
        self.max_retrieved_docs = self.config.get('max_retrieved_docs', 5)
        self.confidence_boost_threshold = self.config.get('confidence_boost_threshold', 0.1)
        self.rag_weight = self.config.get('rag_weight', 0.3)
        
        # Initialize with some sample knowledge if empty
        self._initialize_sample_knowledge()
        
        logging.info("RAG-Enhanced Error Classifier initialized")
    
    def _initialize_sample_knowledge(self):
        """Initialize with sample knowledge if knowledge base is empty"""
        stats = self.knowledge_base.get_knowledge_stats()
        
        if stats['total_documents'] == 0:
            # Add sample historical error cases
            sample_errors = [
                {
                    'error_message': 'BigQuery job failed: Access denied to dataset prod-analytics',
                    'error_type': 'permission_denied',
                    'resolution_strategy': 'Added BigQuery Data Editor role to service account',
                    'success': True,
                    'resolution_time_minutes': 15,
                    'context': 'Production data pipeline accessing analytics dataset',
                    'business_impact': 'medium'
                },
                {
                    'error_message': 'Dataflow job timeout after 60 minutes in region us-central1',
                    'error_type': 'network_timeout', 
                    'resolution_strategy': 'Increased worker machine type from n1-standard-1 to n1-standard-4',
                    'success': True,
                    'resolution_time_minutes': 45,
                    'context': 'Large dataset processing during peak hours',
                    'business_impact': 'high'
                },
                {
                    'error_message': 'Cloud Storage upload failed: Insufficient permissions for bucket',
                    'error_type': 'permission_denied',
                    'resolution_strategy': 'Added Storage Object Creator role to service account',
                    'success': True,
                    'resolution_time_minutes': 10,
                    'context': 'Daily backup job writing to storage bucket',
                    'business_impact': 'medium'
                },
                {
                    'error_message': 'Pub/Sub subscription acknowledgment failed: quota exceeded',
                    'error_type': 'quota_exceeded',
                    'resolution_strategy': 'Requested quota increase and implemented message batching',
                    'success': True,
                    'resolution_time_minutes': 120,
                    'context': 'High-volume message processing during traffic spike',
                    'business_impact': 'critical'
                },
                {
                    'error_message': 'SQL query failed: table not found in dataset',
                    'error_type': 'schema_mismatch',
                    'resolution_strategy': 'Updated table name reference and verified schema',
                    'success': True,
                    'resolution_time_minutes': 30,
                    'context': 'Analytics pipeline after table rename',
                    'business_impact': 'low'
                }
            ]
            
            for error in sample_errors:
                self.knowledge_base.add_error_case(error)
            
            # Add sample runbooks
            sample_runbooks = [
                {
                    'title': 'BigQuery Permission Issues Troubleshooting',
                    'description': 'Step-by-step guide for resolving BigQuery access denied errors',
                    'steps': '1. Identify service account used 2. Check current IAM roles 3. Add appropriate BigQuery roles 4. Test access 5. Monitor for 24 hours',
                    'applicable_errors': 'permission_denied, authentication_failure',
                    'estimated_time': '15-30 minutes'
                },
                {
                    'title': 'Dataflow Job Performance Optimization',
                    'description': 'Guide for optimizing slow or timing out Dataflow jobs',
                    'steps': '1. Check worker resource utilization 2. Analyze data skew patterns 3. Adjust machine types 4. Optimize windowing 5. Enable autoscaling',
                    'applicable_errors': 'network_timeout, resource_exhaustion',
                    'estimated_time': '30-60 minutes'
                }
            ]
            
            for runbook in sample_runbooks:
                self.knowledge_base.add_runbook(runbook)
            
            # Save the initialized knowledge
            self.knowledge_base.save_knowledge()
            
            logging.info("Initialized RAG knowledge base with sample data")
    
    def classify_with_rag(
        self, 
        error_message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced error classification using RAG
        
        Process:
        1. Get traditional ML classification
        2. Retrieve similar historical cases
        3. Analyze patterns and boost confidence
        4. Generate enhanced classification result
        """
        
        # Step 1: Traditional ML classification
        traditional_result = self.traditional_classifier.classify_error(error_message)
        base_prediction = traditional_result.get('error_type', 'unknown')
        base_confidence = traditional_result.get('confidence', 0.5)
        
        # Normalize confidence to [0, 1] range
        if hasattr(base_confidence, '__iter__') and base_confidence:
            base_confidence = float(max(base_confidence))
        else:
            base_confidence = float(base_confidence) if base_confidence else 0.5
        base_confidence = max(0.0, min(1.0, (base_confidence + 1.0) / 2.0))
        
        # Step 2: RAG retrieval - find similar historical cases
        similar_errors = self.knowledge_base.find_similar_errors(error_message, k=3)
        relevant_runbooks = self.knowledge_base.find_relevant_runbooks(error_message, k=2)
        
        # Step 3: RAG-based confidence adjustment
        rag_analysis = self._analyze_retrieved_knowledge(
            error_message=error_message,
            base_prediction=base_prediction,
            similar_errors=similar_errors,
            runbooks=relevant_runbooks,
            context=context
        )
        
        # Step 4: Combine traditional and RAG results
        final_result = self._combine_predictions(
            base_prediction=base_prediction,
            base_confidence=base_confidence,
            rag_analysis=rag_analysis,
            similar_errors=similar_errors,
            runbooks=relevant_runbooks
        )
        
        return final_result
    
    def _analyze_retrieved_knowledge(
        self,
        error_message: str,
        base_prediction: str,
        similar_errors: List[Dict],
        runbooks: List[Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze retrieved knowledge to enhance classification"""
        
        analysis = {
            'pattern_match_score': 0.0,
            'historical_success_rate': 0.0,
            'confidence_adjustment': 0.0,
            'recommended_strategy': None,
            'reasoning': [],
            'similar_case_count': len(similar_errors),
            'runbook_availability': len(runbooks) > 0
        }
        
        if not similar_errors:
            analysis['reasoning'].append("No similar historical cases found")
            return analysis
        
        # Analyze pattern matching
        error_types_seen = []
        successful_resolutions = 0
        total_resolution_time = 0
        resolution_strategies = []
        
        for error in similar_errors:
            metadata = error.get('metadata', {})
            error_type = metadata.get('error_type')
            success = metadata.get('success', False)
            resolution_time = metadata.get('resolution_time', 0)
            
            if error_type:
                error_types_seen.append(error_type)
            
            if success:
                successful_resolutions += 1
                total_resolution_time += resolution_time
                
                # Extract resolution strategy
                content = error.get('content', '')
                if 'Resolution:' in content:
                    strategy = content.split('Resolution:')[1].split('Context:')[0].strip()
                    if strategy:
                        resolution_strategies.append(strategy)
        
        # Calculate pattern match score
        if error_types_seen:
            most_common_type = max(set(error_types_seen), key=error_types_seen.count)
            type_frequency = error_types_seen.count(most_common_type) / len(error_types_seen)
            analysis['pattern_match_score'] = type_frequency
            
            # Check if most common type matches base prediction
            if most_common_type == base_prediction:
                analysis['confidence_adjustment'] = self.confidence_boost_threshold * type_frequency
                analysis['reasoning'].append(f"Historical pattern strongly supports {base_prediction} classification")
            elif type_frequency > 0.6:  # Strong disagreement
                analysis['confidence_adjustment'] = -self.confidence_boost_threshold * type_frequency
                analysis['reasoning'].append(f"Historical pattern suggests {most_common_type} rather than {base_prediction}")
        
        # Calculate success rate
        if similar_errors:
            analysis['historical_success_rate'] = successful_resolutions / len(similar_errors)
            avg_resolution_time = total_resolution_time / max(successful_resolutions, 1)
            
            analysis['reasoning'].append(f"Found {len(similar_errors)} similar cases with {analysis['historical_success_rate']:.1%} success rate")
            analysis['reasoning'].append(f"Average resolution time: {avg_resolution_time:.1f} minutes")
        
        # Recommend strategy based on most successful approach
        if resolution_strategies:
            # Use most common successful strategy
            strategy_counts = {}
            for strategy in resolution_strategies:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            analysis['recommended_strategy'] = max(strategy_counts.keys(), key=lambda x: strategy_counts[x])
            analysis['reasoning'].append(f"Recommended strategy based on {strategy_counts[analysis['recommended_strategy']]} successful cases")
        
        # Consider runbook availability
        if runbooks:
            analysis['reasoning'].append(f"Found {len(runbooks)} relevant troubleshooting runbooks")
        
        return analysis
    
    def _combine_predictions(
        self,
        base_prediction: str,
        base_confidence: float,
        rag_analysis: Dict[str, Any],
        similar_errors: List[Dict],
        runbooks: List[Dict]
    ) -> Dict[str, Any]:
        """Combine traditional ML prediction with RAG analysis"""
        
        # Apply RAG confidence adjustment
        adjusted_confidence = base_confidence + rag_analysis.get('confidence_adjustment', 0.0)
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # Determine final prediction
        final_prediction = base_prediction
        
        # If RAG strongly disagrees and has high pattern match, consider override
        pattern_score = rag_analysis.get('pattern_match_score', 0.0)
        if (pattern_score > 0.7 and 
            rag_analysis.get('confidence_adjustment', 0.0) < -self.confidence_boost_threshold and
            base_confidence < 0.6):
            
            # Find most common error type from similar cases
            error_types = []
            for error in similar_errors:
                error_type = error.get('metadata', {}).get('error_type')
                if error_type:
                    error_types.append(error_type)
            
            if error_types:
                most_common_type = max(set(error_types), key=error_types.count)
                final_prediction = most_common_type
                rag_analysis['reasoning'].append(f"Classification overridden to {most_common_type} based on strong RAG evidence")
        
        # Build comprehensive result
        result = {
            'classification': final_prediction,
            'confidence': adjusted_confidence,
            'base_ml_prediction': base_prediction,
            'base_ml_confidence': base_confidence,
            'rag_enhanced': True,
            'rag_analysis': rag_analysis,
            'similar_cases': len(similar_errors),
            'reasoning': rag_analysis.get('reasoning', []),
            'recommended_strategy': rag_analysis.get('recommended_strategy'),
            'runbook_available': len(runbooks) > 0,
            'historical_success_rate': rag_analysis.get('historical_success_rate', 0.0),
            'classification_method': 'rag_enhanced'
        }
        
        return result
    
    def add_feedback(
        self, 
        error_message: str, 
        actual_error_type: str, 
        resolution_strategy: str,
        resolution_success: bool,
        resolution_time_minutes: int,
        context: Optional[Dict] = None
    ):
        """Add feedback to improve future classifications"""
        
        error_case = {
            'error_message': error_message,
            'error_type': actual_error_type,
            'resolution_strategy': resolution_strategy,
            'success': resolution_success,
            'resolution_time_minutes': resolution_time_minutes,
            'context': json.dumps(context or {}),
            'business_impact': context.get('business_impact', 'unknown') if context else 'unknown'
        }
        
        # Add to knowledge base
        doc_id = self.knowledge_base.add_error_case(error_case)
        
        # Save updated knowledge
        self.knowledge_base.save_knowledge()
        
        logging.info(f"Added feedback for error type {actual_error_type}, success: {resolution_success}")
        
        return doc_id
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get RAG classifier statistics"""
        kb_stats = self.knowledge_base.get_knowledge_stats()
        
        return {
            'knowledge_base_stats': kb_stats,
            'rag_config': {
                'similarity_threshold': self.similarity_threshold,
                'max_retrieved_docs': self.max_retrieved_docs,
                'confidence_boost_threshold': self.confidence_boost_threshold,
                'rag_weight': self.rag_weight
            },
            'classification_method': 'rag_enhanced',
            'last_updated': datetime.now().isoformat()
        }
    
    def search_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        return self.knowledge_base.search_all_knowledge(query, k)
    
    def classify(self, error_messages: List[str]) -> List[str]:
        """Backward compatibility with traditional classifier interface"""
        results = []
        for error_message in error_messages:
            rag_result = self.classify_with_rag(error_message)
            results.append(rag_result['classification'])
        return results
    
    def predict(self, error_messages: List[str]) -> List[str]:
        """Backward compatibility with traditional classifier interface"""
        return self.classify(error_messages)
    
    def get_model_stats(self) -> Dict[str, any]:
        """Get RAG model performance statistics for analytics dashboard"""
        kb_stats = self.knowledge_base.get_knowledge_stats()
        
        # Combine traditional classifier stats with RAG stats
        if self.traditional_classifier and hasattr(self.traditional_classifier, 'get_model_stats'):
            traditional_stats = self.traditional_classifier.get_model_stats()
        else:
            traditional_stats = {'error': 'Traditional classifier not available'}
        
        # Return combined statistics
        rag_stats = {
            'model_type': 'RAG-Enhanced Error Classifier',
            'knowledge_base_size': kb_stats.get('total_documents', 0),
            'error_cases': kb_stats.get('error_cases', 0),
            'runbooks': kb_stats.get('runbooks', 0),
            'similarity_threshold': self.similarity_threshold,
            'rag_confidence_boost': self.confidence_boost_threshold,
            'rag_weight': self.rag_weight,
            'last_updated': datetime.now().isoformat()
        }
        
        # If traditional classifier stats are available, merge them
        if 'error' not in traditional_stats:
            return {
                **traditional_stats,
                'rag_enhancement': rag_stats
            }
        else:
            return {
                'accuracy': 'N/A (RAG-enhanced)',
                'precision': 'N/A (RAG-enhanced)', 
                'recall': 'N/A (RAG-enhanced)',
                'f1_score': 'N/A (RAG-enhanced)',
                'rag_enhancement': rag_stats,
                'traditional_classifier_error': traditional_stats['error']
            }