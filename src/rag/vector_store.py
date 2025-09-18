"""
Simplified Vector Store Implementation for RAG System
Uses TF-IDF and cosine similarity for semantic search without external dependencies
"""

import json
import pickle
import os
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import re


class SimpleVectorStore:
    """
    Simplified vector store using TF-IDF and cosine similarity
    Designed for RAG-enhanced error classification without external dependencies
    """
    
    def __init__(self, store_path: str = "data/vector_store"):
        self.store_path = store_path
        self.documents: List[Dict] = []
        self.vectors: List[List[float]] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.metadata_index: Dict[str, List[int]] = defaultdict(list)
        
        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Load existing data if available
        self.load_store()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into tokens"""
        # Convert to lowercase and extract words
        text = text.lower()
        # Remove special characters and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        # Remove common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 
            'was', 'will', 'be', 'have', 'has', 'had', 'in', 'of', 'for', 
            'with', 'by', 'an', 'or', 'it', 'this', 'that', 'these', 'those'
        }
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _calculate_tf_idf(self, document_tokens: List[str]) -> List[float]:
        """Calculate TF-IDF vector for a document"""
        # Term frequency
        tf_counts = Counter(document_tokens)
        total_terms = len(document_tokens)
        
        # Create vector
        vector = []
        for word, word_idx in self.vocabulary.items():
            tf = tf_counts.get(word, 0) / max(total_terms, 1)
            idf = self.idf_scores.get(word, 0)
            vector.append(tf * idf)
        
        return vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _build_vocabulary_and_idf(self):
        """Build vocabulary and calculate IDF scores"""
        # Collect all tokens from all documents
        all_tokens = []
        document_tokens = []
        
        for doc in self.documents:
            tokens = self._preprocess_text(doc['content'])
            all_tokens.extend(tokens)
            document_tokens.append(set(tokens))  # Unique tokens per document
        
        # Build vocabulary
        unique_tokens = set(all_tokens)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        
        # Calculate IDF scores
        total_documents = len(self.documents)
        for token in unique_tokens:
            # Count documents containing this token
            doc_count = sum(1 for doc_tokens in document_tokens if token in doc_tokens)
            # Calculate IDF
            idf = math.log(total_documents / max(doc_count, 1))
            self.idf_scores[token] = idf
    
    def _rebuild_vectors(self):
        """Rebuild all document vectors"""
        self._build_vocabulary_and_idf()
        
        self.vectors = []
        for doc in self.documents:
            tokens = self._preprocess_text(doc['content'])
            vector = self._calculate_tf_idf(tokens)
            self.vectors.append(vector)
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """Add a document to the vector store"""
        # Generate unique ID if not provided
        if 'id' not in document:
            document['id'] = f"doc_{len(self.documents)}_{datetime.now().timestamp()}"
        
        # Add timestamp if not provided
        if 'created_at' not in document:
            document['created_at'] = datetime.now().isoformat()
        
        # Store document
        self.documents.append(document)
        
        # Update metadata index
        doc_index = len(self.documents) - 1
        for key, value in document.get('metadata', {}).items():
            self.metadata_index[f"{key}:{value}"].append(doc_index)
        
        # Add document type to metadata index
        if 'type' in document:
            self.metadata_index[f"type:{document['type']}"].append(doc_index)
        
        if 'priority' in document:
            self.metadata_index[f"priority:{document['priority']}"].append(doc_index)
        
        # Rebuild vectors (expensive operation - could be optimized)
        self._rebuild_vectors()
        
        return document['id']
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add multiple documents efficiently"""
        document_ids = []
        
        for document in documents:
            # Generate unique ID if not provided
            if 'id' not in document:
                document['id'] = f"doc_{len(self.documents)}_{datetime.now().timestamp()}"
            
            # Add timestamp if not provided  
            if 'created_at' not in document:
                document['created_at'] = datetime.now().isoformat()
            
            # Store document
            self.documents.append(document)
            document_ids.append(document['id'])
            
            # Update metadata index
            doc_index = len(self.documents) - 1
            for key, value in document.get('metadata', {}).items():
                self.metadata_index[f"{key}:{value}"].append(doc_index)
            
            # Add document type to metadata index
            if 'type' in document:
                self.metadata_index[f"type:{document['type']}"].append(doc_index)
            
            if 'priority' in document:
                self.metadata_index[f"priority:{document['priority']}"].append(doc_index)
        
        # Rebuild vectors once at the end
        self._rebuild_vectors()
        
        return document_ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter_criteria: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        
        if not self.documents:
            return []
        
        # Preprocess query
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []
        
        # Calculate query vector
        query_vector = self._calculate_tf_idf(query_tokens)
        if not query_vector:
            return []
        
        # Apply filters to get candidate document indices
        candidate_indices = set(range(len(self.documents)))
        
        if filter_criteria:
            filtered_indices = set()
            for key, value in filter_criteria.items():
                metadata_key = f"{key}:{value}"
                if metadata_key in self.metadata_index:
                    if not filtered_indices:
                        filtered_indices = set(self.metadata_index[metadata_key])
                    else:
                        filtered_indices &= set(self.metadata_index[metadata_key])
            
            if filtered_indices:
                candidate_indices = filtered_indices
        
        # Calculate similarities for candidate documents
        similarities = []
        for idx in candidate_indices:
            if idx < len(self.vectors):
                similarity = self._cosine_similarity(query_vector, self.vectors[idx])
                if similarity >= min_similarity:
                    similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for idx, similarity in similarities[:k]:
            result = self.documents[idx].copy()
            result['similarity_score'] = similarity
            results.append(result)
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        for doc in self.documents:
            if doc.get('id') == doc_id:
                return doc
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        for i, doc in enumerate(self.documents):
            if doc.get('id') == doc_id:
                del self.documents[i]
                # Rebuild vectors and indices
                self._rebuild_vectors()
                self._rebuild_metadata_index()
                return True
        return False
    
    def _rebuild_metadata_index(self):
        """Rebuild metadata index"""
        self.metadata_index = defaultdict(list)
        
        for doc_index, document in enumerate(self.documents):
            for key, value in document.get('metadata', {}).items():
                self.metadata_index[f"{key}:{value}"].append(doc_index)
            
            if 'type' in document:
                self.metadata_index[f"type:{document['type']}"].append(doc_index)
            
            if 'priority' in document:
                self.metadata_index[f"priority:{document['priority']}"].append(doc_index)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'storage_path': self.store_path,
            'last_updated': datetime.now().isoformat(),
            'document_types': dict(Counter(doc.get('type', 'unknown') for doc in self.documents)),
            'priority_distribution': dict(Counter(doc.get('priority', 'unknown') for doc in self.documents))
        }
    
    def save_store(self):
        """Save vector store to disk"""
        store_data = {
            'documents': self.documents,
            'vectors': self.vectors,
            'vocabulary': self.vocabulary,
            'idf_scores': self.idf_scores,
            'metadata_index': dict(self.metadata_index),
            'last_updated': datetime.now().isoformat()
        }
        
        store_file = os.path.join(self.store_path, 'vector_store.pkl')
        with open(store_file, 'wb') as f:
            pickle.dump(store_data, f)
        
        # Also save as JSON for readability (without vectors)
        readable_data = {
            'documents': self.documents,
            'vocabulary_size': len(self.vocabulary),
            'total_documents': len(self.documents),
            'last_updated': datetime.now().isoformat()
        }
        
        json_file = os.path.join(self.store_path, 'vector_store_readable.json')
        with open(json_file, 'w') as f:
            json.dump(readable_data, f, indent=2)
    
    def load_store(self):
        """Load vector store from disk"""
        store_file = os.path.join(self.store_path, 'vector_store.pkl')
        
        if os.path.exists(store_file):
            try:
                with open(store_file, 'rb') as f:
                    store_data = pickle.load(f)
                
                self.documents = store_data.get('documents', [])
                self.vectors = store_data.get('vectors', [])
                self.vocabulary = store_data.get('vocabulary', {})
                self.idf_scores = store_data.get('idf_scores', {})
                self.metadata_index = defaultdict(list, store_data.get('metadata_index', {}))
                
                print(f"Loaded vector store with {len(self.documents)} documents")
                
            except Exception as e:
                print(f"Error loading vector store: {e}")
                # Initialize empty store
                self.documents = []
                self.vectors = []
                self.vocabulary = {}
                self.idf_scores = {}
                self.metadata_index = defaultdict(list)


class RAGKnowledgeBase:
    """
    RAG-specific knowledge base wrapper around SimpleVectorStore
    Optimized for error classification and remediation knowledge
    """
    
    def __init__(self, store_path: str = "data/rag_knowledge"):
        self.vector_store = SimpleVectorStore(store_path)
        
    def add_error_case(self, error_data: Dict[str, Any]) -> str:
        """Add historical error case to knowledge base"""
        document = {
            'type': 'historical_error',
            'content': f"Error: {error_data.get('error_message', '')} Resolution: {error_data.get('resolution_strategy', '')} Context: {error_data.get('context', '')}",
            'metadata': {
                'error_type': error_data.get('error_type'),
                'success': error_data.get('success', False),
                'resolution_time': error_data.get('resolution_time_minutes', 0),
                'pipeline_id': error_data.get('pipeline_id', ''),
                'business_impact': error_data.get('business_impact', 'unknown')
            },
            'priority': 'high' if error_data.get('success') else 'low'
        }
        
        return self.vector_store.add_document(document)
    
    def add_runbook(self, runbook_data: Dict[str, Any]) -> str:
        """Add troubleshooting runbook to knowledge base"""
        document = {
            'type': 'runbook',
            'content': f"Title: {runbook_data.get('title', '')} Description: {runbook_data.get('description', '')} Steps: {runbook_data.get('steps', '')}",
            'metadata': {
                'applicable_errors': runbook_data.get('applicable_errors', ''),
                'estimated_time': runbook_data.get('estimated_time', ''),
                'automation_available': runbook_data.get('automation_available', False)
            },
            'priority': 'high'
        }
        
        return self.vector_store.add_document(document)
    
    def add_best_practice(self, practice_data: Dict[str, Any]) -> str:
        """Add best practice to knowledge base"""
        document = {
            'type': 'best_practice',
            'content': f"Practice: {practice_data.get('practice_title', '')} Description: {practice_data.get('description', '')} Implementation: {practice_data.get('implementation_steps', '')}",
            'metadata': {
                'category': practice_data.get('category', ''),
                'compliance_related': practice_data.get('compliance_related', False)
            },
            'priority': 'medium'
        }
        
        return self.vector_store.add_document(document)
    
    def find_similar_errors(self, error_message: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find similar historical errors"""
        return self.vector_store.similarity_search(
            query=error_message,
            k=k,
            filter_criteria={'type': 'historical_error', 'success': True},
            min_similarity=0.1
        )
    
    def find_relevant_runbooks(self, error_message: str, k: int = 2) -> List[Dict[str, Any]]:
        """Find relevant troubleshooting runbooks"""
        return self.vector_store.similarity_search(
            query=error_message,
            k=k,
            filter_criteria={'type': 'runbook'},
            min_similarity=0.1
        )
    
    def find_applicable_practices(self, context: str, k: int = 2) -> List[Dict[str, Any]]:
        """Find applicable best practices"""
        return self.vector_store.similarity_search(
            query=context,
            k=k,
            filter_criteria={'type': 'best_practice'},
            min_similarity=0.1
        )
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return self.vector_store.get_stats()
    
    def save_knowledge(self):
        """Save knowledge base to disk"""
        self.vector_store.save_store()
    
    def search_all_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search across all knowledge types"""
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            min_similarity=0.05
        )