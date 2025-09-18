"""
BigQuery Service for Data Storage and Analytics
Handles all database operations for the monitoring system
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryService:
    """
    BigQuery service simulation using SQLite for local development
    In production, this would use the actual Google Cloud BigQuery client
    """
    
    def __init__(self, config):
        self.config = config
        self.db_path = "pipeline_monitor.db"
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection with proper locking"""
        self.conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False,
            timeout=30.0  # 30 second timeout for database operations
        )
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        # Enable WAL mode for better concurrent access
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        self.conn.commit()
    
    def initialize_tables(self):
        """Create all required tables"""
        
        # Jobs table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_name TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT,
                error_type TEXT,
                resolution TEXT,
                retry_count INTEGER DEFAULT 0,
                auto_healed BOOLEAN DEFAULT FALSE,
                healing_duration REAL,
                confidence_score REAL
            )
        ''')
        
        # Logs table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                context TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            )
        ''')
        
        # Error patterns table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                error_signature TEXT PRIMARY KEY,
                error_type TEXT NOT NULL,
                resolution_strategy TEXT NOT NULL,
                success_rate REAL DEFAULT 0.0,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Healing actions table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS healing_actions (
                action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                success BOOLEAN,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            )
        ''')
        
        # Pending recommendations table for manual approval workflow
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS pending_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                error_type TEXT NOT NULL,
                rag_recommendation TEXT,
                mapped_strategy TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                approved_by TEXT,
                approved_at TIMESTAMP,
                executed_at TIMESTAMP,
                execution_result TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            )
        ''')
        
        self.conn.commit()
        logger.info("Database tables initialized successfully")
    
    def insert_job(self, job_data: Dict[str, Any]) -> bool:
        """Insert a new job record"""
        try:
            query = '''
                INSERT INTO jobs (
                    job_id, job_name, job_type, status, error_message,
                    error_type, resolution, retry_count, auto_healed,
                    healing_duration, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            self.conn.execute(query, (
                job_data.get('job_id'),
                job_data.get('job_name'),
                job_data.get('job_type'),
                job_data.get('status'),
                job_data.get('error_message'),
                job_data.get('error_type'),
                job_data.get('resolution'),
                job_data.get('retry_count', 0),
                job_data.get('auto_healed', False),
                job_data.get('healing_duration'),
                job_data.get('confidence_score')
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert job: {e}")
            return False
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing job record"""
        try:
            set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
            query = f"UPDATE jobs SET {set_clause} WHERE job_id = ?"
            
            values = list(updates.values()) + [job_id]
            self.conn.execute(query, values)
            self.conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job by ID"""
        try:
            cursor = self.conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", 
                (job_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    def get_jobs(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get jobs with optional filtering"""
        try:
            query = "SELECT * FROM jobs"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get jobs: {e}")
            return []
    
    def insert_log(self, log_data: Dict[str, Any]) -> bool:
        """Insert a log entry"""
        try:
            query = '''
                INSERT INTO logs (job_id, level, message, context)
                VALUES (?, ?, ?, ?)
            '''
            
            self.conn.execute(query, (
                log_data.get('job_id'),
                log_data.get('level'),
                log_data.get('message'),
                json.dumps(log_data.get('context', {}))
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert log: {e}")
            return False
    
    def get_logs(self, job_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs with optional job filtering"""
        try:
            query = "SELECT * FROM logs"
            params = []
            
            if job_id:
                query += " WHERE job_id = ?"
                params.append(job_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                log_dict = dict(row)
                if log_dict['context']:
                    log_dict['context'] = json.loads(log_dict['context'])
                logs.append(log_dict)
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return []
    
    def update_error_pattern(self, error_signature: str, pattern_data: Dict[str, Any]) -> bool:
        """Update or insert error pattern"""
        try:
            # Check if pattern exists
            cursor = self.conn.execute(
                "SELECT frequency, success_rate FROM error_patterns WHERE error_signature = ?",
                (error_signature,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                new_frequency = existing['frequency'] + 1
                
                # Update success rate if provided
                if 'success' in pattern_data:
                    current_successes = existing['success_rate'] * existing['frequency']
                    new_successes = current_successes + (1 if pattern_data['success'] else 0)
                    new_success_rate = new_successes / new_frequency
                else:
                    new_success_rate = existing['success_rate']
                
                query = '''
                    UPDATE error_patterns 
                    SET frequency = ?, success_rate = ?, last_seen = CURRENT_TIMESTAMP
                    WHERE error_signature = ?
                '''
                self.conn.execute(query, (new_frequency, new_success_rate, error_signature))
            else:
                # Insert new pattern
                query = '''
                    INSERT INTO error_patterns (
                        error_signature, error_type, resolution_strategy, 
                        success_rate, frequency
                    ) VALUES (?, ?, ?, ?, ?)
                '''
                self.conn.execute(query, (
                    error_signature,
                    pattern_data.get('error_type'),
                    pattern_data.get('resolution_strategy'),
                    1.0 if pattern_data.get('success', True) else 0.0,
                    1
                ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update error pattern: {e}")
            return False
    
    def get_error_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error patterns sorted by frequency"""
        try:
            cursor = self.conn.execute(
                "SELECT * FROM error_patterns ORDER BY frequency DESC, last_seen DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get error patterns: {e}")
            return []
    
    def insert_healing_action(self, action_data: Dict[str, Any]) -> bool:
        """Insert a healing action record"""
        try:
            query = '''
                INSERT INTO healing_actions (
                    job_id, action_type, status, completed_at, success
                ) VALUES (?, ?, ?, ?, ?)
            '''
            
            self.conn.execute(query, (
                action_data.get('job_id'),
                action_data.get('action_type'),
                action_data.get('status'),
                action_data.get('completed_at'),
                action_data.get('success')
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert healing action: {e}")
            return False
    
    def get_healing_actions(self, job_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get healing actions with optional job filtering"""
        try:
            query = "SELECT * FROM healing_actions"
            params = []
            
            if job_id:
                query += " WHERE job_id = ?"
                params.append(job_id)
            
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get healing actions: {e}")
            return []
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get key metrics for dashboard display"""
        try:
            metrics = {}
            
            # Total jobs
            cursor = self.conn.execute("SELECT COUNT(*) as total FROM jobs")
            metrics['total_jobs'] = cursor.fetchone()['total']
            
            # Failed jobs
            cursor = self.conn.execute("SELECT COUNT(*) as failed FROM jobs WHERE status = 'failed'")
            metrics['failed_jobs'] = cursor.fetchone()['failed']
            
            # Auto-healed jobs
            cursor = self.conn.execute("SELECT COUNT(*) as healed FROM jobs WHERE auto_healed = 1")
            metrics['auto_healed_jobs'] = cursor.fetchone()['healed']
            
            # Success rate
            if metrics['total_jobs'] > 0:
                success_rate = ((metrics['total_jobs'] - metrics['failed_jobs']) / metrics['total_jobs']) * 100
                metrics['success_rate'] = round(success_rate, 2)
            else:
                metrics['success_rate'] = 100.0
            
            # Healing rate
            if metrics['failed_jobs'] > 0:
                healing_rate = (metrics['auto_healed_jobs'] / metrics['failed_jobs']) * 100
                metrics['healing_rate'] = round(healing_rate, 2)
            else:
                metrics['healing_rate'] = 0.0
            
            # Average healing duration
            cursor = self.conn.execute(
                "SELECT AVG(healing_duration) as avg_duration FROM jobs WHERE auto_healed = 1"
            )
            result = cursor.fetchone()
            metrics['avg_healing_duration'] = round(result['avg_duration'] or 0, 2)
            
            # Error type distribution
            cursor = self.conn.execute('''
                SELECT error_type, COUNT(*) as count 
                FROM jobs 
                WHERE error_type IS NOT NULL 
                GROUP BY error_type 
                ORDER BY count DESC
            ''')
            error_distribution = [dict(row) for row in cursor.fetchall()]
            metrics['error_distribution'] = error_distribution
            
            # Recent jobs (last 24 hours)
            cursor = self.conn.execute('''
                SELECT COUNT(*) as recent_jobs 
                FROM jobs 
                WHERE created_at >= datetime('now', '-1 day')
            ''')
            metrics['recent_jobs'] = cursor.fetchone()['recent_jobs']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            return {}
    
    def get_trend_data(self, days: int = 7) -> Dict[str, Any]:
        """Get trend data for analytics dashboard"""
        try:
            trend_data = {}
            
            # Jobs over time
            cursor = self.conn.execute('''
                SELECT DATE(created_at) as date, 
                       COUNT(*) as total_jobs,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_jobs,
                       SUM(CASE WHEN auto_healed = 1 THEN 1 ELSE 0 END) as healed_jobs
                FROM jobs 
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY DATE(created_at)
                ORDER BY date
            '''.format(days))
            
            trend_data['daily_stats'] = [dict(row) for row in cursor.fetchall()]
            
            # Error types over time
            cursor = self.conn.execute('''
                SELECT error_type, COUNT(*) as count
                FROM jobs 
                WHERE error_type IS NOT NULL 
                AND created_at >= datetime('now', '-{} days')
                GROUP BY error_type
                ORDER BY count DESC
            '''.format(days))
            
            trend_data['error_types'] = [dict(row) for row in cursor.fetchall()]
            
            # Healing success rate over time
            cursor = self.conn.execute('''
                SELECT resolution, COUNT(*) as count
                FROM jobs 
                WHERE resolution IS NOT NULL 
                AND created_at >= datetime('now', '-{} days')
                GROUP BY resolution
            '''.format(days))
            
            trend_data['resolution_stats'] = [dict(row) for row in cursor.fetchall()]
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Failed to get trend data: {e}")
            return {}
    
    def insert_pending_recommendation(self, recommendation_data: Dict[str, Any]) -> bool:
        """Insert a new pending recommendation"""
        try:
            query = '''
                INSERT INTO pending_recommendations (
                    recommendation_id, job_id, error_type, rag_recommendation,
                    mapped_strategy, confidence_score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            
            self.conn.execute(query, (
                recommendation_data.get('recommendation_id'),
                recommendation_data.get('job_id'),
                recommendation_data.get('error_type'),
                recommendation_data.get('rag_recommendation'),
                recommendation_data.get('mapped_strategy'),
                recommendation_data.get('confidence_score'),
                recommendation_data.get('status', 'pending')
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert pending recommendation: {e}")
            return False
    
    def get_pending_recommendations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all pending recommendations"""
        try:
            query = '''
                SELECT pr.*, j.job_name, j.job_type
                FROM pending_recommendations pr
                JOIN jobs j ON pr.job_id = j.job_id
                WHERE pr.status = 'pending'
                ORDER BY pr.created_at DESC
                LIMIT ?
            '''
            
            cursor = self.conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Failed to get pending recommendations: {e}")
            return []
    
    def update_recommendation_status(self, recommendation_id: str, updates: Dict[str, Any]) -> bool:
        """Update recommendation status and details"""
        try:
            set_clause = ', '.join([f"{key} = ?" for key in updates.keys()])
            query = f"UPDATE pending_recommendations SET {set_clause} WHERE recommendation_id = ?"
            
            values = list(updates.values()) + [recommendation_id]
            self.conn.execute(query, values)
            self.conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update recommendation {recommendation_id}: {e}")
            return False
    
    def get_recommendation(self, recommendation_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific recommendation by ID"""
        try:
            query = '''
                SELECT pr.*, j.job_name, j.job_type
                FROM pending_recommendations pr
                JOIN jobs j ON pr.job_id = j.job_id
                WHERE pr.recommendation_id = ?
            '''
            
            cursor = self.conn.execute(query, (recommendation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get recommendation {recommendation_id}: {e}")
            return None
