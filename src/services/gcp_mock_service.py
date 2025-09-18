"""
Mock GCP Services for Development and Testing
Simulates Google Cloud Platform services for local development
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockCloudComposer:
    """Mock Google Cloud Composer service"""
    
    def __init__(self):
        self.dags = {}
        self.dag_runs = {}
    
    def create_dag(self, dag_id: str, schedule: str, tasks: List[Dict]) -> Dict[str, Any]:
        """Create a new DAG"""
        dag = {
            'dag_id': dag_id,
            'schedule': schedule,
            'tasks': tasks,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self.dags[dag_id] = dag
        logger.info(f"Created DAG: {dag_id}")
        
        return dag
    
    def trigger_dag(self, dag_id: str, config: Optional[Dict] = None) -> str:
        """Trigger a DAG run"""
        if dag_id not in self.dags:
            raise ValueError(f"DAG {dag_id} not found")
        
        run_id = f"{dag_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        dag_run = {
            'run_id': run_id,
            'dag_id': dag_id,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'config': config or {},
            'task_instances': []
        }
        
        self.dag_runs[run_id] = dag_run
        logger.info(f"Triggered DAG run: {run_id}")
        
        return run_id
    
    def get_dag_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get DAG run status"""
        if run_id not in self.dag_runs:
            raise ValueError(f"DAG run {run_id} not found")
        
        return self.dag_runs[run_id]
    
    def list_dags(self) -> List[Dict[str, Any]]:
        """List all DAGs"""
        return list(self.dags.values())

class MockDataflow:
    """Mock Google Cloud Dataflow service"""
    
    def __init__(self):
        self.jobs = {}
    
    def create_job(self, job_name: str, template: str, parameters: Dict) -> str:
        """Create a new Dataflow job"""
        job_id = f"dataflow_{uuid.uuid4().hex[:12]}"
        
        job = {
            'job_id': job_id,
            'job_name': job_name,
            'template': template,
            'parameters': parameters,
            'status': 'running',
            'created_at': datetime.now().isoformat(),
            'region': 'us-central1'
        }
        
        self.jobs[job_id] = job
        logger.info(f"Created Dataflow job: {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        # Simulate job progression
        job = self.jobs[job_id]
        if job['status'] == 'running':
            # Random chance to complete or fail
            rand = random.random()
            if rand < 0.1:  # 10% chance to fail
                job['status'] = 'failed'
                job['error'] = 'Simulated job failure'
            elif rand < 0.3:  # 20% chance to complete
                job['status'] = 'done'
                job['completed_at'] = datetime.now().isoformat()
        
        return job
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id not in self.jobs:
            return False
        
        self.jobs[job_id]['status'] = 'cancelled'
        logger.info(f"Cancelled Dataflow job: {job_id}")
        
        return True
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List jobs with optional status filter"""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job['status'] == status]
        
        return jobs

class MockPubSub:
    """Mock Google Cloud Pub/Sub service"""
    
    def __init__(self):
        self.topics = {}
        self.subscriptions = {}
        self.messages = {}
    
    def create_topic(self, topic_name: str) -> Dict[str, Any]:
        """Create a new topic"""
        topic = {
            'name': topic_name,
            'created_at': datetime.now().isoformat(),
            'message_count': 0
        }
        
        self.topics[topic_name] = topic
        self.messages[topic_name] = []
        
        logger.info(f"Created Pub/Sub topic: {topic_name}")
        
        return topic
    
    def create_subscription(self, subscription_name: str, topic_name: str) -> Dict[str, Any]:
        """Create a subscription to a topic"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic {topic_name} not found")
        
        subscription = {
            'name': subscription_name,
            'topic': topic_name,
            'created_at': datetime.now().isoformat(),
            'message_count': 0
        }
        
        self.subscriptions[subscription_name] = subscription
        logger.info(f"Created subscription: {subscription_name}")
        
        return subscription
    
    def publish_message(self, topic_name: str, data: Dict, attributes: Optional[Dict] = None) -> str:
        """Publish a message to a topic"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic {topic_name} not found")
        
        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        
        message = {
            'message_id': message_id,
            'data': data,
            'attributes': attributes or {},
            'publish_time': datetime.now().isoformat(),
            'topic': topic_name
        }
        
        self.messages[topic_name].append(message)
        self.topics[topic_name]['message_count'] += 1
        
        logger.debug(f"Published message {message_id} to topic {topic_name}")
        
        return message_id
    
    def pull_messages(self, subscription_name: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Pull messages from a subscription"""
        if subscription_name not in self.subscriptions:
            raise ValueError(f"Subscription {subscription_name} not found")
        
        subscription = self.subscriptions[subscription_name]
        topic_name = subscription['topic']
        
        # Get messages from topic
        topic_messages = self.messages.get(topic_name, [])
        pulled_messages = topic_messages[:max_messages]
        
        # Remove pulled messages (simulate acknowledgment)
        self.messages[topic_name] = topic_messages[max_messages:]
        
        return pulled_messages
    
    def get_topic_stats(self, topic_name: str) -> Dict[str, Any]:
        """Get topic statistics"""
        if topic_name not in self.topics:
            raise ValueError(f"Topic {topic_name} not found")
        
        topic = self.topics[topic_name]
        message_count = len(self.messages.get(topic_name, []))
        
        return {
            'name': topic_name,
            'message_count': message_count,
            'created_at': topic['created_at'],
            'subscribers': len([s for s in self.subscriptions.values() if s['topic'] == topic_name])
        }

class MockCloudFunctions:
    """Mock Google Cloud Functions service"""
    
    def __init__(self):
        self.functions = {}
        self.invocations = []
    
    def deploy_function(self, function_name: str, source_code: str, trigger: Dict) -> Dict[str, Any]:
        """Deploy a cloud function"""
        function = {
            'name': function_name,
            'source_code': source_code,
            'trigger': trigger,
            'status': 'active',
            'deployed_at': datetime.now().isoformat(),
            'invocation_count': 0
        }
        
        self.functions[function_name] = function
        logger.info(f"Deployed Cloud Function: {function_name}")
        
        return function
    
    def invoke_function(self, function_name: str, data: Dict) -> Dict[str, Any]:
        """Invoke a cloud function"""
        if function_name not in self.functions:
            raise ValueError(f"Function {function_name} not found")
        
        invocation_id = f"inv_{uuid.uuid4().hex[:12]}"
        
        # Simulate function execution
        execution_time = random.uniform(0.1, 2.0)  # 100ms to 2s
        
        invocation = {
            'invocation_id': invocation_id,
            'function_name': function_name,
            'input_data': data,
            'execution_time': execution_time,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'result': {'message': f'Function {function_name} executed successfully'}
        }
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            invocation['status'] = 'error'
            invocation['result'] = {'error': 'Simulated function execution error'}
        
        self.invocations.append(invocation)
        self.functions[function_name]['invocation_count'] += 1
        
        time.sleep(min(execution_time, 0.1))  # Cap simulation delay
        
        return invocation
    
    def get_function_logs(self, function_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get function execution logs"""
        function_invocations = [
            inv for inv in self.invocations 
            if inv['function_name'] == function_name
        ]
        
        return sorted(function_invocations, key=lambda x: x['timestamp'], reverse=True)[:limit]

class MockCloudWorkflows:
    """Mock Google Cloud Workflows service"""
    
    def __init__(self):
        self.workflows = {}
        self.executions = {}
    
    def create_workflow(self, workflow_name: str, definition: Dict) -> Dict[str, Any]:
        """Create a new workflow"""
        workflow = {
            'name': workflow_name,
            'definition': definition,
            'status': 'active',
            'created_at': datetime.now().isoformat(),
            'execution_count': 0
        }
        
        self.workflows[workflow_name] = workflow
        logger.info(f"Created workflow: {workflow_name}")
        
        return workflow
    
    def execute_workflow(self, workflow_name: str, input_data: Dict) -> str:
        """Execute a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        execution = {
            'execution_id': execution_id,
            'workflow_name': workflow_name,
            'input_data': input_data,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'steps_completed': 0,
            'total_steps': len(self.workflows[workflow_name]['definition'].get('steps', []))
        }
        
        self.executions[execution_id] = execution
        self.workflows[workflow_name]['execution_count'] += 1
        
        logger.info(f"Started workflow execution: {execution_id}")
        
        return execution_id
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        if execution_id not in self.executions:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution = self.executions[execution_id]
        
        # Simulate workflow progression
        if execution['status'] == 'running':
            rand = random.random()
            if rand < 0.2:  # 20% chance to complete
                execution['status'] = 'succeeded'
                execution['end_time'] = datetime.now().isoformat()
                execution['steps_completed'] = execution['total_steps']
            elif rand < 0.05:  # 5% chance to fail
                execution['status'] = 'failed'
                execution['end_time'] = datetime.now().isoformat()
                execution['error'] = 'Simulated workflow failure'
            else:
                # Progress through steps
                if execution['steps_completed'] < execution['total_steps']:
                    execution['steps_completed'] += 1
        
        return execution

class GCPMockService:
    """Unified mock service for all GCP components"""
    
    def __init__(self):
        self.composer = MockCloudComposer()
        self.dataflow = MockDataflow()
        self.pubsub = MockPubSub()
        self.functions = MockCloudFunctions()
        self.workflows = MockCloudWorkflows()
        
        # Initialize some default resources
        self._initialize_default_resources()
    
    def _initialize_default_resources(self):
        """Initialize default GCP resources for testing"""
        
        # Create default Pub/Sub topics
        self.pubsub.create_topic('pipeline-events')
        self.pubsub.create_topic('error-notifications')
        self.pubsub.create_topic('healing-actions')
        
        # Create subscriptions
        self.pubsub.create_subscription('monitor-events', 'pipeline-events')
        self.pubsub.create_subscription('error-processor', 'error-notifications')
        
        # Deploy default Cloud Functions
        self.functions.deploy_function(
            'error-processor',
            'def main(event, context): pass',
            {'type': 'pubsub', 'topic': 'pipeline-events'}
        )
        
        self.functions.deploy_function(
            'healing-orchestrator',
            'def main(event, context): pass',
            {'type': 'pubsub', 'topic': 'error-notifications'}
        )
        
        # Create default workflows
        healing_workflow = {
            'steps': [
                {'name': 'classify_error', 'type': 'function_call'},
                {'name': 'determine_strategy', 'type': 'decision'},
                {'name': 'execute_remediation', 'type': 'action'},
                {'name': 'verify_success', 'type': 'validation'}
            ]
        }
        
        self.workflows.create_workflow('auto-healing-workflow', healing_workflow)
        
        logger.info("Initialized default GCP mock resources")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all mock services"""
        return {
            'composer': {
                'dags': len(self.composer.dags),
                'active_runs': len([r for r in self.composer.dag_runs.values() if r['status'] == 'running'])
            },
            'dataflow': {
                'total_jobs': len(self.dataflow.jobs),
                'running_jobs': len([j for j in self.dataflow.jobs.values() if j['status'] == 'running'])
            },
            'pubsub': {
                'topics': len(self.pubsub.topics),
                'subscriptions': len(self.pubsub.subscriptions),
                'total_messages': sum(len(msgs) for msgs in self.pubsub.messages.values())
            },
            'functions': {
                'deployed_functions': len(self.functions.functions),
                'total_invocations': len(self.functions.invocations)
            },
            'workflows': {
                'workflows': len(self.workflows.workflows),
                'active_executions': len([e for e in self.workflows.executions.values() if e['status'] == 'running'])
            }
        }
    
    def simulate_pipeline_activity(self, duration_minutes: int = 5):
        """Simulate realistic pipeline activity"""
        logger.info(f"Simulating pipeline activity for {duration_minutes} minutes")
        
        # Create some DAGs
        for i in range(3):
            dag_id = f"data_pipeline_{i+1}"
            self.composer.create_dag(
                dag_id=dag_id,
                schedule='0 */6 * * *',  # Every 6 hours
                tasks=[
                    {'name': 'extract_data', 'type': 'python'},
                    {'name': 'transform_data', 'type': 'sql'},
                    {'name': 'load_data', 'type': 'bigquery'}
                ]
            )
        
        # Create some Dataflow jobs
        for i in range(5):
            job_name = f"streaming_job_{i+1}"
            self.dataflow.create_job(
                job_name=job_name,
                template='streaming_template',
                parameters={
                    'input_topic': f'projects/test/topics/input_{i}',
                    'output_table': f'dataset.table_{i}'
                }
            )
        
        logger.info("Pipeline activity simulation completed")
