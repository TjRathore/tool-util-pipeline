"""
Manual Override Dashboard
Interface for manual intervention, escalations, and system control
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

def render_manual_override():
    """Render the manual override dashboard"""
    
    st.title("⚙️ Manual Override Panel")
    
    # Get services from session state
    bigquery_service = st.session_state.bigquery_service
    monitoring_service = st.session_state.monitoring_service
    remediation_service = st.session_state.remediation_service
    
    # Tabs for different override functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Active Jobs", 
        "⚡ Quick Actions", 
        "🚨 Escalations", 
        "⚙️ System Control",
        "📝 Manual Remediation"
    ])
    
    with tab1:
        render_active_jobs(bigquery_service, remediation_service)
    
    with tab2:
        render_quick_actions(monitoring_service, remediation_service)
    
    with tab3:
        render_escalations(bigquery_service, monitoring_service)
    
    with tab4:
        render_system_control(monitoring_service, remediation_service)
    
    with tab5:
        render_manual_remediation(bigquery_service, remediation_service)

def render_active_jobs(bigquery_service, remediation_service):
    """Render active jobs management interface"""
    
    st.subheader("🔧 Active Jobs Management")
    
    # Get recent jobs
    all_jobs = bigquery_service.get_jobs(limit=50)
    
    if not all_jobs:
        st.info("No jobs found.")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "running", "failed", "completed"],
            index=0
        )
    
    with col2:
        job_type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(set(job['job_type'] for job in all_jobs if job['job_type'])),
            index=0
        )
    
    with col3:
        hours_filter = st.selectbox(
            "Time Range",
            ["All time", "Last hour", "Last 6 hours", "Last 24 hours"],
            index=2
        )
    
    # Apply filters
    filtered_jobs = all_jobs
    
    if status_filter != "All":
        filtered_jobs = [job for job in filtered_jobs if job['status'] == status_filter]
    
    if job_type_filter != "All":
        filtered_jobs = [job for job in filtered_jobs if job['job_type'] == job_type_filter]
    
    if hours_filter != "All time":
        hours_map = {"Last hour": 1, "Last 6 hours": 6, "Last 24 hours": 24}
        cutoff = datetime.now() - timedelta(hours=hours_map[hours_filter])
        filtered_jobs = [
            job for job in filtered_jobs 
            if datetime.fromisoformat(job['created_at'].replace('Z', '+00:00')) > cutoff
        ]
    
    st.write(f"**Found {len(filtered_jobs)} jobs matching filters**")
    
    # Jobs table with actions
    if filtered_jobs:
        df_jobs = pd.DataFrame(filtered_jobs)
        
        # Format the data for display
        df_display = df_jobs.copy()
        df_display['created_at'] = pd.to_datetime(df_display['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        df_display['job_name'] = df_display['job_name'].str.replace('_', ' ')
        df_display['job_type'] = df_display['job_type'].str.replace('_', ' ').str.title()
        df_display['status'] = df_display['status'].str.title()
        
        # Add status emoji
        status_emoji = {'Running': '🔄', 'Failed': '❌', 'Completed': '✅'}
        df_display['status'] = df_display['status'].apply(
            lambda x: f"{status_emoji.get(x, '❓')} {x}"
        )
        
        # Select columns to display
        display_columns = ['created_at', 'job_name', 'job_type', 'status']
        if 'error_type' in df_display.columns:
            df_display['error_type'] = df_display['error_type'].fillna('-').str.replace('_', ' ').str.title()
            display_columns.append('error_type')
        
        if 'auto_healed' in df_display.columns:
            df_display['auto_healed'] = df_display['auto_healed'].apply(
                lambda x: '🔧 Yes' if x else '-'
            )
            display_columns.append('auto_healed')
        
        # Rename columns for display
        column_names = {
            'created_at': 'Created',
            'job_name': 'Job Name',
            'job_type': 'Type',
            'status': 'Status',
            'error_type': 'Error Type',
            'auto_healed': 'Auto-Healed'
        }
        
        df_display = df_display[display_columns].rename(columns=column_names)
        
        # Display table
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Job actions
        st.subheader("Job Actions")
        
        # Select job for actions
        job_options = [f"{job['job_id']} - {job['job_name']}" for job in filtered_jobs]
        selected_job_option = st.selectbox("Select Job for Actions", job_options)
        
        if selected_job_option:
            selected_job_id = selected_job_option.split(' - ')[0]
            selected_job = next(job for job in filtered_jobs if job['job_id'] == selected_job_id)
            
            # Show job details
            with st.expander("Job Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Job ID:** {selected_job['job_id']}")
                    st.write(f"**Name:** {selected_job['job_name']}")
                    st.write(f"**Type:** {selected_job['job_type']}")
                    st.write(f"**Status:** {selected_job['status']}")
                
                with col2:
                    st.write(f"**Created:** {selected_job['created_at']}")
                    if selected_job.get('error_type'):
                        st.write(f"**Error Type:** {selected_job['error_type']}")
                    if selected_job.get('confidence_score'):
                        st.write(f"**Confidence:** {selected_job['confidence_score']:.3f}")
                
                if selected_job.get('error_message'):
                    st.write("**Error Message:**")
                    st.code(selected_job['error_message'])
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Retry Job", disabled=selected_job['status'] != 'failed'):
                    # Simulate job retry
                    success = remediation_service.simulate_remediation(
                        selected_job_id, 
                        'retry_with_exponential_backoff',
                        force_success=True
                    )
                    
                    if success['success']:
                        st.success(f"Job {selected_job_id} retry initiated successfully!")
                        # Update job status
                        bigquery_service.update_job(selected_job_id, {'status': 'running'})
                    else:
                        st.error(f"Failed to retry job {selected_job_id}")
                    
                    st.rerun()
            
            with col2:
                if st.button("⚡ Force Heal", disabled=selected_job['status'] != 'failed'):
                    # Force healing with high success rate
                    if selected_job.get('error_type'):
                        strategies = {
                            'permission_denied': 'restart_with_elevated_permissions',
                            'network_timeout': 'retry_with_exponential_backoff',
                            'schema_mismatch': 'update_schema_and_retry',
                            'resource_exhaustion': 'scale_compute_resources'
                        }
                        
                        strategy = strategies.get(selected_job['error_type'], 'retry_with_exponential_backoff')
                        
                        success = remediation_service.simulate_remediation(
                            selected_job_id,
                            strategy,
                            force_success=True
                        )
                        
                        if success['success']:
                            st.success(f"Job {selected_job_id} force healed successfully!")
                            bigquery_service.update_job(selected_job_id, {
                                'status': 'completed',
                                'auto_healed': True,
                                'healing_duration': success['duration']
                            })
                        else:
                            st.error(f"Failed to force heal job {selected_job_id}")
                        
                        st.rerun()
            
            with col3:
                if st.button("🚨 Escalate", disabled=selected_job['status'] != 'failed'):
                    # Mark job for escalation
                    bigquery_service.update_job(selected_job_id, {
                        'resolution': 'escalated_to_human'
                    })
                    
                    # Log escalation
                    bigquery_service.insert_log({
                        'job_id': selected_job_id,
                        'level': 'WARNING',
                        'message': 'Job manually escalated for human intervention',
                        'context': {'escalated_by': 'manual_override', 'timestamp': datetime.now().isoformat()}
                    })
                    
                    st.warning(f"Job {selected_job_id} escalated for human intervention")
                    st.rerun()
            
            with col4:
                if st.button("📋 View Logs"):
                    # Get job logs
                    logs = bigquery_service.get_logs(job_id=selected_job_id, limit=20)
                    
                    if logs:
                        st.write("**Recent Logs:**")
                        for log in logs:
                            timestamp = log['timestamp']
                            level = log['level']
                            message = log['message']
                            
                            level_colors = {
                                'INFO': 'blue',
                                'WARNING': 'orange', 
                                'ERROR': 'red'
                            }
                            
                            st.markdown(f"**{timestamp}** - :{level_colors.get(level, 'gray')}[{level}] {message}")
                    else:
                        st.info("No logs found for this job.")

def render_quick_actions(monitoring_service, remediation_service):
    """Render quick action buttons for common tasks"""
    
    st.subheader("⚡ Quick Actions")
    
    # Emergency actions
    st.write("**Emergency Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚨 Stop All Remediations", type="primary"):
            active_remediations = remediation_service.get_active_remediations()
            
            cancelled_count = 0
            for job_id in active_remediations.keys():
                if remediation_service.cancel_remediation(job_id):
                    cancelled_count += 1
            
            if cancelled_count > 0:
                st.success(f"Cancelled {cancelled_count} active remediations")
            else:
                st.info("No active remediations to cancel")
            
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause Monitoring"):
            if monitoring_service.is_monitoring:
                monitoring_service.stop_monitoring()
                st.warning("Monitoring service paused")
                st.rerun()
            else:
                st.info("Monitoring service is already stopped")
    
    with col3:
        if st.button("▶️ Resume Monitoring"):
            if not monitoring_service.is_monitoring:
                monitoring_service.start_monitoring()
                st.success("Monitoring service resumed")
                st.rerun()
            else:
                st.info("Monitoring service is already running")
    
    with col4:
        if st.button("🔄 Clear Event Queue"):
            # Clear the monitoring service queue
            with monitoring_service.queue_lock:
                queue_size = len(monitoring_service.event_queue)
                monitoring_service.event_queue.clear()
            
            st.success(f"Cleared {queue_size} events from queue")
            st.rerun()
    
    st.markdown("---")
    
    # Batch operations
    st.write("**Batch Operations:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔧 Retry All Failed Jobs"):
            from utils.data_generator import DataGenerator
            
            # Get all failed jobs
            failed_jobs = st.session_state.bigquery_service.get_jobs(limit=100)
            failed_jobs = [job for job in failed_jobs if job['status'] == 'failed']
            
            if failed_jobs:
                success_count = 0
                for job in failed_jobs[:10]:  # Limit to first 10 for demo
                    result = remediation_service.simulate_remediation(
                        job['job_id'],
                        'retry_with_exponential_backoff'
                    )
                    if result['success']:
                        success_count += 1
                        st.session_state.bigquery_service.update_job(job['job_id'], {
                            'status': 'completed',
                            'auto_healed': True
                        })
                
                st.success(f"Initiated retry for {len(failed_jobs[:10])} jobs. {success_count} successful.")
            else:
                st.info("No failed jobs found")
            
            st.rerun()
    
    with col2:
        if st.button("📊 Generate Test Data"):
            # Generate test pipeline events
            from utils.data_generator import DataGenerator
            generator = DataGenerator()
            
            events = generator.generate_pipeline_events(count=10, failure_rate=0.4)
            
            for event in events:
                monitoring_service.inject_event(event)
            
            st.success("Generated 10 test pipeline events")
            st.rerun()
    
    st.markdown("---")
    
    # System maintenance
    st.write("**System Maintenance:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Cleanup Old Logs"):
            # In a real system, this would clean up old log entries
            st.info("Log cleanup would be performed (simulated)")
    
    with col2:
        if st.button("📈 Retrain ML Model"):
            # Simulate model retraining
            st.info("ML model retraining initiated (simulated)")
    
    with col3:
        if st.button("💾 Backup Database"):
            # Simulate database backup
            st.info("Database backup initiated (simulated)")

def render_escalations(bigquery_service, monitoring_service):
    """Render escalation management interface"""
    
    st.subheader("🚨 Escalation Management")
    
    # Get jobs that need escalation (high-severity failures, repeated failures, etc.)
    all_jobs = bigquery_service.get_jobs(limit=100)
    
    # Define escalation criteria
    escalation_candidates = []
    for job in all_jobs:
        if job['status'] == 'failed':
            # Check retry count
            if job.get('retry_count', 0) >= 3:
                escalation_candidates.append({
                    **job,
                    'escalation_reason': 'Multiple retry failures'
                })
            # Check confidence score
            elif job.get('confidence_score', 1.0) < 0.5:
                escalation_candidates.append({
                    **job,
                    'escalation_reason': 'Low ML confidence'
                })
            # Check error type
            elif job.get('error_type') == 'unknown_error':
                escalation_candidates.append({
                    **job,
                    'escalation_reason': 'Unknown error type'
                })
    
    if not escalation_candidates:
        st.success("✅ No jobs require escalation at this time")
        return
    
    st.warning(f"⚠️ {len(escalation_candidates)} jobs require attention")
    
    # Escalation queue
    st.write("**Jobs Requiring Escalation:**")
    
    for i, job in enumerate(escalation_candidates):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{job['job_name']}**")
                st.caption(f"Reason: {job['escalation_reason']}")
                if job.get('error_message'):
                    st.code(job['error_message'][:100] + "..." if len(job['error_message']) > 100 else job['error_message'])
            
            with col2:
                st.write("**Type:**")
                st.write(job['job_type'].replace('_', ' ').title())
            
            with col3:
                st.write("**Created:**")
                st.write(pd.to_datetime(job['created_at']).strftime('%m/%d %H:%M'))
            
            with col4:
                escalation_actions = st.columns(2)
                
                with escalation_actions[0]:
                    if st.button(f"✅ Resolve", key=f"resolve_{i}"):
                        # Mark as resolved
                        bigquery_service.update_job(job['job_id'], {
                            'status': 'completed',
                            'resolution': 'manual_resolution'
                        })
                        
                        # Log resolution
                        bigquery_service.insert_log({
                            'job_id': job['job_id'],
                            'level': 'INFO',
                            'message': f'Job manually resolved: {job["escalation_reason"]}',
                            'context': {'resolved_by': 'manual_override'}
                        })
                        
                        st.success("Job marked as resolved")
                        st.rerun()
                
                with escalation_actions[1]:
                    if st.button(f"🔄 Retry", key=f"retry_{i}"):
                        # Attempt remediation
                        remediation_service.simulate_remediation(
                            job['job_id'],
                            'retry_with_exponential_backoff',
                            force_success=True
                        )
                        
                        st.success("Retry initiated")
                        st.rerun()
            
            st.markdown("---")
    
    # Escalation workflow
    st.subheader("Escalation Workflow")
    
    with st.expander("Create New Escalation"):
        st.write("Manually escalate a specific job or issue:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            escalation_type = st.selectbox(
                "Escalation Type",
                ["Critical Pipeline Failure", "Resource Exhaustion", "Security Issue", "Data Quality Issue", "Other"]
            )
            
            priority = st.selectbox(
                "Priority",
                ["High", "Medium", "Low"]
            )
        
        with col2:
            affected_systems = st.multiselect(
                "Affected Systems",
                ["Dataflow", "Composer", "BigQuery", "Dataproc", "Cloud Functions"]
            )
        
        description = st.text_area(
            "Description",
            placeholder="Describe the issue and any steps taken..."
        )
        
        if st.button("Create Escalation"):
            if description:
                # Create escalation record
                escalation_id = f"escalation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Log the escalation
                bigquery_service.insert_log({
                    'job_id': escalation_id,
                    'level': 'WARNING',
                    'message': f'Manual escalation created: {escalation_type}',
                    'context': {
                        'escalation_type': escalation_type,
                        'priority': priority,
                        'affected_systems': affected_systems,
                        'description': description,
                        'created_by': 'manual_override'
                    }
                })
                
                st.success(f"Escalation {escalation_id} created successfully")
                st.balloons()
            else:
                st.error("Please provide a description")

def render_system_control(monitoring_service, remediation_service):
    """Render system control panel"""
    
    st.subheader("⚙️ System Control Panel")
    
    # System status
    monitoring_stats = monitoring_service.get_monitoring_stats()
    remediation_stats = remediation_service.get_remediation_stats()
    
    st.write("**Current System Status:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "🟢 Running" if monitoring_stats['is_monitoring'] else "🔴 Stopped"
        st.metric("Monitoring Service", status)
        
        if monitoring_stats['is_monitoring']:
            if st.button("⏹️ Stop Monitoring"):
                monitoring_service.stop_monitoring()
                st.success("Monitoring stopped")
                st.rerun()
        else:
            if st.button("▶️ Start Monitoring"):
                monitoring_service.start_monitoring()
                st.success("Monitoring started") 
                st.rerun()
    
    with col2:
        st.metric("Active Remediations", remediation_stats['active_remediations'])
        
        if remediation_stats['active_remediations'] > 0:
            if st.button("⏸️ Pause All Remediations"):
                # Cancel all active remediations
                active = remediation_service.get_active_remediations()
                for job_id in active.keys():
                    remediation_service.cancel_remediation(job_id)
                st.success("All remediations paused")
                st.rerun()
    
    with col3:
        queue_size = monitoring_stats['queue_size']
        st.metric("Event Queue Size", queue_size)
        
        if queue_size > 0:
            if st.button("🗑️ Clear Queue"):
                with monitoring_service.queue_lock:
                    monitoring_service.event_queue.clear()
                st.success("Event queue cleared")
                st.rerun()
    
    st.markdown("---")
    
    # Configuration controls
    st.subheader("Configuration Controls")
    
    with st.expander("ML Model Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.75, 0.05,
                help="Minimum confidence score required for auto-remediation"
            )
            
            max_retries = st.slider(
                "Max Retry Attempts", 
                1, 10, 3,
                help="Maximum number of retry attempts for failed jobs"
            )
        
        with col2:
            auto_healing_enabled = st.checkbox(
                "Auto-healing Enabled", 
                value=True,
                help="Enable automatic remediation of failed jobs"
            )
            
            real_time_monitoring = st.checkbox(
                "Real-time Monitoring",
                value=True,
                help="Enable real-time pipeline event monitoring"
            )
        
        if st.button("Apply Configuration Changes"):
            # Update configuration (in a real system, this would update the config service)
            st.success("Configuration updated successfully")
    
    with st.expander("Alert Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_on_failures = st.checkbox("Alert on Failures", value=True)
            alert_on_healing_failures = st.checkbox("Alert on Healing Failures", value=True)
        
        with col2:
            alert_email = st.text_input("Alert Email", placeholder="admin@company.com")
            alert_slack_webhook = st.text_input("Slack Webhook URL", placeholder="https://hooks.slack.com/...")
        
        if st.button("Update Alert Settings"):
            st.success("Alert settings updated")
    
    # System maintenance
    st.markdown("---")
    st.subheader("System Maintenance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Database Cleanup"):
            st.info("Database cleanup initiated (simulation)")
            st.success("Cleanup completed - removed old logs and events")
    
    with col2:
        if st.button("📊 Export Metrics"):
            # Generate metrics export
            metrics_data = {
                'monitoring_stats': monitoring_stats,
                'remediation_stats': remediation_stats,
                'export_timestamp': datetime.now().isoformat()
            }
            
            st.download_button(
                label="📥 Download Metrics JSON",
                data=json.dumps(metrics_data, indent=2),
                file_name=f"pipeline_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 System Health Check"):
            st.info("Performing system health check...")
            
            # Simulate health check
            health_results = {
                "Monitoring Service": "✅ Healthy",
                "ML Model": "✅ Loaded and functional", 
                "Database Connection": "✅ Connected",
                "Event Processing": "✅ Normal",
                "Memory Usage": "✅ Normal (45%)",
                "Disk Space": "✅ Sufficient (78% free)"
            }
            
            for component, status in health_results.items():
                st.write(f"**{component}:** {status}")

def render_manual_remediation(bigquery_service, remediation_service):
    """Render manual remediation interface"""
    
    st.subheader("📝 Manual Remediation")
    
    st.write("Create custom remediation strategies for specific scenarios:")
    
    # Get failed jobs for manual remediation
    failed_jobs = bigquery_service.get_jobs(limit=50)
    failed_jobs = [job for job in failed_jobs if job['status'] == 'failed']
    
    if not failed_jobs:
        st.info("No failed jobs available for manual remediation.")
        return
    
    # Job selection
    job_options = [f"{job['job_id']} - {job['job_name']}" for job in failed_jobs]
    selected_job_option = st.selectbox("Select Job for Manual Remediation", job_options)
    
    if selected_job_option:
        selected_job_id = selected_job_option.split(' - ')[0]
        selected_job = next(job for job in failed_jobs if job['job_id'] == selected_job_id)
        
        # Show job details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Job Information:**")
            st.write(f"ID: {selected_job['job_id']}")
            st.write(f"Name: {selected_job['job_name']}")
            st.write(f"Type: {selected_job['job_type']}")
            st.write(f"Status: {selected_job['status']}")
            
            if selected_job.get('error_type'):
                st.write(f"Error Type: {selected_job['error_type']}")
            
            if selected_job.get('confidence_score'):
                st.write(f"ML Confidence: {selected_job['confidence_score']:.3f}")
        
        with col2:
            st.write("**Error Details:**")
            if selected_job.get('error_message'):
                st.code(selected_job['error_message'])
            else:
                st.write("No error message available")
        
        # Manual remediation options
        st.subheader("Remediation Strategy")
        
        # Pre-defined strategies
        strategy_options = [
            "retry_with_exponential_backoff",
            "restart_with_elevated_permissions", 
            "update_schema_and_retry",
            "scale_compute_resources",
            "run_deduplication_script",
            "install_dependencies_and_retry",
            "update_configuration_and_retry",
            "request_quota_increase",
            "refresh_credentials_and_retry",
            "run_data_validation_and_cleanup",
            "escalate_to_human",
            "custom_strategy"
        ]
        
        selected_strategy = st.selectbox(
            "Choose Remediation Strategy",
            strategy_options,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Custom strategy input
        if selected_strategy == "custom_strategy":
            custom_strategy_name = st.text_input("Custom Strategy Name")
            custom_strategy_description = st.text_area(
                "Strategy Description",
                placeholder="Describe the steps for this custom remediation strategy..."
            )
        
        # Additional parameters
        col1, col2 = st.columns(2)
        
        with col1:
            max_attempts = st.slider("Max Retry Attempts", 1, 5, 1)
            timeout_minutes = st.slider("Timeout (minutes)", 1, 60, 10)
        
        with col2:
            force_success = st.checkbox(
                "Force Success (for testing)", 
                help="Forces the remediation to succeed for testing purposes"
            )
            
            notify_on_completion = st.checkbox("Send Notification on Completion", value=True)
        
        # Execute remediation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Execute Remediation", type="primary"):
                if selected_strategy != "custom_strategy" or (custom_strategy_name and custom_strategy_description):
                    # Execute the remediation
                    strategy_name = custom_strategy_name if selected_strategy == "custom_strategy" else selected_strategy
                    
                    with st.spinner("Executing remediation..."):
                        result = remediation_service.simulate_remediation(
                            selected_job_id,
                            strategy_name,
                            force_success=force_success
                        )
                    
                    if result['success']:
                        st.success(f"✅ Remediation successful! Duration: {result['duration']:.2f}s")
                        
                        # Update job status
                        updates = {
                            'status': 'completed',
                            'auto_healed': True,
                            'healing_duration': result['duration'],
                            'resolution': strategy_name
                        }
                        bigquery_service.update_job(selected_job_id, updates)
                        
                        # Log manual remediation
                        bigquery_service.insert_log({
                            'job_id': selected_job_id,
                            'level': 'INFO',
                            'message': f'Manual remediation successful: {strategy_name}',
                            'context': {
                                'strategy': strategy_name,
                                'duration': result['duration'],
                                'max_attempts': max_attempts,
                                'manual_override': True
                            }
                        })
                        
                        if notify_on_completion:
                            st.balloons()
                        
                    else:
                        st.error(f"❌ Remediation failed: {result.get('details', 'Unknown error')}")
                        
                        # Log failed remediation
                        bigquery_service.insert_log({
                            'job_id': selected_job_id,
                            'level': 'ERROR',
                            'message': f'Manual remediation failed: {strategy_name}',
                            'context': {
                                'strategy': strategy_name,
                                'error': result.get('details', 'Unknown error'),
                                'manual_override': True
                            }
                        })
                    
                    st.rerun()
                else:
                    st.error("Please provide a name and description for the custom strategy")
        
        with col2:
            if st.button("📋 Dry Run"):
                st.info("Dry run simulation:")
                st.write(f"Strategy: {selected_strategy.replace('_', ' ').title()}")
                st.write(f"Job: {selected_job['job_name']}")
                st.write(f"Max Attempts: {max_attempts}")
                st.write(f"Timeout: {timeout_minutes} minutes")
                st.write("✅ Remediation would be executed with these parameters")
        
        with col3:
            if st.button("📊 View Similar Cases"):
                # Find similar failed jobs
                similar_jobs = []
                if selected_job.get('error_type'):
                    similar_jobs = [
                        job for job in failed_jobs 
                        if job.get('error_type') == selected_job['error_type'] 
                        and job['job_id'] != selected_job_id
                    ]
                
                if similar_jobs:
                    st.write(f"Found {len(similar_jobs)} similar cases:")
                    for job in similar_jobs[:3]:
                        st.write(f"• {job['job_name']} ({job['job_id']})")
                else:
                    st.write("No similar cases found")
        
        # Remediation history for this job
        st.subheader("Remediation History")
        
        healing_actions = bigquery_service.get_healing_actions(job_id=selected_job_id)
        
        if healing_actions:
            st.write(f"**Previous remediation attempts for this job:**")
            
            for action in healing_actions:
                status_emoji = "✅" if action['success'] else "❌"
                st.write(f"{status_emoji} {action['action_type'].replace('_', ' ').title()} - {action['started_at']}")
        else:
            st.info("No previous remediation attempts found for this job")
