"""
Main Dashboard
Real-time pipeline monitoring and health metrics display
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

def render_main_dashboard():
    """Render the main monitoring dashboard"""
    
    st.title("🔧 AI Pipeline Monitor - Main Dashboard")
    
    # Get services from session state
    bigquery_service = st.session_state.bigquery_service
    monitoring_service = st.session_state.monitoring_service
    remediation_service = st.session_state.remediation_service
    
    # Start monitoring if not already started
    if not monitoring_service.is_monitoring:
        monitoring_service.set_services(bigquery_service, remediation_service)
        monitoring_service.start_monitoring()
    
    # Auto-refresh mechanism
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Refresh every 30 seconds
    refresh_interval = 30
    if (datetime.now() - st.session_state.last_refresh).seconds > refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Header controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown("**Real-time Pipeline Health Monitoring**")
    
    with col2:
        if st.button("🔄 Refresh", help="Refresh dashboard data"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True, help="Automatically refresh every 30 seconds")
    
    with col4:
        st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")
    
    # Key Metrics Row
    render_key_metrics(bigquery_service, monitoring_service, remediation_service)
    
    st.markdown("---")
    
    # Pending Recommendations Section
    render_pending_recommendations(bigquery_service, monitoring_service)
    
    st.markdown("---")
    
    # Main content columns
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Recent Events
        render_recent_events(monitoring_service)
        
        # Pipeline Status Overview
        render_pipeline_status(bigquery_service)
    
    with right_col:
        # System Health
        render_system_health(monitoring_service, remediation_service)
        
        # Error Classification Stats
        render_error_stats(bigquery_service)
    
    # Bottom row - Charts
    st.markdown("---")
    render_charts_row(bigquery_service)
    
    # Simulation Controls (for testing)
    render_simulation_controls(monitoring_service)

def render_key_metrics(bigquery_service, monitoring_service, remediation_service):
    """Render key performance metrics at the top"""
    
    # Get dashboard metrics
    metrics = bigquery_service.get_dashboard_metrics()
    monitoring_stats = monitoring_service.get_monitoring_stats()
    remediation_stats = remediation_service.get_remediation_stats()
    
    # Create 5 columns for metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_jobs = metrics.get('total_jobs', 0)
        recent_jobs = metrics.get('recent_jobs', 0)
        delta = f"+{recent_jobs} (24h)" if recent_jobs > 0 else "No recent jobs"
        st.metric(
            label="Total Jobs",
            value=f"{total_jobs:,}",
            delta=delta,
            help="Total number of pipeline jobs processed"
        )
    
    with col2:
        success_rate = metrics.get('success_rate', 100.0)
        delta_color = "normal" if success_rate >= 95 else "inverse"
        st.metric(
            label="Success Rate",
            value=f"{success_rate:.1f}%",
            delta=f"{success_rate - 95:.1f}%" if success_rate != 100 else None,
            delta_color=delta_color,
            help="Percentage of successful pipeline executions"
        )
    
    with col3:
        healing_rate = metrics.get('healing_rate', 0.0)
        auto_healed = metrics.get('auto_healed_jobs', 0)
        st.metric(
            label="Auto-Healing Rate",
            value=f"{healing_rate:.1f}%",
            delta=f"{auto_healed} healed",
            help="Percentage of failed jobs automatically healed"
        )
    
    with col4:
        avg_duration = metrics.get('avg_healing_duration', 0.0)
        active_remediations = remediation_stats.get('active_remediations', 0)
        st.metric(
            label="Avg Healing Time",
            value=f"{avg_duration:.1f}s",
            delta=f"{active_remediations} active" if active_remediations > 0 else "No active healing",
            help="Average time to heal failed pipelines"
        )
    
    with col5:
        detection_rate = monitoring_stats.get('error_detection_rate', 0.0)
        detected_errors = monitoring_stats.get('detected_errors', 0)
        st.metric(
            label="Error Detection",
            value=f"{detection_rate:.1f}%",
            delta=f"{detected_errors} detected",
            help="Rate of error detection in monitored events"
        )

def render_recent_events(monitoring_service):
    """Render recent pipeline events"""
    
    st.subheader("📋 Recent Pipeline Events")
    
    # Get recent events
    events = monitoring_service.get_recent_events(limit=20)
    
    if not events:
        st.info("No pipeline events found. Events will appear here as pipelines are monitored.")
        return
    
    # Convert to DataFrame for better display
    df_events = pd.DataFrame(events)
    
    # Add status emoji and format
    status_emoji = {
        'completed': '✅',
        'running': '🔄',
        'failed': '❌'
    }
    
    df_events['Status'] = df_events['status'].apply(lambda x: f"{status_emoji.get(x, '❓')} {x.title()}")
    df_events['Job Name'] = df_events['job_name'].str.replace('_', ' ').str.title()
    df_events['Type'] = df_events['job_type'].str.replace('_', ' ').str.title()
    df_events['Time'] = pd.to_datetime(df_events['timestamp']).dt.strftime('%H:%M:%S')
    
    # Display table
    display_columns = ['Time', 'Job Name', 'Type', 'Status']
    
    if 'error_type' in df_events.columns:
        df_events['Error Type'] = df_events['error_type'].fillna('-').str.replace('_', ' ').str.title()
        display_columns.append('Error Type')
    
    if 'auto_healed' in df_events.columns:
        df_events['Auto-Healed'] = df_events['auto_healed'].apply(lambda x: '🔧 Yes' if x else '-')
        display_columns.append('Auto-Healed')
    
    st.dataframe(
        df_events[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Show details for failed jobs
    failed_events = [e for e in events if e['status'] == 'failed']
    if failed_events:
        with st.expander(f"Failed Job Details ({len(failed_events)} jobs)"):
            for event in failed_events[:5]:  # Show first 5
                st.write(f"**{event['job_name']}** ({event['job_type']})")
                if event.get('error_message'):
                    st.code(event['error_message'], language=None)
                if event.get('error_type'):
                    st.caption(f"Classified as: {event['error_type'].replace('_', ' ').title()}")
                st.markdown("---")

def render_pipeline_status(bigquery_service):
    """Render pipeline status overview"""
    
    st.subheader("🚀 Pipeline Status Overview")
    
    # Get jobs by status
    all_jobs = bigquery_service.get_jobs(limit=100)
    
    if not all_jobs:
        st.info("No pipeline jobs found.")
        return
    
    df_jobs = pd.DataFrame(all_jobs)
    
    # Status distribution
    status_counts = df_jobs['status'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status pie chart
        fig_pie = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Job Status Distribution",
            color_discrete_map={
                'completed': '#28a745',
                'failed': '#dc3545',
                'running': '#ffc107'
            }
        )
        fig_pie.update_layout(showlegend=True, height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Job types
        if 'job_type' in df_jobs.columns:
            type_counts = df_jobs['job_type'].value_counts()
            
            fig_bar = px.bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation='h',
                title="Jobs by Type",
                labels={'x': 'Count', 'y': 'Job Type'}
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

def render_system_health(monitoring_service, remediation_service):
    """Render system health indicators"""
    
    st.subheader("🏥 System Health")
    
    # Monitoring service status
    monitoring_stats = monitoring_service.get_monitoring_stats()
    
    # Health indicators
    health_items = [
        ("Monitoring Service", "🟢 Active" if monitoring_stats['is_monitoring'] else "🔴 Inactive"),
        ("Events Processed", f"{monitoring_stats['processed_events']:,}"),
        ("Errors Detected", f"{monitoring_stats['detected_errors']:,}"),
        ("Healing Attempts", f"{monitoring_stats['healing_attempts']:,}"),
    ]
    
    for label, value in health_items:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**{label}:**")
        with col2:
            st.write(value)
    
    # Active remediations
    active_remediations = remediation_service.get_active_remediations()
    
    if active_remediations:
        st.write("**Active Remediations:**")
        for job_id, remediation in active_remediations.items():
            st.write(f"• {job_id}: {remediation['strategy']}")
    else:
        st.write("**Active Remediations:** None")
    
    # Performance metrics
    st.write("**Performance Metrics:**")
    detection_rate = monitoring_stats.get('error_detection_rate', 0)
    healing_rate = monitoring_stats.get('healing_rate', 0)
    
    # Progress bars for rates
    st.write("Error Detection Rate:")
    st.progress(detection_rate / 100)
    st.write(f"{detection_rate:.1f}%")
    
    st.write("Healing Success Rate:")
    st.progress(healing_rate / 100)
    st.write(f"{healing_rate:.1f}%")

def render_error_stats(bigquery_service):
    """Render error classification statistics"""
    
    st.subheader("🔍 Error Classification")
    
    metrics = bigquery_service.get_dashboard_metrics()
    error_distribution = metrics.get('error_distribution', [])
    
    if not error_distribution:
        st.info("No error data available.")
        return
    
    # Error type chart
    if error_distribution:
        df_errors = pd.DataFrame(error_distribution)
        df_errors['error_type'] = df_errors['error_type'].str.replace('_', ' ').str.title()
        
        fig = px.bar(
            df_errors,
            x='count',
            y='error_type',
            orientation='h',
            title="Error Types",
            labels={'count': 'Count', 'error_type': 'Error Type'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Error patterns
    error_patterns = bigquery_service.get_error_patterns(limit=10)
    
    if error_patterns:
        st.write("**Top Error Patterns:**")
        df_patterns = pd.DataFrame(error_patterns)
        
        for _, pattern in df_patterns.head(5).iterrows():
            success_rate = pattern['success_rate'] * 100
            st.write(f"**{pattern['error_type'].replace('_', ' ').title()}**")
            st.write(f"Success Rate: {success_rate:.1f}% | Frequency: {pattern['frequency']}")
            st.progress(success_rate / 100)
            st.markdown("---")

def render_charts_row(bigquery_service):
    """Render charts showing trends and analytics"""
    
    st.subheader("📊 Trends & Analytics")
    
    # Get trend data
    trend_data = bigquery_service.get_trend_data(days=7)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily job statistics
        daily_stats = trend_data.get('daily_stats', [])
        
        if daily_stats:
            df_daily = pd.DataFrame(daily_stats)
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_daily['date'],
                y=df_daily['total_jobs'],
                mode='lines+markers',
                name='Total Jobs',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_daily['date'],
                y=df_daily['failed_jobs'],
                mode='lines+markers',
                name='Failed Jobs',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=df_daily['date'],
                y=df_daily['healed_jobs'],
                mode='lines+markers',
                name='Healed Jobs',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title="Daily Job Statistics (7 days)",
                xaxis_title="Date",
                yaxis_title="Number of Jobs",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily statistics available.")
    
    with col2:
        # Error types over time
        error_types = trend_data.get('error_types', [])
        
        if error_types:
            df_errors = pd.DataFrame(error_types)
            df_errors['error_type'] = df_errors['error_type'].str.replace('_', ' ').str.title()
            
            fig = px.bar(
                df_errors,
                x='count',
                y='error_type',
                orientation='h',
                title="Error Types (7 days)",
                labels={'count': 'Count', 'error_type': 'Error Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error type data available.")

def render_simulation_controls(monitoring_service):
    """Render simulation controls for testing"""
    
    with st.expander("🧪 Simulation Controls (Testing)"):
        st.write("Generate test events to see the system in action:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Simulate Network Timeout"):
                job_id = monitoring_service.simulate_pipeline_failure(
                    job_type="dataflow",
                    error_type="network_timeout"
                )
                st.success(f"Simulated network timeout error for job: {job_id}")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("Simulate Permission Error"):
                job_id = monitoring_service.simulate_pipeline_failure(
                    job_type="composer_dag",
                    error_type="permission_denied"
                )
                st.success(f"Simulated permission error for job: {job_id}")
                time.sleep(1)
                st.rerun()
        
        with col3:
            if st.button("Simulate Schema Mismatch"):
                job_id = monitoring_service.simulate_pipeline_failure(
                    job_type="bigquery_load",
                    error_type="schema_mismatch"
                )
                st.success(f"Simulated schema mismatch for job: {job_id}")
                time.sleep(1)
                st.rerun()
        
        # Bulk simulation
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            num_events = st.slider("Number of events to generate", 1, 20, 5)
        
        with col2:
            failure_rate = st.slider("Failure rate", 0.0, 1.0, 0.3, 0.1)
        
        if st.button("Generate Bulk Events"):
            from utils.data_generator import DataGenerator
            generator = DataGenerator()
            
            events = generator.generate_pipeline_events(
                count=num_events,
                failure_rate=failure_rate
            )
            
            for event in events:
                monitoring_service.inject_event(event)
            
            st.success(f"Generated {num_events} events with {failure_rate*100:.0f}% failure rate")
            time.sleep(1)
            st.rerun()

def render_pending_recommendations(bigquery_service, monitoring_service):
    """Render pending remediation recommendations section"""
    
    st.subheader("🎯 Pending Remediation Recommendations")
    
    # Get pending recommendations
    pending_recommendations = bigquery_service.get_pending_recommendations()
    
    if not pending_recommendations:
        st.info("No pending recommendations at this time. Failed jobs will appear here for manual approval.")
        return
    
    st.write(f"**{len(pending_recommendations)} recommendation(s) awaiting your approval:**")
    
    for i, rec in enumerate(pending_recommendations):
        with st.container():
            st.markdown("---")
            
            # Create columns for recommendation display
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                # Job and error details
                st.markdown(f"**Job:** `{rec['job_name']}`")
                st.markdown(f"**Error Type:** {rec['error_type'].replace('_', ' ').title()}")
                st.markdown(f"**RAG Recommendation:** {rec['rag_recommendation'] or 'Standard remediation strategy'}")
                
            with col2:
                # Strategy and confidence
                st.markdown(f"**Proposed Strategy:** {rec['mapped_strategy'].replace('_', ' ').title()}")
                confidence_color = "🟢" if rec['confidence_score'] > 0.8 else "🟡" if rec['confidence_score'] > 0.6 else "🔴"
                st.markdown(f"**Confidence:** {confidence_color} {rec['confidence_score']:.1%}")
                st.markdown(f"**Created:** {rec['created_at'][:16]}")
                
            with col3:
                # Approve button
                approve_key = f"approve_{rec['recommendation_id']}"
                if st.button("✅ Approve", key=approve_key, type="primary"):
                    with st.spinner("Executing remediation..."):
                        result = monitoring_service.execute_approved_recommendation(
                            rec['recommendation_id'],
                            approved_by="dashboard_user"
                        )
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")
                        
                        time.sleep(2)
                        st.rerun()
                        
            with col4:
                # Reject button
                reject_key = f"reject_{rec['recommendation_id']}"
                if st.button("❌ Reject", key=reject_key):
                    result = monitoring_service.reject_recommendation(
                        rec['recommendation_id'],
                        rejected_by="dashboard_user"
                    )
                    
                    if result['success']:
                        st.warning(f"Recommendation rejected: {result['message']}")
                    else:
                        st.error(f"Error rejecting: {result.get('error', 'Unknown error')}")
                    
                    time.sleep(1)
                    st.rerun()
            
            # Additional details in expandable section
            with st.expander(f"📋 Details for {rec['job_id']}", expanded=False):
                details_col1, details_col2 = st.columns(2)
                
                with details_col1:
                    st.markdown(f"""
                    **Job Details:**
                    - **Job ID:** `{rec['job_id']}`
                    - **Job Type:** {rec['job_type']}
                    - **Error Category:** {rec['error_type']}
                    """)
                
                with details_col2:
                    st.markdown(f"""
                    **Recommendation Details:**
                    - **Recommendation ID:** `{rec['recommendation_id']}`
                    - **Strategy:** {rec['mapped_strategy']}
                    - **Confidence Score:** {rec['confidence_score']:.3f}
                    """)
    
    # Summary info
    st.markdown("---")
    st.info(f"""
    💡 **Approval Workflow:** Review each recommendation above and click **Approve** to execute the suggested remediation strategy, 
    or **Reject** if you believe the recommendation is not appropriate. All approved remediations will be executed immediately 
    and the results will be logged for future ML model training.
    """)  
        
    # Bulk actions
    if len(pending_recommendations) > 1:
        st.markdown("### Bulk Actions")
        bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
        
        with bulk_col1:
            if st.button("✅ Approve All High Confidence", help="Approve all recommendations with >80% confidence"):
                high_confidence_recs = [r for r in pending_recommendations if r['confidence_score'] > 0.8]
                if high_confidence_recs:
                    approved_count = 0
                    for rec in high_confidence_recs:
                        result = monitoring_service.execute_approved_recommendation(
                            rec['recommendation_id'],
                            approved_by="dashboard_user_bulk"
                        )
                        if result['success']:
                            approved_count += 1
                    
                    st.success(f"✅ Bulk approved {approved_count} high-confidence recommendations")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("No high-confidence recommendations found")
        
        with bulk_col2:
            if st.button("❌ Reject All Low Confidence", help="Reject all recommendations with <60% confidence"):
                low_confidence_recs = [r for r in pending_recommendations if r['confidence_score'] < 0.6]
                if low_confidence_recs:
                    rejected_count = 0
                    for rec in low_confidence_recs:
                        result = monitoring_service.reject_recommendation(
                            rec['recommendation_id'],
                            rejected_by="dashboard_user_bulk"
                        )
                        if result['success']:
                            rejected_count += 1
                    
                    st.warning(f"❌ Bulk rejected {rejected_count} low-confidence recommendations")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.info("No low-confidence recommendations found")
        
        with bulk_col3:
            if st.button("🔄 Refresh Recommendations", help="Reload pending recommendations"):
                st.rerun()
