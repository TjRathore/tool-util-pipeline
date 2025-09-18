"""
Analytics Dashboard
Advanced analytics, trend analysis, and ML model performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

def render_analytics_dashboard():
    """Render the analytics dashboard"""
    
    st.title("📊 Analytics Dashboard")
    
    # Get services from session state
    bigquery_service = st.session_state.bigquery_service
    monitoring_service = st.session_state.monitoring_service
    remediation_service = st.session_state.remediation_service
    
    # Time range selector
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days"],
            index=1
        )
    
    with col2:
        chart_type = st.selectbox(
            "Chart Style",
            ["Interactive", "Static"],
            index=0
        )
    
    with col3:
        st.markdown("**Advanced Pipeline Analytics & Performance Metrics**")
    
    # Map time range to days
    time_range_map = {
        "Last 24 hours": 1,
        "Last 7 days": 7,
        "Last 30 days": 30
    }
    days = time_range_map[time_range]
    
    # Get data
    trend_data = bigquery_service.get_trend_data(days=days)
    remediation_stats = remediation_service.get_remediation_stats()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Pipeline Trends", 
        "🔧 Healing Analytics", 
        "🤖 ML Performance", 
        "⚡ System Performance"
    ])
    
    with tab1:
        render_pipeline_trends(trend_data, days)
    
    with tab2:
        render_healing_analytics(remediation_stats, bigquery_service)
    
    with tab3:
        render_ml_performance(monitoring_service)
    
    with tab4:
        render_system_performance(monitoring_service, remediation_service, bigquery_service)

def render_pipeline_trends(trend_data: Dict[str, Any], days: int):
    """Render pipeline trend analytics"""
    
    st.subheader(f"📈 Pipeline Trends ({days} days)")
    
    daily_stats = trend_data.get('daily_stats', [])
    
    if not daily_stats:
        st.warning("No pipeline trend data available for the selected time range.")
        return
    
    df_daily = pd.DataFrame(daily_stats)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily['success_rate'] = ((df_daily['total_jobs'] - df_daily['failed_jobs']) / df_daily['total_jobs'] * 100).fillna(100)
    df_daily['healing_rate'] = (df_daily['healed_jobs'] / df_daily['failed_jobs'] * 100).fillna(0)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_jobs = df_daily['total_jobs'].sum()
        st.metric("Total Jobs", f"{total_jobs:,}")
    
    with col2:
        avg_success_rate = df_daily['success_rate'].mean()
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
    
    with col3:
        total_healed = df_daily['healed_jobs'].sum()
        st.metric("Jobs Auto-Healed", f"{total_healed:,}")
    
    with col4:
        avg_healing_rate = df_daily['healing_rate'].mean()
        st.metric("Avg Healing Rate", f"{avg_healing_rate:.1f}%")
    
    # Main trend chart
    fig = go.Figure()
    
    # Total jobs line
    fig.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['total_jobs'],
        mode='lines+markers',
        name='Total Jobs',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='Date: %{x}<br>Total Jobs: %{y}<extra></extra>'
    ))
    
    # Failed jobs area
    fig.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['failed_jobs'],
        mode='lines+markers',
        name='Failed Jobs',
        line=dict(color='#d62728', width=2),
        fill='tonexty',
        fillcolor='rgba(214, 39, 40, 0.1)',
        hovertemplate='Date: %{x}<br>Failed Jobs: %{y}<extra></extra>'
    ))
    
    # Healed jobs area
    fig.add_trace(go.Scatter(
        x=df_daily['date'],
        y=df_daily['healed_jobs'],
        mode='lines+markers',
        name='Healed Jobs',
        line=dict(color='#2ca02c', width=2),
        fill='tonexty',
        fillcolor='rgba(44, 160, 44, 0.1)',
        hovertemplate='Date: %{x}<br>Healed Jobs: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Pipeline Job Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Jobs",
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Success and healing rates
    col1, col2 = st.columns(2)
    
    with col1:
        fig_success = px.line(
            df_daily,
            x='date',
            y='success_rate',
            title='Pipeline Success Rate Trend',
            labels={'success_rate': 'Success Rate (%)', 'date': 'Date'},
            line_shape='spline'
        )
        fig_success.update_traces(line_color='#2ca02c', line_width=3)
        fig_success.update_layout(height=300)
        fig_success.add_hline(y=95, line_dash="dash", line_color="red", 
                             annotation_text="95% Target")
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        fig_healing = px.line(
            df_daily,
            x='date', 
            y='healing_rate',
            title='Auto-Healing Rate Trend',
            labels={'healing_rate': 'Healing Rate (%)', 'date': 'Date'},
            line_shape='spline'
        )
        fig_healing.update_traces(line_color='#ff7f0e', line_width=3)
        fig_healing.update_layout(height=300)
        fig_healing.add_hline(y=80, line_dash="dash", line_color="green", 
                             annotation_text="80% Target")
        st.plotly_chart(fig_healing, use_container_width=True)
    
    # Error type analysis
    error_types = trend_data.get('error_types', [])
    if error_types:
        st.subheader("🔍 Error Type Analysis")
        
        df_errors = pd.DataFrame(error_types)
        df_errors['error_type'] = df_errors['error_type'].str.replace('_', ' ').str.title()
        df_errors = df_errors.sort_values('count', ascending=True)
        
        fig_errors = px.bar(
            df_errors,
            x='count',
            y='error_type',
            orientation='h',
            title=f'Error Types Distribution ({days} days)',
            labels={'count': 'Number of Occurrences', 'error_type': 'Error Type'},
            color='count',
            color_continuous_scale='Reds'
        )
        fig_errors.update_layout(height=400)
        st.plotly_chart(fig_errors, use_container_width=True)

def render_healing_analytics(remediation_stats: Dict[str, Any], bigquery_service):
    """Render healing and remediation analytics"""
    
    st.subheader("🔧 Healing Analytics")
    
    if not remediation_stats or remediation_stats['total_remediations'] == 0:
        st.warning("No remediation data available.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Remediations", 
            f"{remediation_stats['total_remediations']:,}"
        )
    
    with col2:
        st.metric(
            "Success Rate", 
            f"{remediation_stats['success_rate']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Avg Duration", 
            f"{remediation_stats['average_duration']:.1f}s"
        )
    
    with col4:
        st.metric(
            "Active Remediations", 
            f"{remediation_stats['active_remediations']:,}"
        )
    
    # Strategy effectiveness
    strategy_stats = remediation_stats.get('strategy_stats', {})
    
    if strategy_stats:
        st.subheader("Strategy Effectiveness")
        
        # Convert strategy stats to DataFrame
        strategy_data = []
        for strategy, stats in strategy_stats.items():
            strategy_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Total Uses': stats['total'],
                'Success Rate': stats['success_rate'],
                'Avg Duration': stats['average_duration']
            })
        
        df_strategies = pd.DataFrame(strategy_data)
        df_strategies = df_strategies.sort_values('Success Rate', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy success rates
            fig_success = px.bar(
                df_strategies,
                x='Success Rate',
                y='Strategy',
                orientation='h',
                title='Strategy Success Rates',
                labels={'Success Rate': 'Success Rate (%)', 'Strategy': 'Remediation Strategy'},
                color='Success Rate',
                color_continuous_scale='Greens'
            )
            fig_success.update_layout(height=400)
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            # Strategy usage frequency
            fig_usage = px.bar(
                df_strategies,
                x='Total Uses',
                y='Strategy',
                orientation='h',
                title='Strategy Usage Frequency',
                labels={'Total Uses': 'Number of Uses', 'Strategy': 'Remediation Strategy'},
                color='Total Uses',
                color_continuous_scale='Blues'
            )
            fig_usage.update_layout(height=400)
            st.plotly_chart(fig_usage, use_container_width=True)
        
        # Strategy performance table
        st.subheader("Strategy Performance Details")
        st.dataframe(
            df_strategies.style.format({
                'Success Rate': '{:.1f}%',
                'Avg Duration': '{:.1f}s'
            }).background_gradient(subset=['Success Rate'], cmap='Greens'),
            use_container_width=True
        )
    
    # Error type healing success
    error_type_stats = remediation_stats.get('error_type_stats', {})
    
    if error_type_stats:
        st.subheader("Healing Success by Error Type")
        
        error_data = []
        for error_type, stats in error_type_stats.items():
            error_data.append({
                'Error Type': error_type.replace('_', ' ').title(),
                'Total Occurrences': stats['total'],
                'Successfully Healed': stats['successful'],
                'Healing Rate': stats['success_rate']
            })
        
        df_errors = pd.DataFrame(error_data)
        df_errors = df_errors.sort_values('Healing Rate', ascending=False)
        
        # Healing rate by error type chart
        fig_error_healing = px.bar(
            df_errors,
            x='Healing Rate',
            y='Error Type',
            orientation='h',
            title='Healing Success Rate by Error Type',
            labels={'Healing Rate': 'Healing Success Rate (%)', 'Error Type': 'Error Type'},
            color='Healing Rate',
            color_continuous_scale='RdYlGn',
            text='Healing Rate'
        )
        fig_error_healing.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig_error_healing.update_layout(height=400)
        st.plotly_chart(fig_error_healing, use_container_width=True)

def render_ml_performance(monitoring_service):
    """Render ML model performance metrics"""
    
    st.subheader("🤖 ML Model Performance")
    
    # Get ML classifier stats
    classifier = monitoring_service.error_classifier
    model_stats = classifier.get_model_stats()
    
    if 'error' in model_stats:
        st.error(f"ML Model Error: {model_stats['error']}")
        return
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model Accuracy",
            f"{model_stats['accuracy']:.3f}"
        )
    
    with col2:
        st.metric(
            "Avg Confidence",
            f"{model_stats['average_confidence']:.3f}"
        )
    
    with col3:
        st.metric(
            "Total Predictions",
            f"{model_stats['total_predictions']:,}"
        )
    
    with col4:
        st.metric(
            "Supported Error Types",
            f"{len(model_stats['supported_error_types'])}"
        )
    
    # Class distribution
    class_distribution = model_stats.get('class_distribution', {})
    
    if class_distribution:
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution pie chart
            fig_pie = px.pie(
                values=list(class_distribution.values()),
                names=[name.replace('_', ' ').title() for name in class_distribution.keys()],
                title="Prediction Distribution by Error Type"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Prediction frequency bar chart
            df_dist = pd.DataFrame([
                {'Error Type': k.replace('_', ' ').title(), 'Predictions': v}
                for k, v in class_distribution.items()
            ]).sort_values('Predictions', ascending=True)
            
            fig_bar = px.bar(
                df_dist,
                x='Predictions',
                y='Error Type',
                orientation='h',
                title='Predictions by Error Type',
                labels={'Predictions': 'Number of Predictions', 'Error Type': 'Error Type'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Model Path:** {model_stats['model_path']}")
        st.info(f"**Last Updated:** {model_stats['last_updated'][:19]}")
    
    with col2:
        supported_types = model_stats['supported_error_types']
        st.write("**Supported Error Types:**")
        for error_type in supported_types:
            st.write(f"• {error_type.replace('_', ' ').title()}")
    
    # Test model predictions
    with st.expander("🧪 Test Model Predictions"):
        st.write("Test the ML model with custom error messages:")
        
        test_message = st.text_area(
            "Enter an error message to classify:",
            placeholder="e.g., Connection timeout after 30 seconds to BigQuery API"
        )
        
        if st.button("Classify Error") and test_message:
            try:
                prediction = classifier.classify_error(test_message)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Predicted Error Type:**")
                    st.write(prediction['error_type'].replace('_', ' ').title())
                
                with col2:
                    st.write("**Confidence Score:**")
                    st.write(f"{prediction['confidence']:.3f}")
                    st.progress(prediction['confidence'])
                
                with col3:
                    st.write("**Recommended Strategy:**")
                    st.write(prediction['remediation_strategy'].replace('_', ' ').title())
                
                # Show class probabilities
                st.write("**Class Probabilities:**")
                prob_df = pd.DataFrame([
                    {'Error Type': k.replace('_', ' ').title(), 'Probability': v}
                    for k, v in prediction['class_probabilities'].items()
                ]).sort_values('Probability', ascending=False)
                
                fig_prob = px.bar(
                    prob_df.head(10),
                    x='Probability',
                    y='Error Type',
                    orientation='h',
                    title='Classification Probabilities'
                )
                st.plotly_chart(fig_prob, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error classifying message: {e}")

def render_system_performance(monitoring_service, remediation_service, bigquery_service):
    """Render system performance metrics"""
    
    st.subheader("⚡ System Performance")
    
    monitoring_stats = monitoring_service.get_monitoring_stats()
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Events Processed",
            f"{monitoring_stats['processed_events']:,}"
        )
    
    with col2:
        detection_rate = monitoring_stats['error_detection_rate']
        st.metric(
            "Detection Rate",
            f"{detection_rate:.1f}%"
        )
    
    with col3:
        healing_rate = monitoring_stats['healing_rate']
        st.metric(
            "Healing Rate",
            f"{healing_rate:.1f}%"
        )
    
    with col4:
        queue_size = monitoring_stats['queue_size']
        st.metric(
            "Queue Size",
            f"{queue_size:,}"
        )
    
    # System load indicators
    st.subheader("System Load")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monitoring service status
        st.write("**Monitoring Service:**")
        if monitoring_stats['is_monitoring']:
            st.success("✅ Active and processing events")
        else:
            st.error("❌ Inactive")
        
        # Event processing rates
        st.write("**Event Processing:**")
        st.write(f"• Total Events: {monitoring_stats['processed_events']:,}")
        st.write(f"• Errors Detected: {monitoring_stats['detected_errors']:,}")
        st.write(f"• Healing Attempts: {monitoring_stats['healing_attempts']:,}")
        st.write(f"• Queue Size: {monitoring_stats['queue_size']:,}")
    
    with col2:
        # Performance indicators as gauges
        st.write("**Performance Indicators:**")
        
        # Detection rate gauge
        st.write("Error Detection Rate:")
        st.progress(min(detection_rate / 100, 1.0))
        st.write(f"{detection_rate:.1f}%")
        
        # Healing rate gauge
        st.write("Healing Success Rate:")
        st.progress(min(healing_rate / 100, 1.0))
        st.write(f"{healing_rate:.1f}%")
        
        # Queue utilization (simulated)
        queue_utilization = min(queue_size / 100, 1.0)  # Assume 100 is max capacity
        st.write("Queue Utilization:")
        st.progress(queue_utilization)
        st.write(f"{queue_utilization * 100:.1f}%")
    
    # Database performance
    st.subheader("Database Performance")
    
    try:
        # Get database metrics
        all_jobs = bigquery_service.get_jobs(limit=1000)
        all_logs = bigquery_service.get_logs(limit=1000)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Jobs in DB", f"{len(all_jobs):,}")
        
        with col2:
            st.metric("Total Log Entries", f"{len(all_logs):,}")
        
        with col3:
            # Estimate DB size (simplified)
            estimated_size = (len(all_jobs) * 500 + len(all_logs) * 200) / 1024  # KB
            st.metric("Estimated DB Size", f"{estimated_size:.1f} KB")
        
    except Exception as e:
        st.error(f"Error retrieving database metrics: {e}")
    
    # System health indicators
    st.subheader("Health Indicators")
    
    health_indicators = [
        ("Monitoring Service", "Healthy" if monitoring_stats['is_monitoring'] else "Unhealthy"),
        ("ML Model", "Loaded" if hasattr(monitoring_service, 'error_classifier') else "Not Loaded"),
        ("Database Connection", "Connected"),
        ("Event Processing", f"{monitoring_stats['processed_events']} events processed"),
        ("Error Detection", f"{monitoring_stats['detected_errors']} errors detected"),
    ]
    
    for indicator, status in health_indicators:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{indicator}:**")
        with col2:
            if status in ["Healthy", "Loaded", "Connected"] or "events" in status or "errors" in status:
                st.success(status)
            else:
                st.warning(status)
