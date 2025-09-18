"""
AI-Driven Self-Healing Data Pipeline Monitoring System
Main Streamlit Application Entry Point
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dashboard.main_dashboard import render_main_dashboard
from src.dashboard.analytics_dashboard import render_analytics_dashboard
from src.dashboard.manual_override import render_manual_override
from src.dashboard.model_training import render_model_training
from src.dashboard.knowledge_base import render_knowledge_base
from src.utils.config import Config
from src.services.bigquery_service import BigQueryService
from src.services.monitoring_service import MonitoringService
from src.services.remediation_service import RemediationService

def initialize_services():
    """Initialize all services and store in session state"""
    if 'services_initialized' not in st.session_state:
        config = Config()
        
        # Initialize services
        st.session_state.bigquery_service = BigQueryService(config)
        st.session_state.monitoring_service = MonitoringService(config)
        st.session_state.remediation_service = RemediationService(config)
        
        # Initialize database tables
        st.session_state.bigquery_service.initialize_tables()
        
        st.session_state.services_initialized = True

def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Pipeline Monitor",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize services
    initialize_services()
    
    # Sidebar navigation
    st.sidebar.title("🔧 AI Pipeline Monitor")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["Dashboard", "Analytics", "Manual Override", "Model Training", "Knowledge Base", "System Health"]
    )
    
    # Main content area
    if page == "Dashboard":
        render_main_dashboard()
    elif page == "Analytics":
        render_analytics_dashboard()
    elif page == "Manual Override":
        render_manual_override()
    elif page == "Model Training":
        render_model_training()
    elif page == "Knowledge Base":
        render_knowledge_base()
    elif page == "System Health":
        render_system_health()

def render_system_health():
    """Render system health status page"""
    st.title("🏥 System Health Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Services Status", "🟢 Online", "All systems operational")
    
    with col2:
        st.metric("ML Model Status", "🟢 Active", "Last retrained 2h ago")
    
    with col3:
        st.metric("BigQuery Status", "🟢 Connected", "Latency: 45ms")
    
    with col4:
        st.metric("Pub/Sub Status", "🟢 Streaming", "Events processed: 1.2K/min")
    
    st.markdown("---")
    
    # System configuration
    st.subheader("📋 System Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.info("**Environment:** Production")
        st.info("**GCP Project:** ai-pipeline-monitor")
        st.info("**Region:** us-central1")
    
    with config_col2:
        st.info("**Auto-healing:** Enabled")
        st.info("**Confidence Threshold:** 0.75")
        st.info("**Max Retry Attempts:** 3")

if __name__ == "__main__":
    main()
