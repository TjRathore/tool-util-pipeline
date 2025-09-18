"""
Custom ML Model Training Dashboard
Allows users to upload CSV/Excel files and retrain the error classification model
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

def render_model_training():
    """Render the custom model training interface"""
    
    st.title("🎯 Custom Model Training")
    st.markdown("Upload your own error data to improve the AI classification model")
    
    # Initialize error classifier from session state
    if 'services_initialized' not in st.session_state:
        st.error("Services not initialized. Please return to the main dashboard first.")
        return
    
    # Create tabs for different training options
    tab1, tab2, tab3 = st.tabs(["📤 Upload Training Data", "📊 Model Performance", "⚙️ Training Settings"])
    
    with tab1:
        render_data_upload()
    
    with tab2:
        render_model_performance()
        
    with tab3:
        render_training_settings()

def render_data_upload():
    """Render the data upload and training interface"""
    
    st.subheader("Upload Training Data")
    
    # File upload section
    st.markdown("### 📁 Upload Your Error Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing error messages and their classifications"
    )
    
    # Data format requirements
    with st.expander("📋 Data Format Requirements"):
        st.markdown("""
        Your data file should contain at least these columns:
        - **error_message**: The actual error message or log entry
        - **error_type**: The category/type of error (see supported types below)
        
        Optional columns:
        - **success**: Whether the error was successfully resolved (True/False)
        - **timestamp**: When the error occurred
        - **job_id**: Pipeline job identifier
        - **severity**: Error severity level
        
        **Supported Error Types:**
        - `permission_denied`
        - `network_timeout` 
        - `schema_mismatch`
        - `resource_exhaustion`
        - `data_duplication`
        - `missing_dependency`
        - `configuration_error`
        - `quota_exceeded`
        - `authentication_failure`
        - `data_corruption`
        - `unknown_error`
        """)
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            # Display data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data validation
            validation_results = validate_training_data(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Data Validation")
                if validation_results['is_valid']:
                    st.success("Data format is valid!")
                    st.info(f"✓ {validation_results['valid_records']} valid records found")
                    if validation_results['invalid_records'] > 0:
                        st.warning(f"⚠️ {validation_results['invalid_records']} invalid records will be skipped")
                else:
                    st.error("Data format issues found:")
                    for issue in validation_results['issues']:
                        st.error(f"• {issue}")
            
            with col2:
                st.markdown("### 📈 Data Statistics")
                if 'error_type' in df.columns:
                    error_counts = df['error_type'].value_counts()
                    st.plotly_chart(
                        px.pie(
                            values=error_counts.values,
                            names=error_counts.index,
                            title="Error Type Distribution"
                        ),
                        use_container_width=True
                    )
            
            # Training section
            if validation_results['is_valid'] and validation_results['valid_records'] > 0:
                st.markdown("---")
                st.markdown("### 🚀 Start Training")
                
                training_col1, training_col2 = st.columns(2)
                
                with training_col1:
                    test_split = st.slider(
                        "Test Data Split (%)",
                        min_value=10,
                        max_value=30,
                        value=20,
                        help="Percentage of data to use for model validation"
                    )
                
                with training_col2:
                    min_confidence = st.slider(
                        "Minimum Confidence for Auto-Remediation",
                        min_value=0.5,
                        max_value=0.95,
                        value=0.75,
                        help="Only errors classified with this confidence or higher will trigger automatic remediation"
                    )
                
                if st.button("🎯 Train Custom Model", type="primary"):
                    train_custom_model(df, validation_results, test_split/100, min_confidence)
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def validate_training_data(df: pd.DataFrame) -> Dict:
    """Validate the uploaded training data"""
    
    issues = []
    valid_records = 0
    invalid_records = 0
    
    # Check required columns
    required_columns = ['error_message', 'error_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        return {
            'is_valid': False,
            'issues': issues,
            'valid_records': 0,
            'invalid_records': len(df)
        }
    
    # Get supported error types
    from src.models.error_classifier import ErrorClassifier
    temp_classifier = ErrorClassifier()
    supported_types = temp_classifier.error_categories
    
    # Validate data quality
    for idx, row in df.iterrows():
        row_issues = []
        
        # Check for empty error messages
        try:
            error_msg = row['error_message']
            if pd.isna(error_msg) or str(error_msg).strip() == '':
                row_issues.append("Empty error message")
        except (KeyError, TypeError):
            row_issues.append("Missing error_message field")
        
        # Check for valid error types  
        try:
            error_type = row['error_type']
            if pd.isna(error_type) or str(error_type) not in supported_types:
                row_issues.append(f"Invalid error type: {error_type}")
        except (KeyError, TypeError):
            row_issues.append("Missing error_type field")
        
        if row_issues:
            invalid_records += 1
        else:
            valid_records += 1
    
    # Check if we have enough data
    if valid_records < 10:
        issues.append("At least 10 valid records are required for training")
    
    # Check error type distribution
    if valid_records > 0:
        try:
            valid_df = df.dropna(subset=['error_message', 'error_type'])
            valid_df = valid_df[valid_df['error_type'].isin(supported_types)]
            
            if len(valid_df) > 0:
                error_distribution = valid_df['error_type'].value_counts()
                
                if len(error_distribution) < 2:
                    issues.append("At least 2 different error types are required")
                
                if len(error_distribution) > 0:
                    min_samples_per_class = min(error_distribution.values)
                    if min_samples_per_class < 2:
                        issues.append("Each error type needs at least 2 samples")
        except Exception as e:
            issues.append(f"Error analyzing data distribution: {str(e)}")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'valid_records': valid_records,
        'invalid_records': invalid_records
    }

def train_custom_model(df: pd.DataFrame, validation_results: Dict, test_split: float, min_confidence: float):
    """Train a custom model with the uploaded data"""
    
    with st.spinner("Training custom model... This may take a few minutes."):
        
        try:
            # Get the error classifier from session state
            error_classifier = st.session_state.get('error_classifier')
            if not error_classifier:
                from src.models.error_classifier import ErrorClassifier
                error_classifier = ErrorClassifier()
            
            # Clean the data
            clean_df = df.dropna(subset=['error_message', 'error_type'])
            clean_df = clean_df[clean_df['error_type'].isin(error_classifier.error_categories)]
            
            # Add success column if not present (assume True for training data)
            if 'success' not in clean_df.columns:
                clean_df = clean_df.copy()
                clean_df['success'] = True
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start training
            status_text.text("Preparing training data...")
            progress_bar.progress(0.2)
            
            status_text.text("Training model...")
            progress_bar.progress(0.5)
            
            # Ensure clean_df is properly typed as DataFrame
            clean_df = pd.DataFrame(clean_df)
            
            # Use the retrain_model method
            training_results = error_classifier.retrain_model(clean_df)
            
            progress_bar.progress(0.8)
            status_text.text("Evaluating model performance...")
            
            # Get updated model stats
            model_stats = error_classifier.get_model_stats()
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Display results
            if training_results['success']:
                st.success("🎉 Model training completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Model Accuracy",
                        f"{training_results['new_accuracy']:.1%}",
                        f"{training_results['improvement']:+.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Training Samples",
                        f"{training_results['feedback_samples']:,}",
                        "New samples added"
                    )
                
                with col3:
                    st.metric(
                        "Confidence Threshold",
                        f"{min_confidence:.1%}",
                        "For auto-remediation"
                    )
                
                # Update the session state with new classifier
                st.session_state['error_classifier'] = error_classifier
                
                # Show detailed results
                with st.expander("📊 Detailed Training Results"):
                    st.json(training_results)
                
            else:
                st.warning("⚠️ Model was not updated - no improvement detected")
                st.info(f"Current model accuracy: {training_results['old_accuracy']:.1%}")
                st.info(f"New model accuracy: {training_results['new_accuracy']:.1%}")
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)

def render_model_performance():
    """Render model performance metrics and visualizations"""
    
    st.subheader("Model Performance Metrics")
    
    # Get model stats
    try:
        if 'services_initialized' in st.session_state:
            from src.models.error_classifier import ErrorClassifier
            error_classifier = ErrorClassifier()
            stats = error_classifier.get_model_stats()
            
            if 'error' not in stats:
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Model Accuracy", f"{stats['accuracy']:.1%}")
                
                with col2:
                    st.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
                
                with col3:
                    st.metric("Total Predictions", f"{stats['total_predictions']:,}")
                
                with col4:
                    st.metric("Error Categories", len(stats['supported_error_types']))
                
                # Class distribution chart
                if 'class_distribution' in stats:
                    st.markdown("### 📊 Error Type Distribution")
                    
                    class_dist = stats['class_distribution']
                    df_dist = pd.DataFrame([
                        {'Error Type': error_type, 'Count': count}
                        for error_type, count in class_dist.items()
                    ])
                    
                    fig = px.bar(
                        df_dist,
                        x='Error Type',
                        y='Count',
                        title="Model Training Data Distribution by Error Type"
                    )
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model information
                st.markdown("### ℹ️ Model Information")
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.info(f"**Model Path:** {stats['model_path']}")
                    st.info(f"**Last Updated:** {stats['last_updated'][:19]}")
                
                with info_col2:
                    st.info(f"**Supported Error Types:** {len(stats['supported_error_types'])}")
                    with st.expander("View All Error Types"):
                        for error_type in stats['supported_error_types']:
                            st.text(f"• {error_type}")
            else:
                st.error(stats['error'])
                
    except Exception as e:
        st.error(f"Failed to load model stats: {str(e)}")

def render_training_settings():
    """Render training configuration settings"""
    
    st.subheader("Training Configuration")
    
    # Model parameters
    st.markdown("### 🔧 Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Text Preprocessing:**")
        max_features = st.number_input("Max TF-IDF Features", value=5000, min_value=1000, max_value=10000)
        ngram_range = st.selectbox("N-gram Range", ["(1,2)", "(1,3)", "(1,4)"], index=1)
        
    with col2:
        st.markdown("**Random Forest Parameters:**")
        n_estimators = st.number_input("Number of Trees", value=100, min_value=50, max_value=300)
        max_depth = st.number_input("Max Tree Depth", value=20, min_value=10, max_value=50)
    
    # Confidence settings
    st.markdown("### ⚡ Auto-Remediation Settings")
    
    confidence_threshold = st.slider(
        "Minimum Confidence for Auto-Remediation",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        help="Errors classified with confidence below this threshold will be escalated to humans"
    )
    
    max_retry_attempts = st.number_input(
        "Maximum Retry Attempts",
        value=3,
        min_value=1,
        max_value=10,
        help="Maximum number of automatic remediation attempts before escalation"
    )
    
    # Advanced settings
    with st.expander("🔬 Advanced Settings"):
        st.markdown("**Data Validation:**")
        min_samples_per_class = st.number_input("Min Samples per Error Type", value=2, min_value=1)
        
        st.markdown("**Training Process:**")
        random_state = st.number_input("Random Seed", value=42, help="For reproducible results")
        class_weight = st.selectbox("Class Weight Strategy", ["balanced", "balanced_subsample", "None"])
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        settings = {
            'max_features': max_features,
            'ngram_range': eval(ngram_range),
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'confidence_threshold': confidence_threshold,
            'max_retry_attempts': max_retry_attempts,
            'min_samples_per_class': min_samples_per_class,
            'random_state': random_state,
            'class_weight': class_weight if class_weight != "None" else None
        }
        
        # Save to session state
        st.session_state['training_settings'] = settings
        
        st.success("✅ Settings saved successfully!")
        st.info("Settings will be applied to the next training session.")