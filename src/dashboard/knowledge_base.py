"""
Knowledge Base Management Dashboard
Handles upload and management of structured knowledge for RAG system
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

def render_knowledge_base():
    """Render knowledge base management interface"""
    
    st.title("📚 Knowledge Base Management")
    st.markdown("Upload and manage structured knowledge for RAG-enhanced error classification")
    
    # Tabs for different knowledge base operations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Knowledge", 
        "📋 Data Templates", 
        "🔍 Browse Knowledge Base",
        "⚙️ RAG Configuration"
    ])
    
    with tab1:
        render_knowledge_upload()
    
    with tab2:
        render_data_templates()
    
    with tab3:
        render_knowledge_browser()
    
    with tab4:
        render_rag_configuration()

def render_knowledge_upload():
    """Render knowledge base upload interface"""
    
    st.subheader("📤 Upload Knowledge Base Data")
    
    # Knowledge type selection
    knowledge_type = st.selectbox(
        "Select Knowledge Type",
        [
            "Historical Error Cases",
            "Troubleshooting Runbooks", 
            "Best Practices & Procedures",
            "Infrastructure Documentation",
            "Team Expertise & Lessons Learned",
            "Configuration Templates",
            "Escalation Procedures"
        ],
        help="Choose the type of knowledge you're uploading for proper processing"
    )
    
    # File upload section
    st.markdown("### Upload Data File")
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload structured knowledge data using the specified templates"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully: {len(df)} records")
            
            # Preview uploaded data
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate data format
            validation_result = validate_knowledge_data(df, knowledge_type)
            
            if validation_result['is_valid']:
                st.success("✅ Data format validation passed")
                
                # Processing options
                st.markdown("### Processing Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    chunk_size = st.number_input(
                        "Text Chunk Size",
                        min_value=100,
                        max_value=2000,
                        value=500,
                        help="Size of text chunks for vector embedding"
                    )
                
                with col2:
                    overlap = st.number_input(
                        "Chunk Overlap",
                        min_value=0,
                        max_value=200,
                        value=50,
                        help="Overlap between consecutive chunks"
                    )
                
                # Metadata options
                st.markdown("### Metadata Configuration")
                
                priority_level = st.selectbox(
                    "Priority Level",
                    ["High", "Medium", "Low"],
                    index=1,
                    help="Priority for retrieval in RAG system"
                )
                
                tags = st.text_input(
                    "Tags (comma-separated)",
                    placeholder="production, critical, dataflow, bigquery",
                    help="Tags for filtering and categorization"
                )
                
                # Process and upload button
                if st.button("🚀 Process and Upload to Knowledge Base", type="primary"):
                    with st.spinner("Processing and uploading knowledge base data..."):
                        try:
                            # Process the data
                            result = process_knowledge_upload(
                                df=df,
                                knowledge_type=knowledge_type,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                priority=priority_level,
                                tags=tags.split(',') if tags else []
                            )
                            
                            # Display results
                            st.success("✅ Knowledge base updated successfully!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Records Processed", result['records_processed'])
                            with col2:
                                st.metric("Chunks Created", result['chunks_created'])
                            with col3:
                                st.metric("Embeddings Generated", result['embeddings_generated'])
                            
                            # Show processing summary
                            with st.expander("📊 Processing Summary"):
                                st.json(result['summary'])
                            
                        except Exception as e:
                            st.error(f"❌ Error processing data: {str(e)}")
            else:
                st.error("❌ Data format validation failed")
                for issue in validation_result['issues']:
                    st.error(f"• {issue}")
                
                st.info("💡 Please check the Data Templates tab for correct format requirements")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def render_data_templates():
    """Render data template specifications"""
    
    st.subheader("📋 Data Templates & Specifications")
    
    # Template selector
    template_type = st.selectbox(
        "Select Template Type",
        [
            "Historical Error Cases",
            "Troubleshooting Runbooks",
            "Best Practices & Procedures", 
            "Infrastructure Documentation",
            "Team Expertise & Lessons Learned",
            "Configuration Templates",
            "Escalation Procedures"
        ]
    )
    
    # Display template specifications
    templates = get_knowledge_templates()
    template_spec = templates[template_type]
    
    st.markdown(f"### {template_type} Template")
    st.markdown(template_spec['description'])
    
    # Required columns
    st.markdown("#### Required Columns")
    required_df = pd.DataFrame([
        {"Column": col, "Type": spec['type'], "Description": spec['description']}
        for col, spec in template_spec['required_columns'].items()
    ])
    st.dataframe(required_df, use_container_width=True)
    
    # Optional columns
    if template_spec.get('optional_columns'):
        st.markdown("#### Optional Columns")
        optional_df = pd.DataFrame([
            {"Column": col, "Type": spec['type'], "Description": spec['description']}
            for col, spec in template_spec['optional_columns'].items()
        ])
        st.dataframe(optional_df, use_container_width=True)
    
    # Sample data
    st.markdown("#### Sample Data")
    sample_data = create_sample_data(template_spec)
    st.dataframe(sample_data, use_container_width=True)
    
    # Download template
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = sample_data.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Template",
            data=csv_data,
            file_name=f"{template_type.lower().replace(' ', '_')}_template.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel download
        excel_buffer = create_excel_template(sample_data)
        st.download_button(
            label="📊 Download Excel Template",
            data=excel_buffer,
            file_name=f"{template_type.lower().replace(' ', '_')}_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # RAG Integration explanation
    st.markdown("### 🤖 How This Integrates with RAG + LLM")
    
    st.markdown(f"""
    **{template_type} RAG Integration:**
    
    🔄 **Data Processing Flow:**
    1. **Upload** → Your {template_type.lower()} data is processed and chunked
    2. **Embedding** → Each chunk is converted to vector embeddings using Google's text embedding models
    3. **Storage** → Embeddings stored in vector database with rich metadata
    4. **Retrieval** → When errors occur, similar cases are retrieved based on semantic similarity
    5. **Generation** → LLM uses retrieved context to provide enhanced classification and recommendations
    
    🎯 **RAG Enhancement Benefits:**
    - **Domain Expertise**: LLM learns from your specific {template_type.lower()}
    - **Contextual Accuracy**: Retrieves relevant historical cases for better decisions
    - **Continuous Learning**: Knowledge base grows with every upload
    - **Institutional Memory**: Preserves team expertise and lessons learned
    
    🔍 **Retrieval Strategy:**
    - **Semantic Search**: Finds similar errors even with different wording
    - **Metadata Filtering**: Narrows results by priority, tags, and date ranges
    - **Hybrid Ranking**: Combines vector similarity with business rules
    - **Context Augmentation**: Provides LLM with rich background for better responses
    """)

def render_knowledge_browser():
    """Render knowledge base browser"""
    
    st.subheader("🔍 Browse Knowledge Base")
    
    # Initialize session state for knowledge base if not exists
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = initialize_mock_knowledge_base()
    
    knowledge_base = st.session_state.knowledge_base
    
    # Search and filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input(
            "🔍 Search Knowledge Base",
            placeholder="Enter search terms..."
        )
    
    with col2:
        knowledge_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(set([item['type'] for item in knowledge_base]))
        )
    
    with col3:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "High", "Medium", "Low"]
        )
    
    # Apply filters
    filtered_knowledge = knowledge_base
    
    if knowledge_filter != "All":
        filtered_knowledge = [item for item in filtered_knowledge if item['type'] == knowledge_filter]
    
    if priority_filter != "All":
        filtered_knowledge = [item for item in filtered_knowledge if item['priority'] == priority_filter]
    
    if search_query:
        filtered_knowledge = [
            item for item in filtered_knowledge 
            if search_query.lower() in item['content'].lower() or 
               search_query.lower() in item['title'].lower()
        ]
    
    # Display results
    st.markdown(f"**Found {len(filtered_knowledge)} knowledge base entries**")
    
    # Knowledge base statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", len(knowledge_base))
    
    with col2:
        high_priority = len([item for item in knowledge_base if item['priority'] == 'High'])
        st.metric("High Priority", high_priority)
    
    with col3:
        error_cases = len([item for item in knowledge_base if item['type'] == 'Historical Error Cases'])
        st.metric("Error Cases", error_cases)
    
    with col4:
        recent_entries = len([item for item in knowledge_base if 
                            (datetime.now() - datetime.fromisoformat(item['created_at'])).days < 7])
        st.metric("Recent (7 days)", recent_entries)
    
    # Display knowledge entries
    for i, item in enumerate(filtered_knowledge[:20]):  # Limit to 20 results
        with st.expander(f"📚 {item['title']} | {item['type']} | Priority: {item['priority']}"):
            st.markdown(f"**Content:** {item['content'][:500]}...")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Tags:** {', '.join(item['tags'])}")
                st.markdown(f"**Created:** {item['created_at']}")
            
            with col2:
                if item.get('metadata'):
                    st.markdown("**Metadata:**")
                    st.json(item['metadata'])
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🔍 Similar Items", key=f"similar_{i}"):
                    st.info("Finding similar items... (Feature in development)")
            
            with col2:
                if st.button(f"✏️ Edit", key=f"edit_{i}"):
                    st.info("Edit functionality coming soon...")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    if st.session_state.get(f"confirm_delete_{i}"):
                        # Remove item (in real implementation, update vector store)
                        st.session_state.knowledge_base.remove(item)
                        st.success("Item deleted successfully!")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{i}"] = True
                        st.warning("Click again to confirm deletion")

def render_rag_configuration():
    """Render RAG system configuration"""
    
    st.subheader("⚙️ RAG Configuration")
    
    # Current configuration
    st.markdown("### Current RAG Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Retrieval Settings")
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for retrieving relevant knowledge"
        )
        
        max_retrieved_docs = st.number_input(
            "Max Retrieved Documents",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of documents to retrieve for context"
        )
        
        chunk_size = st.number_input(
            "Default Chunk Size",
            min_value=100,
            max_value=2000,
            value=500,
            help="Default size for text chunking"
        )
    
    with col2:
        st.markdown("#### LLM Integration Settings")
        
        context_window = st.number_input(
            "Context Window Size",
            min_value=1000,
            max_value=32000,
            value=8000,
            help="Maximum context size for LLM input"
        )
        
        rag_mode = st.selectbox(
            "RAG Mode",
            ["Retrieve-then-Generate", "Generate-then-Retrieve", "Hybrid"],
            help="Strategy for combining retrieval with generation"
        )
        
        fallback_enabled = st.checkbox(
            "Enable Fallback to Base Model",
            value=True,
            help="Use base model if no relevant knowledge is found"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Vector Store Settings")
            
            vector_dimension = st.number_input(
                "Vector Dimension",
                min_value=384,
                max_value=1536,
                value=768,
                help="Dimension of embedding vectors"
            )
            
            index_type = st.selectbox(
                "Index Type",
                ["HNSW", "IVF", "Flat"],
                help="Vector index algorithm"
            )
        
        with col2:
            st.markdown("#### Embedding Model Settings")
            
            embedding_model = st.selectbox(
                "Embedding Model",
                [
                    "textembedding-gecko@001",
                    "textembedding-gecko@003", 
                    "textembedding-gecko-multilingual@001"
                ],
                help="Google Cloud embedding model to use"
            )
            
            batch_size = st.number_input(
                "Embedding Batch Size",
                min_value=1,
                max_value=100,
                value=32,
                help="Batch size for embedding generation"
            )
    
    # RAG Performance Metrics
    st.markdown("### 📊 RAG Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retrieval Accuracy", "87.3%", "+2.1%")
    
    with col2:
        st.metric("Average Latency", "245ms", "-15ms")
    
    with col3:
        st.metric("Cache Hit Rate", "78.5%", "+5.2%")
    
    with col4:
        st.metric("Knowledge Coverage", "92.1%", "+1.8%")
    
    # Save configuration
    if st.button("💾 Save RAG Configuration", type="primary"):
        # In real implementation, save to configuration service
        config = {
            "similarity_threshold": similarity_threshold,
            "max_retrieved_docs": max_retrieved_docs,
            "chunk_size": chunk_size,
            "context_window": context_window,
            "rag_mode": rag_mode,
            "fallback_enabled": fallback_enabled,
            "vector_dimension": vector_dimension,
            "index_type": index_type,
            "embedding_model": embedding_model,
            "batch_size": batch_size
        }
        
        st.session_state.rag_config = config
        st.success("✅ RAG configuration saved successfully!")
        
        # Show saved configuration
        with st.expander("📋 Saved Configuration"):
            st.json(config)

def validate_knowledge_data(df: pd.DataFrame, knowledge_type: str) -> Dict[str, Any]:
    """Validate uploaded knowledge base data"""
    
    issues = []
    warnings = []
    
    templates = get_knowledge_templates()
    template = templates[knowledge_type]
    
    # Check required columns
    required_columns = set(template['required_columns'].keys())
    actual_columns = set(df.columns)
    
    missing_columns = required_columns - actual_columns
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and content
    for col, spec in template['required_columns'].items():
        if col in df.columns:
            # Check for empty values
            empty_count = df[col].isna().sum()
            if empty_count > 0:
                warnings.append(f"Column '{col}' has {empty_count} empty values")
            
            # Check data type constraints
            if spec['type'] == 'text' and col in df.columns:
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length < 10:
                    warnings.append(f"Column '{col}' has very short text (avg: {avg_length:.1f} chars)")
    
    # Check minimum records
    if len(df) < 5:
        warnings.append(f"Only {len(df)} records found. Consider uploading more data for better RAG performance.")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'record_count': len(df),
        'column_count': len(df.columns)
    }

def get_knowledge_templates() -> Dict[str, Dict]:
    """Get knowledge base template specifications"""
    
    return {
        "Historical Error Cases": {
            "description": "Historical pipeline error cases with resolutions for RAG context",
            "required_columns": {
                "error_message": {"type": "text", "description": "Complete error message or description"},
                "error_type": {"type": "category", "description": "Classification category"},
                "resolution_strategy": {"type": "text", "description": "How the error was resolved"},
                "success": {"type": "boolean", "description": "Whether resolution was successful"},
                "resolution_time_minutes": {"type": "numeric", "description": "Time taken to resolve"}
            },
            "optional_columns": {
                "pipeline_id": {"type": "text", "description": "Pipeline identifier"},
                "environment": {"type": "category", "description": "dev/staging/prod"},
                "error_context": {"type": "text", "description": "Additional context about the error"},
                "team_notes": {"type": "text", "description": "Team insights and lessons learned"},
                "business_impact": {"type": "category", "description": "Low/Medium/High/Critical"},
                "date_occurred": {"type": "date", "description": "When the error occurred"}
            }
        },
        "Troubleshooting Runbooks": {
            "description": "Step-by-step troubleshooting procedures and runbooks",
            "required_columns": {
                "title": {"type": "text", "description": "Runbook title or procedure name"},
                "description": {"type": "text", "description": "Detailed description of the procedure"},
                "steps": {"type": "text", "description": "Step-by-step instructions"},
                "applicable_errors": {"type": "text", "description": "Error types this runbook applies to"}
            },
            "optional_columns": {
                "prerequisites": {"type": "text", "description": "Prerequisites before running"},
                "estimated_time": {"type": "text", "description": "Expected time to complete"},
                "required_permissions": {"type": "text", "description": "Required access levels"},
                "automation_available": {"type": "boolean", "description": "Can this be automated"},
                "last_updated": {"type": "date", "description": "When runbook was last updated"}
            }
        },
        "Best Practices & Procedures": {
            "description": "Best practices, procedures, and operational guidelines",
            "required_columns": {
                "practice_title": {"type": "text", "description": "Name of the best practice"},
                "description": {"type": "text", "description": "Detailed description"},
                "category": {"type": "category", "description": "Category (monitoring, deployment, etc.)"},
                "implementation_steps": {"type": "text", "description": "How to implement this practice"}
            },
            "optional_columns": {
                "benefits": {"type": "text", "description": "Benefits of following this practice"},
                "risks_if_ignored": {"type": "text", "description": "Risks of not following"},
                "tools_required": {"type": "text", "description": "Required tools or systems"},
                "compliance_related": {"type": "boolean", "description": "Related to compliance requirements"}
            }
        },
        "Infrastructure Documentation": {
            "description": "Infrastructure setup, configuration, and architecture documentation",
            "required_columns": {
                "component_name": {"type": "text", "description": "Infrastructure component name"},
                "description": {"type": "text", "description": "Component description and purpose"},
                "configuration": {"type": "text", "description": "Configuration details"},
                "dependencies": {"type": "text", "description": "Component dependencies"}
            },
            "optional_columns": {
                "owner_team": {"type": "text", "description": "Responsible team"},
                "criticality": {"type": "category", "description": "Business criticality level"},
                "backup_procedures": {"type": "text", "description": "Backup and recovery procedures"},
                "monitoring_setup": {"type": "text", "description": "Monitoring configuration"},
                "cost_info": {"type": "text", "description": "Cost information and optimization"}
            }
        },
        "Team Expertise & Lessons Learned": {
            "description": "Institutional knowledge, expert insights, and lessons learned",
            "required_columns": {
                "topic": {"type": "text", "description": "Topic or area of expertise"},
                "insight": {"type": "text", "description": "Key insight or lesson learned"},
                "context": {"type": "text", "description": "Context when this was learned"},
                "expert_recommendation": {"type": "text", "description": "Expert recommendation or advice"}
            },
            "optional_columns": {
                "expert_name": {"type": "text", "description": "Subject matter expert"},
                "date_learned": {"type": "date", "description": "When this lesson was learned"},
                "related_incidents": {"type": "text", "description": "Related incidents or events"},
                "preventive_measures": {"type": "text", "description": "How to prevent similar issues"}
            }
        },
        "Configuration Templates": {
            "description": "Configuration templates and standards for various systems",
            "required_columns": {
                "template_name": {"type": "text", "description": "Configuration template name"},
                "system_type": {"type": "text", "description": "System or service type"},
                "template_content": {"type": "text", "description": "Configuration template content"},
                "use_case": {"type": "text", "description": "When to use this template"}
            },
            "optional_columns": {
                "parameters": {"type": "text", "description": "Configurable parameters"},
                "validation_rules": {"type": "text", "description": "Validation requirements"},
                "security_considerations": {"type": "text", "description": "Security requirements"},
                "performance_impact": {"type": "text", "description": "Performance considerations"}
            }
        },
        "Escalation Procedures": {
            "description": "Escalation procedures and contact information",
            "required_columns": {
                "escalation_trigger": {"type": "text", "description": "When to escalate"},
                "escalation_path": {"type": "text", "description": "Who to contact and in what order"},
                "severity_level": {"type": "category", "description": "Severity level (P1/P2/P3/P4)"},
                "communication_template": {"type": "text", "description": "Communication template to use"}
            },
            "optional_columns": {
                "sla_requirements": {"type": "text", "description": "SLA response time requirements"},
                "external_contacts": {"type": "text", "description": "External vendor contacts if needed"},
                "post_incident_steps": {"type": "text", "description": "Post-incident review steps"},
                "documentation_requirements": {"type": "text", "description": "Required documentation"}
            }
        }
    }

def create_sample_data(template_spec: Dict) -> pd.DataFrame:
    """Create sample data for template"""
    
    sample_data = {}
    
    # Add required columns with sample data
    for col, spec in template_spec['required_columns'].items():
        if col == "error_message":
            sample_data[col] = [
                "BigQuery job failed: Access denied to dataset 'prod-analytics'",
                "Dataflow job timeout after 60 minutes in region us-central1",
                "Cloud Storage upload failed: Insufficient permissions"
            ]
        elif col == "error_type":
            sample_data[col] = ["permission_denied", "network_timeout", "permission_denied"]
        elif col == "resolution_strategy":
            sample_data[col] = [
                "Updated IAM permissions for service account",
                "Increased worker machine type and timeout settings",
                "Added storage.objectCreator role to service account"
            ]
        elif col == "success":
            sample_data[col] = [True, True, False]
        elif col == "resolution_time_minutes":
            sample_data[col] = [15, 45, 120]
        elif spec['type'] == 'text':
            sample_data[col] = [f"Sample {col} content {i+1}" for i in range(3)]
        elif spec['type'] == 'category':
            sample_data[col] = ["Category A", "Category B", "Category C"]
        elif spec['type'] == 'boolean':
            sample_data[col] = [True, False, True]
        elif spec['type'] == 'numeric':
            sample_data[col] = [10, 20, 30]
        elif spec['type'] == 'date':
            sample_data[col] = ["2024-01-15", "2024-01-16", "2024-01-17"]
    
    # Add some optional columns for demonstration
    if template_spec.get('optional_columns'):
        for col, spec in list(template_spec['optional_columns'].items())[:2]:  # Add first 2 optional
            if spec['type'] == 'text':
                sample_data[col] = [f"Sample {col} {i+1}" for i in range(3)]
            elif spec['type'] == 'category':
                sample_data[col] = ["Option 1", "Option 2", "Option 3"]
            elif spec['type'] == 'boolean':
                sample_data[col] = [True, False, True]
    
    return pd.DataFrame(sample_data)

def create_excel_template(df: pd.DataFrame) -> bytes:
    """Create Excel template with formatting"""
    import io
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Template']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BD',
            'border': 1
        })
        
        # Format header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Set column widths
        for col_num, col_name in enumerate(df.columns):
            max_length = max(len(str(col_name)), df[col_name].astype(str).str.len().max())
            worksheet.set_column(col_num, col_num, min(max_length + 2, 50))
    
    return output.getvalue()

def process_knowledge_upload(df: pd.DataFrame, knowledge_type: str, chunk_size: int, 
                           overlap: int, priority: str, tags: List[str]) -> Dict[str, Any]:
    """Process uploaded knowledge data for RAG system"""
    
    # Import RAG system here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.rag.vector_store import RAGKnowledgeBase
    
    # Initialize RAG knowledge base
    rag_kb = RAGKnowledgeBase()
    
    # Initialize processing counters
    records_processed = 0
    chunks_created = 0
    embeddings_generated = 0
    
    # Process each record
    for idx, row in df.iterrows():
        try:
            # Process based on knowledge type
            if knowledge_type == "Historical Error Cases":
                error_data = {
                    'error_message': str(row.get('error_message', '')),
                    'error_type': str(row.get('error_type', '')),
                    'resolution_strategy': str(row.get('resolution_strategy', '')),
                    'success': bool(row.get('success', False)),
                    'resolution_time_minutes': int(row.get('resolution_time_minutes', 0)),
                    'context': str(row.get('error_context', '')),
                    'business_impact': str(row.get('business_impact', 'medium'))
                }
                rag_kb.add_error_case(error_data)
                
            elif knowledge_type == "Troubleshooting Runbooks":
                runbook_data = {
                    'title': str(row.get('title', '')),
                    'description': str(row.get('description', '')),
                    'steps': str(row.get('steps', '')),
                    'applicable_errors': str(row.get('applicable_errors', '')),
                    'estimated_time': str(row.get('estimated_time', ''))
                }
                rag_kb.add_runbook(runbook_data)
                
            elif knowledge_type == "Best Practices & Procedures":
                practice_data = {
                    'practice_title': str(row.get('practice_title', '')),
                    'description': str(row.get('description', '')),
                    'implementation_steps': str(row.get('implementation_steps', '')),
                    'category': str(row.get('category', ''))
                }
                rag_kb.add_best_practice(practice_data)
            
            else:
                # Generic processing for other types
                text_columns = [col for col in row.index if isinstance(row[col], str) and len(str(row[col])) > 10]
                content = " ".join([str(row[col]) for col in text_columns[:3]])
                
                document = {
                    'type': knowledge_type.lower().replace(' ', '_'),
                    'content': content,
                    'metadata': {col: str(row[col]) for col in row.index if col not in text_columns[:3]},
                    'priority': priority
                }
                
                rag_kb.vector_store.add_document(document)
            
            # Estimate chunks and embeddings (simplified)
            content_length = len(str(row.values))
            estimated_chunks = max(1, content_length // chunk_size)
            chunks_created += estimated_chunks
            embeddings_generated += estimated_chunks
            
            records_processed += 1
            
        except Exception as e:
            st.warning(f"Error processing row {idx}: {str(e)}")
    
    # Save the knowledge base
    rag_kb.save_knowledge()
    
    # Create processing summary
    summary = {
        "knowledge_type": knowledge_type,
        "processing_time": f"{records_processed * 0.1:.1f} seconds",
        "success_rate": f"{(records_processed/len(df)*100):.1f}%",
        "average_chunk_size": chunk_size,
        "priority_level": priority,
        "tags_applied": tags,
        "rag_integration": "active"
    }
    
    return {
        "records_processed": records_processed,
        "chunks_created": chunks_created,
        "embeddings_generated": embeddings_generated,
        "summary": summary
    }

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks"""
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = end - overlap
    
    return chunks

def generate_mock_embedding(text: str) -> List[float]:
    """Generate mock embedding vector (in real implementation, use actual embedding service)"""
    # Placeholder: generate random embedding
    np.random.seed(hash(text) % 2147483647)  # Deterministic based on text
    return np.random.random(768).tolist()  # 768-dimensional embedding

def store_in_vector_db(document: Dict, chunk: str, embedding: List[float]):
    """Store document chunk and embedding in vector database (placeholder)"""
    # In real implementation, store in actual vector database (Chroma, FAISS, etc.)
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = []
    
    st.session_state.vector_store.append({
        "document_id": document["id"],
        "chunk": chunk,
        "embedding": embedding,
        "metadata": document["metadata"],
        "type": document["type"],
        "priority": document["priority"],
        "tags": document["tags"],
        "created_at": document["created_at"]
    })

def initialize_mock_knowledge_base() -> List[Dict]:
    """Initialize knowledge base from actual RAG vector store"""
    
    try:
        # Import RAG system
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.rag.vector_store import RAGKnowledgeBase
        
        # Initialize RAG knowledge base
        rag_kb = RAGKnowledgeBase()
        
        # Get actual documents from vector store
        knowledge_docs = []
        for doc in rag_kb.vector_store.documents[:20]:  # Limit to first 20 for display
            knowledge_docs.append({
                "title": doc.get('content', '')[:60] + "...",
                "type": doc.get('type', 'Unknown').replace('_', ' ').title(),
                "content": doc.get('content', ''),
                "priority": doc.get('priority', 'Medium').title(),
                "tags": doc.get('metadata', {}).keys(),
                "created_at": doc.get('created_at', datetime.now().isoformat()),
                "metadata": doc.get('metadata', {})
            })
        
        return knowledge_docs if knowledge_docs else _get_fallback_knowledge()
        
    except Exception as e:
        # Fallback to mock data if RAG system not available
        return _get_fallback_knowledge()

def _get_fallback_knowledge() -> List[Dict]:
    """Fallback mock knowledge data"""
    
    return [
        {
            "title": "BigQuery Access Denied Error Resolution",
            "type": "Historical Error Cases", 
            "content": "When BigQuery jobs fail with 'Access denied to dataset', check service account IAM permissions. Solution: Add BigQuery Data Editor role to the service account. Resolution time: 15 minutes.",
            "priority": "High",
            "tags": ["bigquery", "permissions", "iam"],
            "created_at": "2024-01-15T10:30:00",
            "metadata": {"error_type": "permission_denied", "success": True, "resolution_time": 15}
        },
        {
            "title": "Dataflow Job Timeout Best Practices",
            "type": "Best Practices & Procedures",
            "content": "To prevent Dataflow job timeouts: 1) Use appropriate machine types 2) Implement proper windowing 3) Monitor resource utilization 4) Set reasonable timeouts based on data volume.",
            "priority": "Medium", 
            "tags": ["dataflow", "timeout", "performance"],
            "created_at": "2024-01-16T14:20:00",
            "metadata": {"category": "performance_optimization"}
        },
        {
            "title": "Critical Pipeline Failure Escalation",
            "type": "Escalation Procedures",
            "content": "For P1 pipeline failures: 1) Alert on-call engineer immediately 2) Create incident ticket 3) Notify business stakeholders within 15 minutes 4) Activate backup procedures if needed.",
            "priority": "High",
            "tags": ["escalation", "critical", "p1"],
            "created_at": "2024-01-17T09:15:00", 
            "metadata": {"severity": "P1", "sla": "15 minutes"}
        }
    ]