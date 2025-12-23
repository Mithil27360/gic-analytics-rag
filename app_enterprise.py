"""
Enterprise RAG Analytics Engine - Streamlit Interface
Deterministic, auditable, evidence-only system
"""

import streamlit as st
import sys
import os
from pathlib import Path

# CRITICAL: Load .env BEFORE any other initializations
from dotenv import load_dotenv
load_dotenv(override=True)

# Add Rag directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Rag'))

from rag_enterprise import EnterpriseRAGEngine

# Page Configuration
st.set_page_config(
    page_title="Enterprise RAG Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Minimal, professional
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG System (Cached)
@st.cache_resource
def initialize_enterprise_rag():
    """Initialize and cache the Enterprise RAG engine"""
    try:
        engine = EnterpriseRAGEngine()
        engine.ingest_documents()
        return engine
    except Exception as e:
        st.error(f"Error initializing RAG engine: {e}")
        return None

# Header
st.markdown('<p class="main-header">🏢 Enterprise RAG Analytics Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deterministic | Auditable | Evidence-Only | Temperature=0.0</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("System Configuration")
    
    # Initialize system
    with st.spinner("Initializing enterprise engine..."):
        engine = initialize_enterprise_rag()
    
    if engine:
        st.success("✓ System Online")
        
        if engine.groq_client:
            st.info("🤖 Groq LLM: Active (T=0.0)")
        else:
            st.warning("⚠️ Groq API: Not configured")
        
        if engine.collection:
            doc_count = engine.collection.count()
            st.info(f"📚 Vector DB: {doc_count} documents")
        
        if engine.bm25:
            st.info("🔎 BM25: Active")
    else:
        st.error("❌ System initialization failed")
        st.stop()
    
    st.divider()
    
    # RAG Settings
    st.header("Query Settings")
    k_docs = st.slider("Documents to retrieve", min_value=3, max_value=10, value=6,
                       help="Number of documents for retrieval")
    
    st.divider()
    
    # Output Contract
    st.header("Output Contract")
    st.code("""Answer:
[Direct factual response]

Metrics:
- [Name]: [Value] [Source_ID]

Sources:
- [Source_ID]""", language=None)
    
    st.divider()
    
    # Sample Queries
    st.header("Sample Queries")
    st.markdown("""
    **Industry Metrics:**
    - Total industry premium FY25?
    - Compare FY25 vs FY24 growth
    
    **Segment Analysis:**
    - Which segment has highest premium?
    - Health segment YoY growth?
    
    **Company Analysis:**
    - Top 5 companies by premium
    - High crop risk exposure?
    """)
    
    st.divider()
    
    # Data Coverage
    with st.expander("Data Coverage"):
        st.markdown("""
        **Period:** FY24 & FY25 (Apr-Oct)
        
        **Companies:** 34 insurers
        
        **Segments:** 9 segments
        - health, motor_total, fire
        - misc, marine_total, engineering
        - liability, personal_accident, aviation
        
        **Metrics:**
        - Premium volumes (₹ Cr)
        - YoY growth (%)
        - Market share
        """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display answer
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources Referenced"):
                    for source in message["sources"]:
                        st.markdown(f"- `{source}`")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter analytical query..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating deterministic response..."):
            try:
                # Call enterprise RAG
                result = engine.query(prompt, k=k_docs, verbose=False)
                
                # Display answer
                st.markdown(result['answer'])
                
                # Display sources
                if result.get('sources'):
                    with st.expander("📚 Sources Referenced"):
                        for source in result['sources']:
                            st.markdown(f"- `{source}`")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result.get('sources', [])
                })
            
            except Exception as e:
                error_msg = f"**Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Mode:** Enterprise Deterministic")
with col2:
    st.markdown("**Temperature:** 0.0")
with col3:
    st.markdown("**LLM:** Groq Llama 3.3 70B")
with col4:
    st.markdown("**Vector DB:** ChromaDB")
