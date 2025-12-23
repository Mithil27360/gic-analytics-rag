"""
GIC Insurance Analytics RAG Copilot - Advanced
Streamlit Web Interface

Powered by Groq (Llama 3.3 70B) and ChromaDB
Features: Hybrid Search, Query Expansion, Reranking, Citation Tracking
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add Rag directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Rag'))

from rag_advanced import AdvancedRAGCopilot


# Page Configuration
st.set_page_config(
    page_title="GIC Analytics Copilot 🚀",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        background: #10b981;
        color: white;
    }
    .feature-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 500;
        background: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Initialize RAG System (Cached)
@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the Advanced RAG copilot"""
    try:
        copilot = AdvancedRAGCopilot()
        
        # Check if knowledge base exists
        if not Path(copilot.knowledge_base_path).exists():
            st.error("Knowledge base not found. Generating documents...")
            # Auto-generate knowledge base
            from document_generator import RAGDocumentGenerator
            import pandas as pd
            
            # Load data
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "data", "processed", "gic_ytd_master_apr_oct.csv")
            df = pd.read_csv(data_path)
            
            # Set categorical month order
            df["ytd_upto_month"] = pd.Categorical(
                df["ytd_upto_month"],
                categories=["APR", "MAY", "JUNE", "JULY", "AUG", "SEP", "OCT"],
                ordered=True
            )
            
            # Generate documents
            generator = RAGDocumentGenerator(df)
            documents, metadata = generator.generate_all_documents()
            generator.save_documents()
            
            st.success(f"✓ Generated {len(documents)} knowledge base documents")
        
        # Ingest into vector database
        copilot.ingest_documents()
        
        return copilot
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


# Header
st.markdown('<p class="main-header">🛡️ GIC Insurance Analytics Copilot 🚀</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Groq (Llama 3.3 70B) • Hybrid Search • Query Expansion • Reranking</p>', unsafe_allow_html=True)

# Feature Badges
st.markdown("""
<div>
    <span class="feature-badge">🔍 Hybrid Search</span>
    <span class="feature-badge">🎯 Query Expansion</span>
    <span class="feature-badge">📊 Reranking</span>
    <span class="feature-badge">✂️ Context Compression</span>
    <span class="feature-badge">📑 Citation Tracking</span>
    <span class="feature-badge">💯 Confidence Scoring</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Initialize system
    with st.spinner("Initializing Advanced RAG system..."):
        copilot = initialize_rag_system()
    
    if copilot:
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        st.markdown("**✓ System Ready**")
        
        # Show API status
        if copilot.groq_client:
            st.success("🤖 Groq API: Connected")
            st.success("🔍 Hybrid Search: Active")
            st.success("🎯 Query Expansion: Active")
        else:
            st.warning("⚠️ Groq API: Not configured\n\nUsing template-based responses")
        
        # Show vector DB status
        if copilot.collection:
            doc_count = copilot.collection.count()
            st.info(f"📚 Vector DB: {doc_count} documents")
        
        # Show BM25 status
        if copilot.bm25:
            st.info("🔎 BM25 Keyword Search: Active")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("❌ System initialization failed")
        st.stop()
    
    st.divider()
    
    # RAG Settings
    st.header("🎛️ RAG Settings")
    
    k_docs = st.slider("Documents to retrieve", min_value=3, max_value=10, value=5, 
                       help="Number of documents to retrieve from vector database")
    
    verbose_mode = st.checkbox("Verbose Mode", value=False,
                              help="Show detailed RAG pipeline steps")
    
    st.divider()
    
    # Knowledge Base Management
    st.header("📚 Knowledge Base")
    
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Rebuilding knowledge base..."):
            try:
                from document_generator import RAGDocumentGenerator
                import pandas as pd
                
                # Load data
                base_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(base_dir, "data", "processed", "gic_ytd_master_apr_oct.csv")
                df = pd.read_csv(data_path)
                
                # Set categorical month order
                df["ytd_upto_month"] = pd.Categorical(
                    df["ytd_upto_month"],
                    categories=["APR", "MAY", "JUNE", "JULY", "AUG", "SEP", "OCT"],
                    ordered=True
                )
                
                # Generate documents
                generator = RAGDocumentGenerator(df)
                documents, metadata = generator.generate_all_documents()
                generator.save_documents()
                
                # Re-ingest
                st.cache_resource.clear()
                copilot = initialize_rag_system()
                
                st.success(f"✓ Rebuilt {len(documents)} documents")
            except Exception as e:
                st.error(f"Error rebuilding: {e}")
    
    st.divider()
    
    # Sample Queries
    st.header("💡 Sample Queries")
    st.markdown("""
    **Risk Analysis:**
    - Which insurers have risky growth?
    - Companies exposed to crop risk?
    
    **Market Analysis:**
    - Total industry premium in FY25?
    - Compare health and motor segments
    
    **Company Intel:**
    - Top 5 health insurers by premium
    - Who has highest YoY growth?
    
    **Trends:**
    - Key trends in FY25?
    - Segment growth patterns?
    """)
    
    st.divider()
    
    # Data Coverage
    with st.expander("📊 Data Coverage"):
        st.markdown("""
        **Period:** FY24 & FY25 (Apr-Oct)
        
        **Segments:**
        - Health Insurance
        - Motor (Total)
        - Miscellaneous (incl. Crop)
        - Fire & Property
        - Personal Accident
        - Engineering
        - Liability
        - Marine
        - Aviation
        
        **Metrics:**
        - Premium volumes (YTD)
        - YoY growth rates
        - Company-level breakdown
        - Risk classifications
        - Volatility analysis
        """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            cols = st.columns(4)
            with cols[0]:
                st.metric("Docs Used", meta.get('docs_used', 0))
            with cols[1]:
                st.metric("Citations", meta.get('citations_count', 0))
            with cols[2]:
                confidence = meta.get('confidence', 0)
                st.metric("Confidence", f"{confidence*100:.0f}%")
            with cols[3]:
                st.metric("Mode", "Advanced" if copilot.groq_client else "Template")

# Chat input
if prompt := st.chat_input("Ask about premiums, risk, growth, or company comparisons..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing with Advanced RAG pipeline..."):
            try:
                # Call advanced RAG
                result = copilot.ask(prompt, k=k_docs, verbose=verbose_mode)
                
                # Display answer
                st.markdown(result['answer'])
                
                # Display metadata
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Docs Used", result['docs_used'])
                with cols[1]:
                    st.metric("Citations", len(result['citations']))
                with cols[2]:
                    confidence = result['confidence']
                    st.metric("Confidence", f"{confidence*100:.0f}%")
                with cols[3]:
                    st.metric("Mode", "Advanced" if copilot.groq_client else "Template")
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['answer'],
                    "metadata": {
                        'docs_used': result['docs_used'],
                        'citations_count': len(result['citations']),
                        'confidence': result['confidence']
                    }
                })
            
            except Exception as e:
                error_msg = f"**Error:** {str(e)}\n\nPlease try rephrasing your query or check system configuration."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Data:** GIC Monthly Reports")
with col2:
    st.markdown("**LLM:** Groq Llama 3.3 70B")
with col3:
    st.markdown("**Vector DB:** ChromaDB")
with col4:
    st.markdown("**Search:** Hybrid (Semantic+BM25)")
