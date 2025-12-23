#  GIC Insurance Analytics RAG Copilot

Production-grade Insurance Analytics RAG Copilot powered by **Groq (Llama 3.3 70B)** and **ChromaDB**.

##  Overview

This system provides AI-powered analytics over General Insurance Company (GIC) premium data (FY24 & FY25, Apr-Oct). It uses **Retrieval-Augmented Generation (RAG)** to deliver grounded, hallucination-free insights.

### Key Features

-  **Zero Hallucinations**: All answers grounded in actual data with citations
-  **Advanced Analytics**: Growth metrics, volatility analysis, risk classifications
-  **Persistent Storage**: ChromaDB vector database with disk persistence


##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Groq API

Edit `.env` file and add your Groq API key:

```bash
GROQ_API_KEY=your_actual_key_here
```

Get your free API key at: https://console.groq.com/keys

### 3. Generate Knowledge Base

```bash
cd Rag
python document_generator.py
```

This creates **49 semantic documents** from your insurance data:
- 34 company summaries
- 9 segment overviews
- 4 risk classifications
- 1 industry overview
- 1 growth insights document

### 4. Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

##  Sample Queries

Try asking:
- "Which insurers have risky growth?"
- "What is the total industry premium in FY25?"
- "Compare health and motor segments"
- "Which companies are exposed to crop insurance risk?"
- "Show me companies with high YoY growth"
- "What are the key trends in FY25?"

##  Project Structure

```
rag star/
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .env                        # API keys (gitignored)
│
├── Rag/                        # Core RAG system
│   ├── __init__.py            # Package initialization
│   ├── analytics.py           # Data analysis logic
│   ├── document_generator.py  # Semantic document creation
│   └── rag.py                 # RAG engine (Groq + ChromaDB)
│
├── data/
│   ├── processed/
│   │   ├── gic_ytd_master_apr_oct.csv      # Main dataset
│   │   ├── rag_knowledge_base.csv           # Generated documents
│   │   └── rag_metadata.json               # Document metadata
│   ├── cleaned/               # Monthly Excel files
│   └── raw/                   # Original data
│
├── analysis/
│   └── gic_ytd_premium_analysis_report.ipynb
│
└── chroma_db/                 # Vector database (auto-created)
```

##  Architecture

### 1. Analytics Layer (`analytics.py`)
- Computes growth metrics at industry/segment/company levels
- Calculates volatility and risk classifications
- Portfolio concentration analysis
- Company ranking by growth/stability

### 2. Document Generator (`document_generator.py`)
- Transforms analytics into semantic text chunks
- Creates 5 document types optimized for retrieval
- Saves to CSV with metadata

### 3. RAG Engine (`rag.py`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB with persistent storage
- **LLM**: Groq Llama 3.3 70B (ultra-fast inference)
- **Retrieval**: Top-k semantic search
- **Generation**: Grounded answers with citations

### 4. Web Interface (`app.py`)
- Streamlit chat interface
- Real-time query processing
- Knowledge base management
- System status monitoring

##  System Requirements

- Python 3.8+
- 8GB RAM minimum
- Internet connection (for Groq API)
- macOS, Linux, or Windows

##  Usage Guide

### Basic Workflow

1. **Ask a Question**: Type your query in the chat input
2. **RAG Retrieval**: System finds top-4 relevant documents
3. **Groq Generation**: Llama 3.3 generates grounded answer
4. **Citations**: All facts include document IDs

### Knowledge Base Management

- Click **" Rebuild Knowledge Base"** in sidebar to regenerate documents
- Useful after updating the source CSV data
- Automatically re-ingests into vector database

### API Key Management

If Groq API key is missing:
- System falls back to **template-based responses**
- Still functional but less sophisticated
- Add key to `.env` and restart app

##  Testing

### Test Document Generation
```bash
cd Rag
python document_generator.py
```

Expected output: `Saved 49 documents to data/processed`

### Test RAG Engine
```bash
cd Rag
python rag.py
```

Runs 5 test queries and displays answers.

### Test Analytics
```bash
cd Rag
python analytics.py
```

Shows industry metrics and growth calculations.

##  Troubleshooting

### "Knowledge base not found"
Run: `cd Rag && python document_generator.py`

### "GROQ_API_KEY not set"
Edit `.env` file and add your API key. Restart the app.

### Dependency conflicts
Some packages may show version conflicts. These are non-critical and can be ignored unless functionality is broken.

### ChromaDB errors
Delete `chroma_db/` folder and restart app to regenerate the vector database.

##  Data Coverage

**Time Period**: FY24 & FY25 (April-October)

**Segments**:
- Health Insurance
- Motor (Total)
- Miscellaneous (incl. Crop)
- Fire & Property
- Personal Accident
- Engineering
- Liability
- Marine
- Aviation

**Companies**: 34 insurance companies tracked

**Metrics**:
- Premium volumes (YTD)
- YoY growth rates
- Monthly premium trends
- Portfolio concentration
- Risk classifications

##  Key Insights Generated

1. **Company Summaries**: Premium, top segment, volatility, risk notes
2. **Segment Analysis**: Market share, YoY growth, characteristics, risks
3. **Risk Classifications**: Crop risk, health strategy types, concentration
4. **Industry Overview**: Total premium, trends, strategic insights
5. **Growth Patterns**: Momentum analysis, quality indicators

##  Security

- API keys stored in `.env` (gitignored)
- No data sent outside Groq API
- All data processing happens locally
- ChromaDB runs locally (no cloud)

##  Performance

- **Query Response**: 1-3 seconds 
- **Knowledge Base Gen**: ~5 seconds for 49 documents
- **Vector Ingestion**: ~10 seconds for full corpus
- **Embedding Model**: Runs on CPU, very efficient

##  License

This project uses:
- Groq API (subject to Groq's terms)
- ChromaDB (Apache 2.0)
- Streamlit (Apache 2.0)
- Sentence Transformers (Apache 2.0)



For issues or questions:
1. Check the Troubleshooting section
2. Verify Groq API key is valid
3. Ensure all dependencies are installed
4. Review data paths in code

---
