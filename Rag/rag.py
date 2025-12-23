"""
RAG Copilot Core System
Handles document ingestion, retrieval, and grounded answer generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGCopilot:
    """
    Insurance Analytics RAG Copilot
    Provides grounded Q&A over insurance data without hallucinations
    """
    
    def __init__(self, 
                 knowledge_base_path: str = None,
                 chroma_db_path: str = "./chroma_db"):
        """
        Initialize RAG system
        
        Args:
            knowledge_base_path: Path to document CSV (auto-detected if None)
            chroma_db_path: Path to ChromaDB persistent storage
        """
        # Set up paths
        if knowledge_base_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_base_path = os.path.join(base_dir, "data", "processed", "rag_knowledge_base.csv")
        
        self.knowledge_base_path = knowledge_base_path
        self.chroma_db_path = chroma_db_path
        self.documents = None
        self.embeddings = None
        self.collection = None
        self.groq_client = None
        
        # Load documents
        self._load_documents()
        
        # Initialize embedder
        self._initialize_embedder()
        
        # Initialize Groq LLM
        self._initialize_groq()
    
    def _load_documents(self):
        """Load knowledge base documents"""
        if not Path(self.knowledge_base_path).exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {self.knowledge_base_path}. "
                "Run document_generator.py first."
            )
        
        self.documents = pd.read_csv(self.knowledge_base_path)
        print(f"Loaded {len(self.documents)} documents")
        print(f"Document types: {self.documents['doc_type'].value_counts().to_dict()}")
    
    def _initialize_embedder(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedder initialized: all-MiniLM-L6-v2")
        except ImportError:
            print("ERROR: sentence-transformers not installed.")
            print("Install: pip install sentence-transformers")
            raise
    
    def _initialize_groq(self):
        """Initialize Groq LLM for generation"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("WARNING: GROQ_API_KEY not set in .env file")
                print("Generate RAG will fall back to template-based responses")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=api_key)
                print("✓ Groq LLM initialized (Llama 3.3 70B)")
        except ImportError:
            print("ERROR: groq not installed. Install: pip install groq")
            self.groq_client = None
        except Exception as e:
            print(f"ERROR initializing Groq: {e}")
            self.groq_client = None
    
    def ingest_documents(self):
        """
        Ingest documents into vector database
        Uses ChromaDB for efficient similarity search
        """
        try:
            import chromadb
        except ImportError:
            print("ERROR: chromadb not installed. Install: pip install chromadb")
            raise
        
        # Initialize persistent ChromaDB
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        # Get or create collection
        try:
            self.collection = client.get_collection(name="gic_insights")
            print(f"✓ Loaded existing vector database ({self.collection.count()} documents)")
            return self.collection
        except:
            pass
        
        # Create new collection
        self.collection = client.create_collection(
            name="gic_insights",
            metadata={"description": "GIC Insurance Analytics Documents"}
        )
        
        # Generate embeddings
        print("Generating embeddings...")
        texts = self.documents["text"].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add to collection
        print("Adding to vector database...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=self.documents["doc_id"].tolist(),
            metadatas=self.documents[["doc_type", "doc_id"]].to_dict("records")
        )
        
        print(f"✓ Ingested {len(texts)} documents into vector database")
        
        return self.collection
    
    def retrieve(self, query: str, k: int = 3) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User question
            k: Number of documents to retrieve
            
        Returns:
            (documents, metadata): Retrieved texts and metadata
        """
        if self.collection is None:
            # Reinitialize collection
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.chroma_db_path)
                self.collection = client.get_collection("gic_insights")
            except:
                print("ERROR: Collection not found. Run ingest_documents() first.")
                return [], []
        
        # Encode query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        return documents, metadatas
    
    def generate_answer(self, query: str, contexts: List[str], 
                       metadatas: List[Dict]) -> str:
        """
        Generate answer using retrieved contexts
        
        Args:
            query: User question
            contexts: Retrieved document texts
            metadatas: Document metadata
            
        Returns:
            Generated answer with citations
        """
        if not contexts:
            return self._generate_no_data_response(query)
        
        # Build context block with citations
        context_block = self._build_context_block(contexts, metadatas)
        
        # Generate answer
        if self.groq_client:
            try:
                answer = self._generate_with_groq(query, context_block, contexts, metadatas)
            except Exception as e:
                # Fallback to template if Groq fails
                answer = self._generate_with_template(query, contexts, metadatas)
        else:
            answer = self._generate_with_template(query, contexts, metadatas)
        
        return answer
    
    def _build_context_block(self, contexts: List[str], 
                            metadatas: List[Dict]) -> str:
        """Build formatted context block with document IDs"""
        context_parts = []
        
        for i, (text, meta) in enumerate(zip(contexts, metadatas)):
            doc_id = meta.get("doc_id", f"doc_{i}")
            context_parts.append(f"[Document {i+1}: {doc_id}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_with_groq(self, query: str, context_block: str,
                           contexts: List[str], metadatas: List[Dict]) -> str:
        """Generate answer using Groq (Llama 3.3 70B)"""
        system_prompt = """You are an expert Insurance Analytics Copilot.

CRITICAL RULES:
1. Answer STRICTLY based on the Context documents provided
2. If information is not in the documents, say "Data not available in current knowledge base"
3. ALWAYS cite document IDs in square brackets when making claims
4. Use clear, structured formatting with markdown
5. Do NOT fabricate numbers, companies, or facts
6. Be concise but comprehensive

RESPONSE FORMAT:
**Key Findings:**
- Finding 1 [Document ID]
- Finding 2 [Document ID]

**Analysis:**
Brief interpretation and insights

**Data Sources:**
List of document IDs referenced"""
        
        user_prompt = f"""USER QUERY:
{query}

CONTEXT DOCUMENTS:
{context_block}

Provide a grounded answer based ONLY on the context above."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                top_p=0.9
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Groq API error: {e}")
            # Fallback to template with contexts
            return self._generate_with_template(query, contexts, metadatas)
    
    def _generate_with_template(self, query: str, contexts: List[str], 
                               metadatas: List[Dict]) -> str:
        """
        Generate answer using template-based approach
        No external LLM needed - just structured extraction
        """
        # Classify query type
        query_lower = query.lower()
        
        # Extract relevant facts from contexts
        facts = []
        sources = []
        
        for ctx, meta in zip(contexts, metadatas):
            doc_id = meta.get("doc_id", "unknown")
            
            # Extract key information based on query
            if any(word in query_lower for word in ["risk", "risky", "exposure"]):
                risk_info = self._extract_risk_info(ctx, doc_id)
                if risk_info:
                    facts.append(risk_info)
                    sources.append(doc_id)
            
            elif any(word in query_lower for word in ["growth", "growing", "trend"]):
                growth_info = self._extract_growth_info(ctx, doc_id)
                if growth_info:
                    facts.append(growth_info)
                    sources.append(doc_id)
            
            elif any(word in query_lower for word in ["compare", "comparison", "versus", "vs"]):
                comparison_info = self._extract_comparison_info(ctx, doc_id)
                if comparison_info:
                    facts.append(comparison_info)
                    sources.append(doc_id)
            
            else:
                # General extraction
                general_info = self._extract_general_info(ctx, doc_id, query_lower)
                if general_info:
                    facts.append(general_info)
                    sources.append(doc_id)
        
        # Build structured answer
        if not facts:
            return self._generate_no_data_response(query)
        
        answer = f"""**Key Findings:**
{chr(10).join('- ' + fact for fact in facts)}

**Insight:**
{self._generate_insight(query_lower, facts)}

**Sources:**
{chr(10).join('- ' + src for src in set(sources))}"""
        
        return answer
    
    def _extract_risk_info(self, context: str, doc_id: str) -> Optional[str]:
        """Extract risk-related information"""
        risk_keywords = ["risk", "exposure", "volatility", "concentration", "vulnerable"]
        
        lines = context.split("\n")
        relevant_lines = [
            line.strip() for line in lines 
            if any(kw in line.lower() for kw in risk_keywords) and len(line.strip()) > 20
        ]
        
        if relevant_lines:
            # Take most relevant line
            return relevant_lines[0] + f" [{doc_id}]"
        
        return None
    
    def _extract_growth_info(self, context: str, doc_id: str) -> Optional[str]:
        """Extract growth-related information"""
        # Look for growth percentages and trends
        import re
        
        # Find YoY growth mentions
        growth_pattern = r"(?:YoY Growth|growth):\s*([+-]?\d+\.?\d*)%"
        matches = re.findall(growth_pattern, context, re.IGNORECASE)
        
        if matches:
            growth_val = matches[0]
            # Find context around it
            lines = context.split("\n")
            for line in lines:
                if growth_val in line or "growth" in line.lower():
                    return f"{line.strip()} [{doc_id}]"
        
        # Fallback: any line mentioning growth
        lines = context.split("\n")
        for line in lines:
            if "growth" in line.lower() and len(line.strip()) > 20:
                return f"{line.strip()} [{doc_id}]"
        
        return None
    
    def _extract_comparison_info(self, context: str, doc_id: str) -> Optional[str]:
        """Extract comparison-relevant information"""
        # Look for comparative statements
        comp_keywords = ["vs", "versus", "compared", "higher", "lower", "better", "worse"]
        
        lines = context.split("\n")
        for line in lines:
            if any(kw in line.lower() for kw in comp_keywords) and len(line.strip()) > 20:
                return f"{line.strip()} [{doc_id}]"
        
        # Fallback: extract key metrics
        if "Premium" in context or "Share" in context:
            for line in lines:
                if ("premium" in line.lower() or "share" in line.lower()) and ":" in line:
                    return f"{line.strip()} [{doc_id}]"
        
        return None
    
    def _extract_general_info(self, context: str, doc_id: str, 
                             query_words: str) -> Optional[str]:
        """Extract general relevant information"""
        # Find lines that match query keywords
        query_terms = [w for w in query_words.split() if len(w) > 3]
        
        lines = context.split("\n")
        best_match = None
        max_matches = 0
        
        for line in lines:
            if len(line.strip()) < 20:
                continue
            
            matches = sum(1 for term in query_terms if term in line.lower())
            if matches > max_matches:
                max_matches = matches
                best_match = line.strip()
        
        if best_match:
            return f"{best_match} [{doc_id}]"
        
        # Fallback: first substantive line
        for line in lines:
            if len(line.strip()) > 30 and ":" in line:
                return f"{line.strip()} [{doc_id}]"
        
        return None
    
    def _generate_insight(self, query: str, facts: List[str]) -> str:
        """Generate insight based on query type and facts"""
        if "risk" in query:
            return "These indicators suggest varying levels of portfolio risk. Higher concentration and volatility indicate greater vulnerability to segment-specific shocks."
        
        elif "growth" in query:
            return "Growth patterns reflect different strategic approaches. Sustainable growth typically combines moderate pace with low volatility and diversified segments."
        
        elif "compare" in query or "vs" in query:
            return "The comparison reveals different risk-return profiles. Consider growth sustainability, volatility, and portfolio quality alongside premium volume."
        
        else:
            return "Review these metrics in context of overall portfolio strategy and market conditions."
    
    def _generate_no_data_response(self, query: str) -> str:
        """Generate response when no relevant data found"""
        return f"""**Data Not Available**

The query "{query}" could not be answered with the current knowledge base.

**Possible reasons:**
- Information not present in processed GIC data (Apr-Oct 2025)
- Query refers to companies/segments outside dataset
- Requires real-time data not captured in monthly reports

**Available data covers:**
- Industry and segment trends (FY24 vs FY25)
- Company-level premium and growth metrics
- Risk classifications (crop, health strategy, concentration)
- Growth quality indicators

**For assistance:** Try rephrasing your query or contact the analytics team."""
    
    def ask(self, query: str, k: int = 3, verbose: bool = False) -> str:
        """
        Main query interface
        
        Args:
            query: User question
            k: Number of documents to retrieve
            verbose: Print retrieval details
            
        Returns:
            Generated answer
        """
        # Retrieve
        contexts, metadatas = self.retrieve(query, k=k)
        
        if verbose:
            print(f"\n=== Retrieved {len(contexts)} documents ===")
            for i, meta in enumerate(metadatas):
                print(f"{i+1}. {meta.get('doc_id', 'unknown')}")
        
        # Generate
        answer = self.generate_answer(query, contexts, metadatas)
        
        return answer


def main():
    """Demo the RAG copilot"""
    
    print("="*80)
    print("GIC Insurance Analytics RAG Copilot")
    print("="*80)
    
    # Initialize
    copilot = RAGCopilot()
    
    # Ingest documents
    print("\nIngesting documents...")
    copilot.ingest_documents()
    
    # Test queries
    test_queries = [
        "Which insurers have risky growth?",
        "What is the total industry premium in FY25?",
        "Compare health and motor segments",
        "Which companies are exposed to crop insurance risk?",
        "What are the key trends in FY25?"
    ]
    
    print("\n" + "="*80)
    print("TESTING RAG COPILOT")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Q: {query}")
        print(f"{'='*80}")
        
        answer = copilot.ask(query, k=3, verbose=False)
        print(f"\nA:\n{answer}\n")
    
    print("\n" + "="*80)
    print("Setup complete! Use copilot.ask('your question') to query")
    print("="*80)
    
    return copilot


if __name__ == "__main__":
    copilot = main()