"""
Advanced RAG Copilot - Production Grade
Implements: Hybrid Search, Reranking, Query Expansion, Advanced Prompting, Citation Tracking
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import os
import re
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AdvancedRAGCopilot:
    """
    Production-grade Insurance Analytics RAG Copilot
    Features: Hybrid search, reranking, query expansion, advanced prompting
    """
    
    def __init__(self, 
                 knowledge_base_path: str = None,
                 chroma_db_path: str = "./chroma_db"):
        """
        Initialize Advanced RAG system
        
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
        self.bm25 = None  # For keyword search
        
        # Load documents
        self._load_documents()
        
        # Initialize embedder
        self._initialize_embedder()
        
        # Initialize BM25 for keyword search
        self._initialize_bm25()
        
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
        print(f"✓ Loaded {len(self.documents)} documents")
        print(f"  Document types: {self.documents['doc_type'].value_counts().to_dict()}")
    
    def _initialize_embedder(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Embedder initialized: all-MiniLM-L6-v2")
        except ImportError:
            print("ERROR: sentence-transformers not installed.")
            raise
    
    def _initialize_bm25(self):
        """Initialize BM25 for keyword search"""
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents for BM25
            texts = self.documents['text'].tolist()
            tokenized_corpus = [self._tokenize(doc) for doc in texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("✓ BM25 keyword search initialized")
        except ImportError:
            print("⚠️  rank-bm25 not installed. Hybrid search disabled.")
            print("   Install: pip install rank-bm25")
            self.bm25 = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _initialize_groq(self):
        """Initialize Groq LLM for generation"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                print("⚠️  GROQ_API_KEY not configured in .env file")
                print("   Fallback: Using template-based responses")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=api_key)
                print("✓ Groq LLM initialized (Llama 3.3 70B Versatile)")
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
        print("  Generating embeddings...")
        texts = self.documents["text"].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Add to collection
        print("  Adding to vector database...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=self.documents["doc_id"].tolist(),
            metadatas=self.documents[["doc_type", "doc_id"]].to_dict("records")
        )
        
        print(f"✓ Ingested {len(texts)} documents into vector database\n")
        
        return self.collection
    
    def expand_query(self, query: str) -> List[str]:
        """
        Query expansion using Groq LLM
        Generates multiple variations of the query for better retrieval
        """
        if not self.groq_client:
            # Fallback: simple expansion with synonyms
            return [query]
        
        try:
            expansion_prompt = f"""You are a query expansion expert for insurance analytics.

Given a user query, generate 2 alternative phrasings that capture the same intent but use different terminology.

Original Query: "{query}"

Generate variations that:
1. Use different but equivalent insurance/financial terminology
2. Be more specific about metrics (premiums, growth, risk, etc.)
3. Include relevant segment/company terms if applicable

Format: Return ONLY the alternative queries, one per line, no numbering or explanation.

Alternative Queries:"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            expanded = response.choices[0].message.content.strip().split('\n')
            expanded = [q.strip() for q in expanded if q.strip()]
            
            # Include original + expansions (max 3 total)
            return [query] + expanded[:2]
            
        except Exception as e:
            print(f"  Query expansion failed: {e}")
            return [query]
    
    def hybrid_retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Hybrid retrieval: Combines semantic search + BM25 keyword search
        Uses Reciprocal Rank Fusion (RRF) to merge results
        
        Args:
            query: User question
            k: Number of documents to retrieve
            
        Returns:
            (documents, metadata, scores): Retrieved texts, metadata, and fusion scores
        """
        # Semantic search via ChromaDB
        semantic_results = self._semantic_search(query, k=k*2)
        
        # Keyword search via BM25
        if self.bm25:
            keyword_results = self._keyword_search(query, k=k*2)
        else:
            keyword_results = []
        
        # Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, 
            keyword_results, 
            k=k
        )
        
        return fused_results
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[str, Dict, float]]:
        """Semantic search using embeddings"""
        if self.collection is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.chroma_db_path)
                self.collection = client.get_collection("gic_insights")
            except:
                return []
        
        # Encode query
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        # Package results
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
        
        # Convert distance to similarity score (lower distance = higher similarity)
        scores = [1.0 / (1.0 + d) for d in distances]
        
        return list(zip(docs, metas, scores))
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[str, Dict, float]]:
        """Keyword search using BM25"""
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                doc = self.documents.iloc[idx]
                results.append((
                    doc['text'],
                    {'doc_id': doc['doc_id'], 'doc_type': doc['doc_type']},
                    float(scores[idx])
                ))
        
        return results
    
    def _reciprocal_rank_fusion(self, 
                                semantic_results: List[Tuple], 
                                keyword_results: List[Tuple], 
                                k: int = 5,
                                k_param: int = 60) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Reciprocal Rank Fusion (RRF) to combine rankings
        RRF score = sum(1 / (k + rank))
        """
        doc_scores = {}
        doc_metadata = {}
        doc_text = {}
        
        # Process semantic results
        for rank, (text, meta, score) in enumerate(semantic_results, start=1):
            doc_id = meta['doc_id']
            rrf_score = 1.0 / (k_param + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_metadata[doc_id] = meta
            doc_text[doc_id] = text
        
        # Process keyword results
        for rank, (text, meta, score) in enumerate(keyword_results, start=1):
            doc_id = meta['doc_id']
            rrf_score = 1.0 / (k_param + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_metadata[doc_id] = meta
            doc_text[doc_id] = text
        
        # Sort by fused score
        ranked_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Return top-k
        top_k_ids = ranked_doc_ids[:k]
        
        return (
            [doc_text[doc_id] for doc_id in top_k_ids],
            [doc_metadata[doc_id] for doc_id in top_k_ids],
            [doc_scores[doc_id] for doc_id in top_k_ids]
        )
    
    def rerank_documents(self, query: str, documents: List[str], 
                        metadatas: List[Dict], scores: List[float]) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Rerank documents using simple relevance scoring
        In production, you'd use a cross-encoder model
        """
        # Simple reranking: keyword overlap + semantic score
        query_terms = set(self._tokenize(query))
        
        reranked = []
        for doc, meta, score in zip(documents, metadatas, scores):
            doc_terms = set(self._tokenize(doc))
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            
            # Combined score: 70% semantic + 30% keyword overlap
            combined_score = 0.7 * score + 0.3 * overlap
            
            reranked.append((doc, meta, combined_score))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[2], reverse=True)
        
        docs, metas, scores = zip(*reranked) if reranked else ([], [], [])
        return list(docs), list(metas), list(scores)
    
    def compress_context(self, query: str, documents: List[str]) -> List[str]:
        """
        Contextual compression: Extract only relevant sentences
        Reduces noise and improves answer quality
        """
        query_terms = set(self._tokenize(query))
        compressed = []
        
        for doc in documents:
            sentences = doc.split('\n')
            relevant_sentences = []
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:
                    continue
                
                sentence_terms = set(self._tokenize(sentence))
                overlap = len(query_terms & sentence_terms)
                
                # Include if significant overlap or contains numbers/metrics
                if overlap >= 2 or any(char.isdigit() for char in sentence):
                    relevant_sentences.append(sentence.strip())
            
            # Keep top sentences + first 2 sentences for context
            context_sentences = sentences[:2] if sentences else []
            compressed_doc = '\n'.join(context_sentences + relevant_sentences[:8])
            compressed.append(compressed_doc)
        
        return compressed
    
    def generate_answer_advanced(self, query: str, contexts: List[str], 
                                 metadatas: List[Dict]) -> Dict:
        """
        Advanced answer generation with citation tracking and confidence scoring
        
        Returns:
            Dict with answer, citations, confidence, and metadata
        """
        if not contexts:
            return {
                'answer': self._generate_no_data_response(query),
                'citations': [],
                'confidence': 0.0,
                'docs_used': 0
            }
        
        # Build context block with citations
        context_block = self._build_context_block_advanced(contexts, metadatas)
        
        # Generate answer with Groq
        if self.groq_client:
            result = self._generate_with_groq_advanced(query, context_block, metadatas)
        else:
            result = self._generate_with_template_advanced(query, contexts, metadatas)
        
        return result
    
    def _build_context_block_advanced(self, contexts: List[str], metadatas: List[Dict]) -> str:
        """Build formatted context with document IDs and metadata"""
        context_parts = []
        
        for i, (text, meta) in enumerate(zip(contexts, metadatas)):
            doc_id = meta.get("doc_id", f"doc_{i}")
            doc_type = meta.get("doc_type", "unknown")
            
            header = f"[Document {i+1} | ID: {doc_id} | Type: {doc_type}]"
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def _generate_with_groq_advanced(self, query: str, context_block: str, 
                                    metadatas: List[Dict]) -> Dict:
        """Advanced generation with Groq using sophisticated prompting"""
        
        system_prompt = """You are an elite Insurance Analytics Copilot with expertise in GIC premium data analysis.

CORE PRINCIPLES:
1. **Grounded Responses**: Answer STRICTLY from the provided context documents
2. **Citation Discipline**: EVERY factual claim MUST include [Document ID] citation
3. **Numerical Precision**: Quote exact numbers from documents, never round or estimate
4. **No Hallucination**: If information is missing, explicitly state "Data not available"
5. **Structured Output**: Use clear markdown formatting with sections

RESPONSE STRUCTURE:
**Executive Summary**
[2-3 sentence overview of findings]

**Key Findings**
• [Finding 1 with specific metrics] [Doc ID]
• [Finding 2 with specific metrics] [Doc ID]
• [Finding 3 with specific metrics] [Doc ID]

**Detailed Analysis**
[In-depth explanation synthesizing the findings]

**Data Quality Notes**
[Any limitations, missing data, or caveats]

**Sources Referenced**
[List of document IDs used]

QUALITY CHECKS:
✓ All numbers have citations
✓ No speculative statements
✓ Clear section headers
✓ Professional tone"""

        user_prompt = f"""**USER QUERY:**
{query}

**CONTEXT DOCUMENTS:**
{context_block}

**INSTRUCTIONS:**
Analyze the context documents and provide a comprehensive, grounded answer to the user's query. Follow the response structure exactly. Ensure every factual claim is cited."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=1200,
                top_p=0.95
            )
            
            answer = response.choices[0].message.content
            
            # Extract citations
            citations = re.findall(r'\[([^\]]+)\]', answer)
            citations = [c for c in citations if 'doc' in c.lower() or any(m['doc_id'] in c for m in metadatas)]
            
            # Calculate confidence based on citation density
            words = len(answer.split())
            citation_density = len(citations) / max(words / 50, 1)  # Citations per 50 words
            confidence = min(0.95, 0.5 + citation_density * 0.3)  # 50-95% range
            
            return {
                'answer': answer,
                'citations': list(set(citations)),
                'confidence': round(confidence, 2),
                'docs_used': len(metadatas)
            }
            
        except Exception as e:
            print(f"  Groq API error: {e}")
            # Fallback to template
            return self._generate_with_template_advanced(query, [], metadatas)
    
    def _generate_with_template_advanced(self, query: str, contexts: List[str], 
                                        metadatas: List[Dict]) -> Dict:
        """Template-based generation with structure"""
        query_lower = query.lower()
        
        # Extract facts
        facts = []
        sources = []
        
        for ctx, meta in zip(contexts, metadatas):
            doc_id = meta.get("doc_id", "unknown")
            
            # Extract key metrics
            lines = ctx.split('\n')
            for line in lines:
                if len(line.strip()) > 20 and (':' in line or '%' in line or '₹' in line):
                    facts.append(f"{line.strip()} [{doc_id}]")
                    sources.append(doc_id)
                    if len(facts) >= 5:
                        break
        
        if not facts:
            return {
                'answer': self._generate_no_data_response(query),
                'citations': [],
                'confidence': 0.0,
                'docs_used': 0
            }
        
        # Build structured answer
        answer = f"""**Key Findings:**
{chr(10).join('• ' + fact for fact in facts[:5])}

**Analysis:**
The data shows {'risk patterns' if 'risk' in query_lower else 'growth trends' if 'growth' in query_lower else 'insurance metrics'} based on FY24 and FY25 premium data. Review these findings in context of overall portfolio strategy.

**Sources Referenced:**
{chr(10).join('• ' + src for src in set(sources))}"""
        
        return {
            'answer': answer,
            'citations': list(set(sources)),
            'confidence': 0.7,
            'docs_used': len(metadatas)
        }
    
    def _generate_no_data_response(self, query: str) -> str:
        """Generate response when no relevant data found"""
        return f"""**Data Not Available**

The query "{query}" could not be answered with the current knowledge base.

**Possible Reasons:**
• Information not present in processed GIC data (FY24-FY25 Apr-Oct)
• Query refers to companies/segments outside dataset
• Requires real-time or external data not captured

**Available Coverage:**
• Industry and segment premium trends
• Company-level growth metrics
• Risk classifications and portfolio analysis
• YoY comparisons (FY24 vs FY25)

**Suggestion:** Try rephrasing your query or ask about specific companies/segments in the dataset."""
    
    def ask(self, query: str, k: int = 5, verbose: bool = False) -> Dict:
        """
        Advanced query interface with full RAG pipeline
        
        Args:
            query: User question
            k: Number of documents to retrieve
            verbose: Print detailed pipeline info
            
        Returns:
            Dict with answer, citations, confidence, and metadata
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"QUERY: {query}")
            print(f"{'='*80}\n")
        
        # Step 1: Query Expansion
        if verbose:
            print("🔍 Step 1: Query Expansion")
        expanded_queries = self.expand_query(query)
        if verbose and len(expanded_queries) > 1:
            print(f"  Expanded to {len(expanded_queries)} variations")
            for i, q in enumerate(expanded_queries, 1):
                print(f"    {i}. {q}")
        
        # Step 2: Hybrid Retrieval (for each query variation)
        if verbose:
            print("\n🎯 Step 2: Hybrid Retrieval (Semantic + Keyword)")
        
        all_docs = []
        all_metas = []
        all_scores = []
        
        for exp_query in expanded_queries:
            docs, metas, scores = self.hybrid_retrieve(exp_query, k=k)
            all_docs.extend(docs)
            all_metas.extend(metas)
            all_scores.extend(scores)
        
        # Deduplicate by doc_id
        seen = set()
        unique_docs = []
        unique_metas = []
        unique_scores = []
        
        for doc, meta, score in zip(all_docs, all_metas, all_scores):
            doc_id = meta['doc_id']
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
                unique_metas.append(meta)
                unique_scores.append(score)
        
        # Keep top-k
        if len(unique_docs) > k:
            sorted_indices = np.argsort(unique_scores)[::-1][:k]
            unique_docs = [unique_docs[i] for i in sorted_indices]
            unique_metas = [unique_metas[i] for i in sorted_indices]
            unique_scores = [unique_scores[i] for i in sorted_indices]
        
        if verbose:
            print(f"  Retrieved {len(unique_docs)} unique documents")
            for i, meta in enumerate(unique_metas[:3], 1):
                print(f"    {i}. {meta['doc_id']} (score: {unique_scores[i-1]:.3f})")
        
        # Step 3: Reranking
        if verbose:
            print("\n📊 Step 3: Reranking Documents")
        
        reranked_docs, reranked_metas, reranked_scores = self.rerank_documents(
            query, unique_docs, unique_metas, unique_scores
        )
        
        if verbose:
            print(f"  Reranked {len(reranked_docs)} documents")
        
        # Step 4: Contextual Compression
        if verbose:
            print("\n✂️  Step 4: Contextual Compression")
        
        compressed_docs = self.compress_context(query, reranked_docs)
        
        if verbose:
            print(f"  Compressed {len(compressed_docs)} documents")
        
        # Step 5: Advanced Generation
        if verbose:
            print("\n🤖 Step 5: Answer Generation with Citations")
        
        result = self.generate_answer_advanced(query, compressed_docs, reranked_metas)
        
        if verbose:
            print(f"  Generated answer ({result['docs_used']} docs used)")
            print(f"  Citations: {len(result['citations'])}")
            print(f"  Confidence: {result['confidence']*100:.0f}%")
            print(f"\n{'='*80}\n")
        
        return result


def main():
    """Demo the Advanced RAG copilot"""
    
    print("="*80)
    print("🚀 ADVANCED RAG COPILOT - Production Grade")
    print("="*80)
    print()
    
    # Initialize
    print("Initializing system...\n")
    copilot = AdvancedRAGCopilot()
    
    # Ingest documents
    print("\nIngesting documents...")
    copilot.ingest_documents()
    
    # Test queries
    test_queries = [
        "Which insurers have risky growth patterns?",
        "What is the total industry premium in FY25 and how does it compare to FY24?",
        "Compare health and motor segments in terms of growth and market share",
        "Which companies are exposed to crop insurance risk?",
    ]
    
    print("="*80)
    print("TESTING ADVANCED RAG PIPELINE")
    print("="*80)
    
    for query in test_queries:
        result = copilot.ask(query, k=4, verbose=True)
        
        print("ANSWER:")
        print(result['answer'])
        print(f"\n📊 Metadata:")
        print(f"  • Documents Used: {result['docs_used']}")
        print(f"  • Citations: {len(result['citations'])}")
        print(f"  • Confidence: {result['confidence']*100:.0f}%")
        print("\n" + "="*80 + "\n")
    
    print("✓ Setup complete! Use copilot.ask('your question') to query")
    print("="*80)
    
    return copilot


if __name__ == "__main__":
    copilot = main()
