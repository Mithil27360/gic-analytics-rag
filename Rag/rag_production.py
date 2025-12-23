"""
Production-Ready Internal Insurance Analytics Copilot
Strictly compliant with data governance and analytical rigor
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()


class ProductionAnalyticsCopilot:
    """
    Internal Insurance Analytics Copilot
    Regulated, data-sensitive environment with strict correctness requirements
    """
    
    # Production System Prompt
    SYSTEM_PROMPT = """You are an Internal Insurance Analytics Copilot.
You operate in a regulated, data-sensitive environment.

Your job is to answer analytical questions using ONLY the processed outputs
derived from the GIC YTD Premium Analysis (Apr–Oct FY25 vs FY24).

You must prioritize correctness, explainability, and scope clarity.
You are NOT a consumer advisory chatbot.

DATA COVERAGE & DEFINITIONS:

Time Period:
- FY25 YTD (April–October) - PRIMARY ANALYSIS PERIOD
- FY24 YTD (April–October) - BASELINE FOR YoY COMPARISONS
- Full comparative data available for both years

Primary Dataset:
- Contains both FY24 and FY25 data for year-over-year analysis
- 34 insurance companies tracked
- Monthly YTD premiums from April through October for both years

Segments (Top-level, mutually exclusive) with FY25 YTD Oct premiums:
- health: ₹76,564.07 Cr (39.3% share, #1 by premium)
- motor_total: ₹58,992.83 Cr (30.3% share, #2 by premium)  
- misc: ₹20,421.84 Cr (10.5% share, includes crop+credit+other)
- fire: ₹19,237.43 Cr (9.9% share)
- personal_accident: ₹7,504.74 Cr (3.9% share)
- engineering: ₹3,963.90 Cr (2.0% share)
- liability: ₹3,812.98 Cr (2.0% share)
- marine_total: ₹3,565.73 Cr (1.8% share)
- aviation: ₹617.28 Cr (0.3% share)

CRITICAL RULES:
1. All premiums are YTD unless explicitly stated
2. "motor_total" already includes Motor OD + Motor TP
3. "marine_total" already includes Marine Cargo + Marine Hull
4. "misc" is combined (crop, credit guarantee, other misc)
5. NEVER sum sub-components into totals again
6. Do NOT infer profitability, loss ratios, or margins
7. Do NOT extrapolate beyond October FY25
8. If data is missing, respond: "Data not available in the provided analysis"

DERIVED ANALYTICS (with YoY comparisons):
- FY25 YTD industry premium: ₹194,680.80 Cr
- FY24 YTD industry premium: ₹183,474.84 Cr  
- YoY growth (FY25 vs FY24, Oct): 6.11%
- Historical baseline: Full FY24 monthly data available for comparison
- Growth is non-uniform across segments and insurers

ANSWER FORMAT (MANDATORY):

**Answer:**
[Direct, factual response]

**Key Data:**
• [Exact figures with ₹ Cr, %, ranks]
• [More data points]

**Business Interpretation:**
[What this means for growth quality, risk, or stability]

**Scope Note:**
This analysis uses FY25 YTD (Apr-Oct) premium data. [State exclusions/limitations]

**Sources:** [Document IDs referenced]

BEHAVIORAL CONSTRAINTS:
- Do NOT give consumer advice
- Do NOT guess or estimate missing numbers
- Do NOT contradict established segment rankings
- Be concise, analytical, and professional
- If ambiguous, ask for clarification

Your value is CORRECTNESS, not creativity."""

    def __init__(self, knowledge_base_path: str = None, chroma_db_path: str = "./chroma_db"):
        """Initialize Production Analytics Copilot"""
        if knowledge_base_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_base_path = os.path.join(base_dir, "data", "processed", "rag_knowledge_base.csv")
        
        self.knowledge_base_path = knowledge_base_path
        self.chroma_db_path = chroma_db_path
        self.documents = None
        self.collection = None
        self.groq_client = None
        self.bm25 = None
        
        self._load_documents()
        self._initialize_embedder()
        self._initialize_bm25()
        self._initialize_groq()
    
    def _load_documents(self):
        """Load knowledge base"""
        if not Path(self.knowledge_base_path).exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")
        self.documents = pd.read_csv(self.knowledge_base_path)
        print(f"✓ Loaded {len(self.documents)} documents")
    
    def _initialize_embedder(self):
        """Initialize embeddings"""
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedder initialized")
    
    def _initialize_bm25(self):
        """Initialize BM25 keyword search"""
        try:
            from rank_bm25 import BM25Okapi
            texts = self.documents['text'].tolist()
            tokenized = [re.findall(r'\b\w+\b', doc.lower()) for doc in texts]
            self.bm25 = BM25Okapi(tokenized)
            print("✓ BM25 initialized")
        except ImportError:
            print("⚠️  rank-bm25 not available")
            self.bm25 = None
    
    def _initialize_groq(self):
        """Initialize Groq LLM"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                print("⚠️  GROQ_API_KEY not configured")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=api_key)
                print("✓ Groq LLM initialized (Production Mode)")
        except Exception as e:
            print(f"ERROR: {e}")
            self.groq_client = None
    
    def ingest_documents(self):
        """Ingest into ChromaDB"""
        import chromadb
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        
        try:
            self.collection = client.get_collection(name="gic_insights")
            print(f"✓ Vector database loaded ({self.collection.count()} docs)\n")
            return
        except:
            pass
        
        self.collection = client.create_collection(name="gic_insights")
        texts = self.documents["text"].tolist()
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            ids=self.documents["doc_id"].tolist(),
            metadatas=self.documents[["doc_type", "doc_id"]].to_dict("records")
        )
        print(f"✓ Ingested {len(texts)} documents\n")
    
    def hybrid_retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """Hybrid retrieval (semantic + keyword)"""
        # Semantic
        sem_results = self._semantic_search(query, k=k*2)
        
        # Keyword
        kw_results = []
        if self.bm25:
            kw_results = self._keyword_search(query, k=k*2)
        
        # Fusion
        return self._reciprocal_rank_fusion(sem_results, kw_results, k=k)
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple]:
        """Semantic search via ChromaDB"""
        if not self.collection:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_db_path)
            self.collection = client.get_collection("gic_insights")
        
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
        scores = [1.0 / (1.0 + d) for d in distances]
        
        return list(zip(docs, metas, scores))
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple]:
        """BM25 search"""
        if not self.bm25:
            return []
        
        tokens = re.findall(r'\b\w+\b', query.lower())
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents.iloc[idx]
                results.append((
                    doc['text'],
                    {'doc_id': doc['doc_id'], 'doc_type': doc['doc_type']},
                    float(scores[idx])
                ))
        return results
    
    def _reciprocal_rank_fusion(self, sem: List[Tuple], kw: List[Tuple], k: int = 5) -> Tuple:
        """RRF merge"""
        doc_scores, doc_metadata, doc_text = {}, {}, {}
        
        for rank, (text, meta, score) in enumerate(sem, start=1):
            doc_id = meta['doc_id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            doc_metadata[doc_id] = meta
            doc_text[doc_id] = text
        
        for rank, (text, meta, score) in enumerate(kw, start=1):
            doc_id = meta['doc_id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            doc_metadata[doc_id] = meta
            doc_text[doc_id] = text
        
        ranked = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:k]
        
        return (
            [doc_text[d] for d in ranked],
            [doc_metadata[d] for d in ranked],
            [doc_scores[d] for d in ranked]
        )
    
    def generate_production_answer(self, query: str, contexts: List[str], metadatas: List[Dict]) -> Dict:
        """Generate production-grade answer with strict compliance"""
        if not contexts:
            return {
                'answer': self._no_data_response(query),
                'citations': [],
                'confidence': 'Low',
                'docs_used': 0
            }
        
        if self.groq_client:
            return self._groq_production(query, contexts, metadatas)
        else:
            return self._template_production(query, contexts, metadatas)
    
    def _groq_production(self, query: str, contexts: List[str], metadatas: List[Dict]) -> Dict:
        """Production generation with Groq"""
        # Build context
        context_block = "\n\n".join([
            f"[Document {i+1}: {m['doc_id']}]\n{ctx}" 
            for i, (ctx, m) in enumerate(zip(contexts, metadatas))
        ])
        
        user_prompt = f"""User Question: {query}

Context Documents:
{context_block}

Provide a factually correct, analytically rigorous answer following the mandatory format."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,  # Very low for maximum factual accuracy
                max_tokens=700
            )
            
            answer = response.choices[0].message.content
            citations = list(set(re.findall(r'\[Document\s+\d+:\s*([^\]]+)\]', answer)))
            
            return {
                'answer': answer,
                'citations': citations,
                'confidence': 'High',
                'docs_used': len(metadatas),
                'retrieval_debug': [m['doc_id'] for m in metadatas]
            }
            
        except Exception as e:
            print(f"Groq error: {e}")
            return self._template_production(query, contexts, metadatas)
    
    def _template_production(self, query: str, contexts: List[str], metadatas: List[Dict]) -> Dict:
        """Template-based production answer"""
        facts = []
        sources = []
        
        for ctx, meta in zip(contexts, metadatas):
            doc_id = meta.get("doc_id", "")
            lines = [l.strip() for l in ctx.split('\n') if len(l.strip()) > 15]
            
            for line in lines:
                if any(marker in line for marker in [':', '₹', '%', 'Cr']):
                    facts.append(f"{line} [Document: {doc_id}]")
                    sources.append(doc_id)
                    if len(facts) >= 4:
                        break
        
        answer = f"""**Answer:**
{facts[0] if facts else 'Data available in retrieved documents.'}

**Key Data:**
{chr(10).join('• ' + f for f in facts[:3])}

**Business Interpretation:**
This reflects the analytical trends derived from FY25 YTD premium data.

**Scope Note:**
This analysis uses FY25 YTD (Apr-Oct) premium data. Does not include profitability, loss ratios, or full-year projections.

**Sources:** {', '.join(set(sources[:3]))}"""

        return {
            'answer': answer,
            'citations': list(set(sources)),
            'confidence': 'Medium',
            'docs_used': len(metadatas),
            'retrieval_debug': [m['doc_id'] for m in metadatas]
        }
    
    def _no_data_response(self, query: str) -> str:
        """No data response"""
        return f"""**Answer:**
Data not available in the provided analysis.

**Scope Note:**
Knowledge base covers FY24-FY25 (Apr-Oct) GIC premium data for 34 companies across 9 segments.

**Suggestion:**
Try rephrasing or ask about specific companies/segments in the dataset.

**Confidence:** Low"""
    
    def ask(self, query: str, k: int = 5, verbose: bool = False) -> Dict:
        """
        Main production query interface
        
        Args:
            query: Analytical question
            k: Number of documents to retrieve
            verbose: Show pipeline details
            
        Returns:
            Dict with answer, citations, confidence, retrieval_debug
        """
        if verbose:
            print(f"\n{'='*80}\nQUERY: {query}\n{'='*80}\n")
        
        # Hybrid retrieval
        if verbose:
            print("🔍 Retrieving relevant documents...")
        
        docs, metas, scores = self.hybrid_retrieve(query, k=k)
        
        if verbose:
            print(f"  Retrieved {len(docs)} documents:")
            for i, (meta, score) in enumerate(zip(metas[:3], scores[:3]), 1):
                print(f"    {i}. {meta['doc_id']} (relevance: {score:.3f})")
            print()
        
        # Generate production answer
        if verbose:
            print("🤖 Generating production-grade answer...\n")
        
        result = self.generate_production_answer(query, docs, metas)
        
        if verbose:
            print(f"{'='*80}\n")
        
        return result


def main():
    """Production system demo"""
    print("="*80)
    print("🏢 PRODUCTION ANALYTICS COPILOT")
    print("Internal Insurance Analytics | Regulated Environment")
    print("="*80)
    print()
    
    copilot = ProductionAnalyticsCopilot()
    copilot.ingest_documents()
    
    # Critical test queries
    test_queries = [
        "Which segment has the highest premiums in FY25?",
        "What is the total industry premium and YoY growth?",
        "Which companies have high crop insurance exposure?",
        "Compare health and motor segments",
    ]
    
    print("="*80)
    print("PRODUCTION TESTING - CRITICAL QUERIES")
    print("="*80)
    
    for query in test_queries:
        result = copilot.ask(query, k=5, verbose=True)
        print("ANSWER:")
        print(result['answer'])
        print(f"\n📊 Metadata: Confidence={result['confidence']} | Docs={result['docs_used']}")
        print(f"📚 Sources: {', '.join(result['retrieval_debug'][:3])}")
        print("\n" + "="*80 + "\n")
    
    return copilot


if __name__ == "__main__":
    copilot = main()
