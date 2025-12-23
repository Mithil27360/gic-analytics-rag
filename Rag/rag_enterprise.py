"""
Enterprise Retrieval-Augmented Analytics Engine
Deterministic, auditable, evidence-only system
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import os
import re
from dotenv import load_dotenv

load_dotenv()


class EnterpriseRAGEngine:
    """
    Enterprise RAG: Deterministic, auditable, retrieval-first
    LLM used ONLY for formatting and aggregation, NOT reasoning
    """
    
    # Enterprise System Prompt - GIC Internal Analytics Engine
    SYSTEM_PROMPT = """You are the GIC Internal Analytics Engine.
Your role is to answer questions strictly based on the provided Insurance Data Context.

OPERATIONAL GUARDRAILS (NON-NEGOTIABLE):
1. Sector Classification:
   - NEVER guess if a company is Public/Private.
   - Look for the "Entity Type" or "Sector" field in the retrieved text.
   - Public Sector (PSU) = New India, Oriental, National, United, AIC, ECGC.
   - All others are Private Sector.

2. Temporal Logic (Trends):
   - If asked about "Momentum", "Run Rate", or "Trend", look for the "Monthly Momentum" field.
   - "Drag" = Negative Monthly Flow or Negative YoY Growth.

3. Data Integrity (HARDCODED BASELINES):
   - FY24 Industry Total Baseline: Rs.183,474.84 Cr.
   - FY25 Industry Total (Oct YTD): Rs.194,680.80 Cr.
   - Use these numbers for all industry-wide comparisons. Do not sum individual rows.

4. Safety and Scope:
   - NO Investment Advice (Stock/Shares).
   - NO Poems, Creative Writing, or Roleplay.
   - If data is missing (e.g., Incurred Claims Ratio), state "Data not available." Do not hallucinate.

OUTPUT FORMAT:
Answer: [Direct, factual answer. Max 2 sentences.]

Metrics:
- [Metric Name]: [Value] (vs [Prior Period] if avail) [Source_ID]

Sources:
- [Source_ID]

CRITICAL: If user asks for "Top N" (e.g., Top 5, Top 3):
- Extract ONLY the first N items from the ranking document
- Do NOT return all 10 items if user asked for fewer
- Example: "Top 5 health players" = show ONLY first 5 companies from health_segment_rankings

FORMATTING RULES:
- Exact values only (no rounding unless present in source)
- Currency format: Rs.X,XXX.XX Cr
- No emojis
- No headings beyond those specified
- No confidence scores
- No disclaimers

FAIL-SAFE:
If query is ambiguous:
"Query insufficiently specified. Please define metric, entity, or period."
"""

    def __init__(self, knowledge_base_path: str = None, chroma_db_path: str = "./chroma_db"):
        """Initialize Enterprise RAG Engine"""
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
        """Initialize BM25"""
        try:
            from rank_bm25 import BM25Okapi
            texts = self.documents['text'].tolist()
            tokenized = [re.findall(r'\b\w+\b', doc.lower()) for doc in texts]
            self.bm25 = BM25Okapi(tokenized)
            print("✓ BM25 initialized")
        except ImportError:
            self.bm25 = None
    
    def _initialize_groq(self):
        """Initialize Groq LLM"""
        try:
            # CRITICAL: Explicitly load .env file
            from dotenv import load_dotenv
            load_dotenv(override=True)  # Force reload
            
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            
            # More permissive check - accept any non-placeholder key
            if not api_key:
                print("⚠️  GROQ_API_KEY not found in environment")
                self.groq_client = None
            elif api_key == "your_groq_api_key_here":
                print("⚠️  GROQ_API_KEY is placeholder value")
                self.groq_client = None
            elif len(api_key) < 40:  # Groq keys are ~56 chars
                print(f"⚠️  GROQ_API_KEY too short ({len(api_key)} chars)")
                self.groq_client = None
            else:
                self.groq_client = Groq(api_key=api_key)
                print(f"✓ Groq LLM initialized (Enterprise Deterministic Mode) - Key: {api_key[:20]}...")
        except Exception as e:
            print(f"ERROR initializing Groq: {e}")
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
    
    def hybrid_retrieve(self, query: str, k: int = 6) -> Tuple[List[str], List[Dict], List[float]]:
        """Hybrid retrieval"""
        sem_results = self._semantic_search(query, k=k*2)
        
        kw_results = []
        if self.bm25:
            kw_results = self._keyword_search(query, k=k*2)
        
        return self._reciprocal_rank_fusion(sem_results, kw_results, k=k)
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple]:
        """Semantic search"""
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
    
    def _reciprocal_rank_fusion(self, sem: List[Tuple], kw: List[Tuple], k: int = 6) -> Tuple:
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
    
    def generate_deterministic_answer(self, query: str, contexts: List[str], metadatas: List[Dict]) -> Dict:
        """Generate deterministic, evidence-only answer"""
        if not contexts:
            return {
                'answer': "Data not available in indexed corpus.",
                'sources': []
            }
        
        if self.groq_client:
            return self._groq_deterministic(query, contexts, metadatas)
        else:
            return self._template_deterministic(query, contexts, metadatas)
    
    def _groq_deterministic(self, query: str, contexts: List[str], metadatas: List[Dict]) -> Dict:
        """Deterministic generation with Groq"""
        # Build context
        context_block = "\n\n".join([
            f"[{m['doc_id']}]\n{ctx}" 
            for ctx, m in zip(contexts, metadatas)
        ])
        
        user_prompt = f"""Query: {query}

Retrieved Documents:
{context_block}

Provide answer per strict output contract."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Absolute determinism
                max_tokens=400
            )
            
            answer = response.choices[0].message.content
            sources = list(set([m['doc_id'] for m in metadatas]))
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            print(f"Groq error: {e}")
            return self._template_deterministic(query, contexts, metadatas)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations"""
        
        # Company name synonyms for entity resolution
        COMPANY_ALIASES = {
            'star': 'Star Health & Allied Insurance Co Ltd',
            'star health': 'Star Health & Allied Insurance Co Ltd',
            'digit': 'Go Digit General Insurance Limited',
            'go digit': 'Go Digit General Insurance Limited',
            'new india': 'The New India Assurance Co Ltd',
            'oriental': 'The Oriental Insurance Co Ltd',
            'national': 'National Insurance Co Ltd',
            'united india': 'United India Insurance Co Ltd',
            'icici': 'ICICI Lombard General Insurance Co Ltd',
            'bajaj': 'Bajaj General Insurance Limited',
            'tata': 'Tata AIG General Insurance Company Limited',
            'reliance': 'Reliance General Insurance Co Ltd',
            'hdfc ergo': 'HDFC ERGO General Insurance Co Ltd',
            'sbi': 'SBI General Insurance Co Ltd',
            'niva bupa': 'Niva Bupa Health Insurance Company Limited',
            'care': 'Care Health Insurance Ltd'
        }
        
        # Semantic keywords for reasoning
        SEMANTIC_MAPPINGS = {
            'drag': 'negative growth declining lagging',
            'lagging': 'negative growth declining drag',
            'underperforming': 'negative growth declining',
            'best performing': 'highest growth fastest growing',
            'worst performing': 'negative growth declining'
        }
        
        expanded = [query]
        query_lower = query.lower()
        
        # Expand company aliases
        for alias, full_name in COMPANY_ALIASES.items():
            if alias in query_lower:
                expanded.append(query.lower().replace(alias, full_name.lower()))
        
        # Expand semantic terms
        for term, expansion in SEMANTIC_MAPPINGS.items():
            if term in query_lower:
                expanded.append(query + ' ' + expansion)
        
        return expanded[:3]  # Limit to 3 variations
    
    def _template_deterministic(self, query: str, contexts: List[str], metadatas: List[Dict]) -> Dict:
        """Template deterministic answer"""
        metrics = []
        sources = set()
        
        for ctx, meta in zip(contexts, metadatas):
            doc_id = meta.get("doc_id", "")
            sources.add(doc_id)
            
            lines = [l.strip() for l in ctx.split('\n') if len(l.strip()) > 15]
            
            for line in lines:
                if any(marker in line for marker in ['₹', '%', 'Cr', ':']):
                    metrics.append(f"- {line} [{doc_id}]")
                    if len(metrics) >= 4:
                        break
        
        answer = f"""Answer:
{metrics[0].replace('- ', '').split('[')[0].strip() if metrics else 'Data retrieved from indexed corpus.'}

Metrics:
{chr(10).join(metrics[:4])}

Sources:
{chr(10).join('- ' + s for s in list(sources)[:3])}"""

        return {
            'answer': answer,
            'sources': list(sources)
        }
    
    def query(self, query: str, k: int = 6, verbose: bool = False) -> Dict:
        """Query the RAG system with enhanced intent detection"""
        
        # Detect query intent and boost relevant doc types
        query_lower = query.lower()
        boost_filters = {}
        
        # Industry-level queries
        if any(term in query_lower for term in ['industry', 'total market', 'overall', 'fy25 vs fy24', 'compare fy']):
            boost_filters['doc_type'] = 'industry_overview'
        
        # Company ranking queries
        elif any(term in query_lower for term in ['top', 'rank', 'biggest', 'largest', 'leader']):
            if 'health' in query_lower and 'segment' in query_lower:
                boost_filters['doc_type'] = 'health_rankings'
            else:
                boost_filters['doc_type'] = 'company_rankings'
        
        # Segment comparison queries
        elif any(term in query_lower for term in ['segment', 'compare all', 'all segments']):
            boost_filters['doc_type'] = 'segment_comparison'
        
        # Retrieve documents with expanded queries
        expanded_queries = self._expand_query(query)
        all_results = []
        
        for exp_query in expanded_queries:
            # The original `hybrid_retrieve` returns (docs, metas, scores)
            # We need to convert this to a list of dicts for easier processing
            docs, metas, scores = self.hybrid_retrieve(exp_query, k=k)
            for doc, meta, score in zip(docs, metas, scores):
                all_results.append({'text': doc, 'metadata': meta, 'score': score})
        
        # Apply boosting based on intent
        if boost_filters:
            # Prioritize documents matching the boost filter
            boosted = [r for r in all_results if any(r['metadata'].get(key) == val for key, val in boost_filters.items())]
            others = [r for r in all_results if r not in boosted]
            all_results = boosted + others
        
        # Deduplicate and take top k
        seen_ids = set()
        unique_results = []
        for r in all_results:
            doc_id = r['metadata'].get('doc_id', r['text'][:50]) # Use doc_id or first 50 chars as unique identifier
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(r)
                if len(unique_results) >= k:
                    break
        
        contexts = [r['text'] for r in unique_results]
        metadatas = [r['metadata'] for r in unique_results]
        
        if verbose:
            print(f"\nRetrieved {len(contexts)} documents")
            for i, meta in enumerate(metadatas[:3]):
                print(f"{i+1}. {meta.get('doc_type', 'unknown')} - {meta.get('doc_id', '')}")
        
        # Generate answer
        return self.generate_deterministic_answer(query, contexts, metadatas)
    

def main():
    """Enterprise system demo"""
    print("="*80)
    print("ENTERPRISE RAG ENGINE - DETERMINISTIC MODE")
    print("="*80)
    print()
    
    engine = EnterpriseRAGEngine()
    engine.ingest_documents()
    
    # Critical test queries
    test_queries = [
        "What is the total industry premium for FY25?",
        "Which segment has the highest premium?",
        "Compare FY25 vs FY24 industry growth",
    ]
    
    print("="*80)
    print("DETERMINISTIC OUTPUT TESTS")
    print("="*80)
    
    for query in test_queries:
        result = engine.query(query, k=6, verbose=True)
        print("OUTPUT:")
        print(result['answer'])
        print("\n" + "="*80 + "\n")
    
    return engine


if __name__ == "__main__":
    engine = main()
