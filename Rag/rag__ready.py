

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


class InterviewReadyRAGCopilot:
    """
    Interview-Ready Insurance Analytics RAG Copilot
    Decision-oriented with business intelligence layer
    """
    
    def __init__(self, 
                 knowledge_base_path: str = None,
                 chroma_db_path: str = "./chroma_db"):
        """Initialize Interview-Ready RAG system"""
        # Set up paths
        if knowledge_base_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_base_path = os.path.join(base_dir, "data", "processed", "rag_knowledge_base.csv")
        
        self.knowledge_base_path = knowledge_base_path
        self.chroma_db_path = chroma_db_path
        self.documents = None
        self.collection = None
        self.groq_client = None
        self.bm25 = None
        
        # Load documents
        self._load_documents()
        
        # Initialize embedder
        self._initialize_embedder()
        
        # Initialize BM25
        self._initialize_bm25()
        
        # Initialize Groq
        self._initialize_groq()
    
    def _load_documents(self):
        """Load knowledge base documents"""
        if not Path(self.knowledge_base_path).exists():
            raise FileNotFoundError(f"Knowledge base not found at {self.knowledge_base_path}")
        
        self.documents = pd.read_csv(self.knowledge_base_path)
        print(f"✓ Loaded {len(self.documents)} documents")
    
    def _initialize_embedder(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Embedder initialized")
        except ImportError:
            raise ImportError("sentence-transformers not installed")
    
    def _initialize_bm25(self):
        """Initialize BM25 for keyword search"""
        try:
            from rank_bm25 import BM25Okapi
            texts = self.documents['text'].tolist()
            tokenized = [re.findall(r'\b\w+\b', doc.lower()) for doc in texts]
            self.bm25 = BM25Okapi(tokenized)
            print("✓ BM25 initialized")
        except ImportError:
            print("⚠️  rank-bm25 not installed. Hybrid search disabled.")
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
                print("✓ Groq LLM initialized")
        except Exception as e:
            print(f"ERROR: {e}")
            self.groq_client = None
    
    def ingest_documents(self):
        """Ingest documents into ChromaDB"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_db_path)
            
            try:
                self.collection = client.get_collection(name="gic_insights")
                print(f"✓ Loaded vector database ({self.collection.count()} docs)\n")
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
        except ImportError:
            raise ImportError("chromadb not installed")
    
    def classify_question(self, query: str) -> str:
        """Classify question type for template selection"""
        query_lower = query.lower()
        
        # Question type detection
        if any(word in query_lower for word in ['highest', 'top', 'most', 'largest', 'rank', 'leader']):
            return 'ranking'
        elif any(word in query_lower for word in ['risk', 'risky', 'exposure', 'vulnerable', 'danger']):
            return 'risk'
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'between']):
            return 'comparison'
        elif any(word in query_lower for word in ['trend', 'pattern', 'direction', 'momentum']):
            return 'trend'
        elif any(word in query_lower for word in ['total', 'overall', 'industry', 'aggregate']):
            return 'aggregation'
        else:
            return 'general'
    
    def hybrid_retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """Hybrid retrieval with semantic + keyword search"""
        # Semantic search
        semantic_results = self._semantic_search(query, k=k*2)
        
        # Keyword search
        keyword_results = []
        if self.bm25:
            keyword_results = self._keyword_search(query, k=k*2)
        
        # Fusion
        fused = self._reciprocal_rank_fusion(semantic_results, keyword_results, k=k)
        return fused
    
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
        """BM25 keyword search"""
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
    
    def _reciprocal_rank_fusion(self, sem_results: List[Tuple], kw_results: List[Tuple], 
                                k: int = 5) -> Tuple[List[str], List[Dict], List[float]]:
        """RRF to merge rankings"""
        doc_scores = {}
        doc_metadata = {}
        doc_text = {}
        
        for rank, (text, meta, score) in enumerate(sem_results, start=1):
            doc_id = meta['doc_id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            doc_metadata[doc_id] = meta
            doc_text[doc_id] = text
        
        for rank, (text, meta, score) in enumerate(kw_results, start=1):
            doc_id = meta['doc_id']
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            doc_metadata[doc_id] = meta
            doc_text[doc_id] = text
        
        ranked_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:k]
        
        return (
            [doc_text[did] for did in ranked_ids],
            [doc_metadata[did] for did in ranked_ids],
            [doc_scores[did] for did in ranked_ids]
        )
    
    def generate_decision_oriented_answer(self, query: str, contexts: List[str], 
                                         metadatas: List[Dict], question_type: str) -> Dict:
        """
        Generate decision-oriented answer with business implications
        Uses question-aware templates
        """
        if not contexts:
            return {
                'answer': self._no_data_response(query),
                'citations': [],
                'confidence': 0.0,
                'docs_used': 0,
                'question_type': question_type
            }
        
        if self.groq_client:
            result = self._groq_decision_oriented(query, contexts, metadatas, question_type)
        else:
            result = self._template_decision_oriented(query, contexts, metadatas, question_type)
        
        result['question_type'] = question_type
        return result
    
    def _groq_decision_oriented(self, query: str, contexts: List[str], 
                               metadatas: List[Dict], question_type: str) -> Dict:
        """Decision-oriented generation with Groq"""
        
        # Build context
        context_block = "\n\n".join([
            f"[Doc {i+1}: {m['doc_id']}]\n{ctx}" 
            for i, (ctx, m) in enumerate(zip(contexts, metadatas))
        ])
        
        # Question-specific system prompt
        system_prompt = f"""You are a senior Insurance Analytics Advisor analyzing GIC data.

CRITICAL RULES:
1. Answer ONLY from context documents - no speculation
2. EVERY number/claim needs [Doc ID] citation
3. Be concise, sharp, decision-oriented (not verbose)
4. Focus on "so what?" not just "what"

QUESTION TYPE: {question_type}

OUTPUT STRUCTURE (MANDATORY):

**Answer:**
[One clear sentence answering the question]

**Key Data:**
• [Metric 1 with number] [Doc ID]
• [Metric 2 with number] [Doc ID]

**Why It Matters:**
[Business implication - risk/profitability/strategy insight]

**Scope Note:**
This is based on FY25 YTD (Apr-Oct) premium data. [Mention limitations]

**Confidence:** [High/Medium/Low]

STYLE RULES:
- Short sentences
- No fluff or "according to documents" phrases
- Numbers with ₹ symbol and proper units (Cr)
- Lead with insight, not description"""

        user_prompt = f"""Query: {query}

Context:\n{context_block}

Provide a decision-oriented answer."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            answer = response.choices[0].message.content
            citations = list(set(re.findall(r'\[Doc\s+\d+:?\s*([^\]]+)\]', answer)))
            
            return {
                'answer': answer,
                'citations': citations,
                'confidence': 0.90,  # High for Groq
                'docs_used': len(metadatas),
                'retrieval_debug': [m['doc_id'] for m in metadatas]
            }
            
        except Exception as e:
            print(f"Groq error: {e}")
            return self._template_decision_oriented(query, contexts, metadatas, question_type)
    
    def _template_decision_oriented(self, query: str, contexts: List[str], 
                                   metadatas: List[Dict], question_type: str) -> Dict:
        """Template-based decision answer"""
        facts = []
        sources = []
        
        for ctx, meta in zip(contexts, metadatas):
            doc_id = meta.get("doc_id", "")
            lines = [l.strip() for l in ctx.split('\n') if len(l.strip()) > 15]
            
            for line in lines:
                if ':' in line or '₹' in line or '%' in line:
                    facts.append(f"{line} [Doc: {doc_id}]")
                    sources.append(doc_id)
                    if len(facts) >= 4:
                        break
        
        # Business logic by question type
        if question_type == 'risk':
            implication = "Higher risk concentration indicates vulnerability to segment-specific shocks. Diversification reduces tail risk."
        elif question_type == 'ranking':
            implication = "Premium volume does not equal profitability. High-volume segments may have thin margins or regulatory pricing constraints."
        elif question_type == 'comparison':
            implication = "Different risk-return profiles. Consider growth sustainability, claims experience, and operating margins alongside volume."
        else:
            implication = "Review in context of portfolio strategy and competitive positioning."
        
        answer = f"""**Answer:**
{facts[0] if facts else 'Data available in retrieved documents.'}

**Key Data:**
{chr(10).join('• ' + f for f in facts[:3])}

**Why It Matters:**
{implication}

**Scope Note:**
Based on FY25 YTD (Apr-Oct) premium data. Does not include profitability, loss ratios, or full-year projections.

**Confidence:** Medium

**Sources:** {', '.join(set(sources[:3]))}"""

        return {
            'answer': answer,
            'citations': list(set(sources)),
            'confidence': 0.75,
            'docs_used': len(metadatas),
            'retrieval_debug': [m['doc_id'] for m in metadatas]
        }
    
    def _no_data_response(self, query: str) -> str:
        """No data found response"""
        return f"""**Answer:**
Unable to answer "{query}" from current knowledge base.

**Scope Note:**
Knowledge base covers FY24-FY25 (Apr-Oct) GIC premium data for 34 companies across 9 segments.

**Suggestion:**
Try rephrasing or ask about specific companies/segments in the dataset.

**Confidence:** Low"""
    
    def ask(self, query: str, k: int = 5, verbose: bool = False) -> Dict:
        """
        Main query interface - decision-oriented with business intelligence
        
        Returns:
            Dict with answer, citations, confidence, question_type, retrieval_debug
        """
        if verbose:
            print(f"\n{'='*80}\nQUERY: {query}\n{'='*80}\n")
        
        # Classify question
        question_type = self.classify_question(query)
        if verbose:
            print(f"📋 Question Type: {question_type}\n")
        
        # Hybrid retrieval
        if verbose:
            print("🔍 Hybrid Retrieval...")
        docs, metas, scores = self.hybrid_retrieve(query, k=k)
        
        if verbose:
            print(f"  Retrieved {len(docs)} documents:")
            for i, (meta, score) in enumerate(zip(metas[:3], scores[:3]), 1):
                print(f"    {i}. {meta['doc_id']} (score: {score:.3f})")
            print()
        
        # Generate decision-oriented answer
        if verbose:
            print("🤖 Generating decision-oriented answer...\n")
        
        result = self.generate_decision_oriented_answer(query, docs, metas,question_type)
        
        if verbose:
            print(f"{'='*80}\n")
        
        return result


def main():
    """Demo Interview-Ready RAG"""
    print("="*80)
    print("🎯 INTERVIEW-READY RAG COPILOT")
    print("Decision-Oriented | Business Intelligence Layer")
    print("="*80)
    print()
    
    copilot = InterviewReadyRAGCopilot()
    copilot.ingest_documents()
    
    test_queries = [
        "Which segment has the highest premiums in FY25?",
        "Which companies are exposed to crop insurance risk?",
        "Compare health and motor segments",
    ]
    
    print("="*80)
    print("TESTING DECISION-ORIENTED ANSWERS")
    print("="*80)
    
    for query in test_queries:
        result = copilot.ask(query, k=4, verbose=True)
        print("ANSWER:")
        print(result['answer'])
        print(f"\n📊 Meta: Type={result['question_type']} | Docs={result['docs_used']} | Confidence={result['confidence']*100:.0f}%")
        print(f"📚 Sources: {', '.join(result['retrieval_debug'][:3])}")
        print("\n" + "="*80 + "\n")
    
    return copilot


if __name__ == "__main__":
    copilot = main()
