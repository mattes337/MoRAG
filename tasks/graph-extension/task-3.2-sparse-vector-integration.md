# Task 3.2: Sparse Vector Integration

**Phase**: 3 - Retrieval Integration  
**Priority**: Medium  
**Estimated Time**: 6-8 days total  
**Dependencies**: Task 3.1 (Hybrid Retrieval System)

## Overview

This task implements sparse vector retrieval capabilities and integrates them with the existing dense vector and graph-based retrieval systems. Sparse vectors (like BM25) excel at keyword matching and can complement dense vectors for improved retrieval performance, especially for rare terms and specific queries.

## Subtasks

### 3.2.1: Sparse Vector Implementation
**Estimated Time**: 3-4 days  
**Priority**: Medium

#### Implementation Steps

1. **BM25 Implementation**
   ```python
   # src/morag_retrieval/sparse/bm25.py
   from typing import List, Dict, Any, Optional, Tuple
   import numpy as np
   from dataclasses import dataclass
   import math
   import re
   from collections import Counter
   
   @dataclass
   class BM25Config:
       k1: float = 1.5  # Term frequency saturation parameter
       b: float = 0.75  # Document length normalization parameter
       epsilon: float = 0.25  # Smoothing parameter
   
   @dataclass
   class BM25Document:
       id: str
       content: str
       tokens: List[str]
       term_freqs: Dict[str, int]
       doc_length: int
       metadata: Dict[str, Any] = None
   
   class BM25Retriever:
       def __init__(self, config: BM25Config = None):
           self.config = config or BM25Config()
           self.documents: List[BM25Document] = []
           self.doc_count: int = 0
           self.avg_doc_length: float = 0.0
           self.idf: Dict[str, float] = {}
           self.tokenizer = self._default_tokenizer
           self.logger = logging.getLogger(__name__)
           self.is_indexed = False
       
       def add_documents(self, documents: List[Dict[str, Any]]) -> None:
           """Add documents to the retriever."""
           for doc in documents:
               self._add_document(doc)
           
           self.is_indexed = False
       
       def _add_document(self, document: Dict[str, Any]) -> None:
           """Add a single document to the retriever."""
           content = document.get('content', '')
           doc_id = document.get('id', str(len(self.documents)))
           
           # Tokenize and count terms
           tokens = self.tokenizer(content)
           term_freqs = Counter(tokens)
           doc_length = len(tokens)
           
           # Create BM25Document
           bm25_doc = BM25Document(
               id=doc_id,
               content=content,
               tokens=tokens,
               term_freqs=dict(term_freqs),
               doc_length=doc_length,
               metadata=document.get('metadata', {})
           )
           
           self.documents.append(bm25_doc)
       
       def build_index(self) -> None:
           """Build the BM25 index."""
           self.doc_count = len(self.documents)
           
           if self.doc_count == 0:
               self.logger.warning("No documents to index")
               return
           
           # Calculate average document length
           total_length = sum(doc.doc_length for doc in self.documents)
           self.avg_doc_length = total_length / self.doc_count
           
           # Calculate IDF for each term
           term_doc_freq = {}
           all_terms = set()
           
           for doc in self.documents:
               doc_terms = set(doc.tokens)
               all_terms.update(doc_terms)
               
               for term in doc_terms:
                   term_doc_freq[term] = term_doc_freq.get(term, 0) + 1
           
           # Calculate IDF using the BM25 formula
           for term in all_terms:
               n_docs_with_term = term_doc_freq.get(term, 0)
               idf = math.log((self.doc_count - n_docs_with_term + 0.5) / 
                             (n_docs_with_term + 0.5) + 1.0)
               self.idf[term] = max(0.0, idf)  # Ensure non-negative IDF
           
           self.is_indexed = True
           self.logger.info(f"BM25 index built with {self.doc_count} documents and {len(all_terms)} terms")
       
       def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
           """Search for documents matching the query."""
           if not self.is_indexed:
               self.build_index()
           
           if self.doc_count == 0:
               return []
           
           # Tokenize query
           query_tokens = self.tokenizer(query)
           query_terms = Counter(query_tokens)
           
           # Calculate scores for each document
           scores = []
           for i, doc in enumerate(self.documents):
               score = self._score_document(query_terms, doc)
               scores.append((i, score))
           
           # Sort by score and return top_k
           scores.sort(key=lambda x: x[1], reverse=True)
           top_results = scores[:top_k]
           
           # Format results
           results = []
           for doc_idx, score in top_results:
               if score > 0:  # Only include documents with positive scores
                   doc = self.documents[doc_idx]
                   results.append({
                       'id': doc.id,
                       'content': doc.content,
                       'score': score,
                       'metadata': doc.metadata
                   })
           
           return results
       
       def _score_document(self, query_terms: Counter, doc: BM25Document) -> float:
           """Calculate BM25 score for a document given query terms."""
           score = 0.0
           doc_length_norm = doc.doc_length / self.avg_doc_length
           
           for term, query_tf in query_terms.items():
               if term in self.idf:
                   idf = self.idf[term]
                   doc_tf = doc.term_freqs.get(term, 0)
                   
                   # BM25 scoring formula
                   numerator = doc_tf * (self.config.k1 + 1)
                   denominator = doc_tf + self.config.k1 * (1 - self.config.b + self.config.b * doc_length_norm)
                   term_score = idf * numerator / (denominator + self.config.epsilon)
                   
                   score += term_score
           
           return score
       
       def _default_tokenizer(self, text: str) -> List[str]:
           """Default tokenization function."""
           # Convert to lowercase and split on non-alphanumeric characters
           text = text.lower()
           tokens = re.findall(r'\w+', text)
           
           # Filter out very short tokens and common stopwords
           tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
           
           return tokens
   
   # Common English stopwords
   STOPWORDS = set([
       'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
       'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
       'when', 'who', 'how', 'where', 'why', 'is', 'are', 'was', 'were', 'be', 'been',
       'have', 'has', 'had', 'do', 'does', 'did', 'to', 'at', 'in', 'on', 'by', 'with',
       'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
       'above', 'below', 'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again',
       'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
       'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
       'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'should', 'now'
   ])
   ```

2. **Sparse Vector Storage**
   ```python
   # src/morag_retrieval/sparse/storage.py
   import pickle
   import os
   from typing import List, Dict, Any, Optional
   import numpy as np
   from scipy.sparse import csr_matrix, save_npz, load_npz
   
   class SparseVectorStorage:
       def __init__(self, storage_dir: str):
           self.storage_dir = storage_dir
           self.doc_ids = []
           self.term_to_idx = {}
           self.idx_to_term = {}
           self.doc_vectors = None  # Will be a sparse matrix
           self.metadata = {}
           self.logger = logging.getLogger(__name__)
           
           os.makedirs(storage_dir, exist_ok=True)
       
       def add_documents(self, documents: List[Dict[str, Any]]) -> None:
           """Add documents to the sparse vector storage."""
           # Extract document IDs and term frequencies
           doc_ids = []
           doc_term_freqs = []
           doc_metadata = {}
           
           for doc in documents:
               doc_id = doc.get('id', str(len(self.doc_ids) + len(doc_ids)))
               doc_ids.append(doc_id)
               
               # Get term frequencies
               if 'term_freqs' in doc:
                   term_freqs = doc['term_freqs']
               else:
                   # Tokenize and count if term_freqs not provided
                   content = doc.get('content', '')
                   tokens = self._tokenize(content)
                   term_freqs = {}
                   for token in tokens:
                       term_freqs[token] = term_freqs.get(token, 0) + 1
               
               doc_term_freqs.append(term_freqs)
               doc_metadata[doc_id] = doc.get('metadata', {})
           
           # Update term dictionary
           for term_freqs in doc_term_freqs:
               for term in term_freqs:
                   if term not in self.term_to_idx:
                       idx = len(self.term_to_idx)
                       self.term_to_idx[term] = idx
                       self.idx_to_term[idx] = term
           
           # Create sparse matrix for new documents
           num_terms = len(self.term_to_idx)
           num_docs = len(doc_ids)
           
           # Prepare data for CSR matrix
           data = []
           row_indices = []
           col_indices = []
           
           for doc_idx, term_freqs in enumerate(doc_term_freqs):
               for term, freq in term_freqs.items():
                   if term in self.term_to_idx:
                       term_idx = self.term_to_idx[term]
                       data.append(freq)
                       row_indices.append(doc_idx)
                       col_indices.append(term_idx)
           
           # Create sparse matrix for new documents
           new_vectors = csr_matrix(
               (data, (row_indices, col_indices)),
               shape=(num_docs, num_terms)
           )
           
           # Merge with existing vectors if any
           if self.doc_vectors is not None:
               # Ensure same dimensionality
               if self.doc_vectors.shape[1] < num_terms:
                   # Expand existing matrix
                   expanded = csr_matrix(
                       (self.doc_vectors.data, self.doc_vectors.indices, self.doc_vectors.indptr),
                       shape=(self.doc_vectors.shape[0], num_terms)
                   )
                   self.doc_vectors = expanded
               
               # Concatenate matrices
               self.doc_vectors = csr_matrix(np.vstack([self.doc_vectors, new_vectors]))
           else:
               self.doc_vectors = new_vectors
           
           # Update document IDs and metadata
           self.doc_ids.extend(doc_ids)
           self.metadata.update(doc_metadata)
           
           self.logger.info(f"Added {len(doc_ids)} documents to sparse vector storage")
       
       def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
           """Search for documents matching the query."""
           if self.doc_vectors is None or len(self.doc_ids) == 0:
               return []
           
           # Tokenize query and create query vector
           tokens = self._tokenize(query)
           query_vector = np.zeros(len(self.term_to_idx))
           
           for token in tokens:
               if token in self.term_to_idx:
                   term_idx = self.term_to_idx[token]
                   query_vector[term_idx] += 1
           
           # Calculate dot product with all documents
           scores = self.doc_vectors.dot(query_vector)
           
           # Get top-k results
           if len(scores) <= top_k:
               top_indices = np.argsort(scores)[::-1]
           else:
               top_indices = np.argpartition(scores, -top_k)[-top_k:]
               top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
           
           # Format results
           results = []
           for idx in top_indices:
               if scores[idx] > 0:  # Only include documents with positive scores
                   doc_id = self.doc_ids[idx]
                   results.append({
                       'id': doc_id,
                       'score': float(scores[idx]),
                       'metadata': self.metadata.get(doc_id, {})
                   })
           
           return results
       
       def save(self) -> None:
           """Save the sparse vector storage to disk."""
           # Save document vectors
           save_npz(os.path.join(self.storage_dir, 'doc_vectors.npz'), self.doc_vectors)
           
           # Save document IDs
           with open(os.path.join(self.storage_dir, 'doc_ids.pkl'), 'wb') as f:
               pickle.dump(self.doc_ids, f)
           
           # Save term dictionary
           with open(os.path.join(self.storage_dir, 'term_dict.pkl'), 'wb') as f:
               pickle.dump({'term_to_idx': self.term_to_idx, 'idx_to_term': self.idx_to_term}, f)
           
           # Save metadata
           with open(os.path.join(self.storage_dir, 'metadata.pkl'), 'wb') as f:
               pickle.dump(self.metadata, f)
           
           self.logger.info(f"Saved sparse vector storage to {self.storage_dir}")
       
       def load(self) -> bool:
           """Load the sparse vector storage from disk."""
           try:
               # Load document vectors
               vectors_path = os.path.join(self.storage_dir, 'doc_vectors.npz')
               if os.path.exists(vectors_path):
                   self.doc_vectors = load_npz(vectors_path)
               else:
                   return False
               
               # Load document IDs
               with open(os.path.join(self.storage_dir, 'doc_ids.pkl'), 'rb') as f:
                   self.doc_ids = pickle.load(f)
               
               # Load term dictionary
               with open(os.path.join(self.storage_dir, 'term_dict.pkl'), 'rb') as f:
                   term_dict = pickle.load(f)
                   self.term_to_idx = term_dict['term_to_idx']
                   self.idx_to_term = term_dict['idx_to_term']
               
               # Load metadata
               with open(os.path.join(self.storage_dir, 'metadata.pkl'), 'rb') as f:
                   self.metadata = pickle.load(f)
               
               self.logger.info(f"Loaded sparse vector storage with {len(self.doc_ids)} documents")
               return True
           
           except Exception as e:
               self.logger.error(f"Error loading sparse vector storage: {str(e)}")
               return False
       
       def _tokenize(self, text: str) -> List[str]:
           """Simple tokenization function."""
           # Convert to lowercase and split on non-alphanumeric characters
           text = text.lower()
           tokens = re.findall(r'\w+', text)
           
           # Filter out very short tokens and common stopwords
           tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
           
           return tokens
   ```

#### Deliverables
- BM25 implementation for keyword-based retrieval
- Sparse vector storage system
- Tokenization and indexing utilities
- Persistence and loading mechanisms

### 3.2.2: Dense-Sparse Fusion
**Estimated Time**: 3-4 days  
**Priority**: Medium

#### Implementation Steps

1. **Hybrid Vector Retriever**
   ```python
   # src/morag_retrieval/hybrid/vector_fusion.py
   from typing import List, Dict, Any, Optional, Tuple
   from dataclasses import dataclass
   from morag_retrieval.vector import VectorRetriever
   from morag_retrieval.sparse import BM25Retriever, SparseVectorStorage
   
   @dataclass
   class HybridVectorConfig:
       dense_weight: float = 0.7
       sparse_weight: float = 0.3
       dense_top_k: int = 50
       sparse_top_k: int = 50
       final_top_k: int = 20
       fusion_method: str = "weighted"  # "weighted", "reciprocal_rank", "round_robin"
   
   class HybridVectorRetriever:
       def __init__(
           self,
           dense_retriever: VectorRetriever,
           sparse_retriever: Optional[BM25Retriever] = None,
           sparse_storage: Optional[SparseVectorStorage] = None,
           config: HybridVectorConfig = None
       ):
           self.dense_retriever = dense_retriever
           self.sparse_retriever = sparse_retriever
           self.sparse_storage = sparse_storage
           self.config = config or HybridVectorConfig()
           self.logger = logging.getLogger(__name__)
           
           if sparse_retriever is None and sparse_storage is None:
               raise ValueError("Either sparse_retriever or sparse_storage must be provided")
       
       async def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
           """Perform hybrid dense-sparse vector search."""
           if top_k is None:
               top_k = self.config.final_top_k
           
           try:
               # Perform dense vector search
               dense_results = await self.dense_retriever.search(
                   query, limit=self.config.dense_top_k
               )
               
               # Perform sparse vector search
               if self.sparse_retriever:
                   sparse_results = self.sparse_retriever.search(
                       query, top_k=self.config.sparse_top_k
                   )
               elif self.sparse_storage:
                   sparse_results = self.sparse_storage.search(
                       query, top_k=self.config.sparse_top_k
                   )
               else:
                   sparse_results = []
               
               # Fuse results
               if self.config.fusion_method == "weighted":
                   fused_results = self._weighted_fusion(
                       dense_results, sparse_results
                   )
               elif self.config.fusion_method == "reciprocal_rank":
                   fused_results = self._reciprocal_rank_fusion(
                       dense_results, sparse_results
                   )
               elif self.config.fusion_method == "round_robin":
                   fused_results = self._round_robin_fusion(
                       dense_results, sparse_results
                   )
               else:
                   # Default to weighted fusion
                   fused_results = self._weighted_fusion(
                       dense_results, sparse_results
                   )
               
               # Return top-k results
               return fused_results[:top_k]
           
           except Exception as e:
               self.logger.error(f"Error in hybrid vector search: {str(e)}")
               # Fallback to dense retrieval only
               return dense_results[:top_k] if dense_results else []
       
       def _weighted_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
           """Fuse results using weighted combination of scores."""
           # Normalize scores within each result set
           dense_max = max([r['score'] for r in dense_results]) if dense_results else 1.0
           sparse_max = max([r['score'] for r in sparse_results]) if sparse_results else 1.0
           
           # Create ID to result mapping
           id_to_result = {}
           
           # Process dense results
           for result in dense_results:
               doc_id = result['id']
               normalized_score = result['score'] / dense_max
               weighted_score = normalized_score * self.config.dense_weight
               
               id_to_result[doc_id] = {
                   'id': doc_id,
                   'content': result.get('content', ''),
                   'score': weighted_score,
                   'metadata': result.get('metadata', {}),
                   'source': 'dense'
               }
           
           # Process sparse results
           for result in sparse_results:
               doc_id = result['id']
               normalized_score = result['score'] / sparse_max
               weighted_score = normalized_score * self.config.sparse_weight
               
               if doc_id in id_to_result:
                   # Document exists in both result sets, combine scores
                   id_to_result[doc_id]['score'] += weighted_score
                   id_to_result[doc_id]['source'] = 'hybrid'
               else:
                   # New document from sparse results
                   id_to_result[doc_id] = {
                       'id': doc_id,
                       'content': result.get('content', ''),
                       'score': weighted_score,
                       'metadata': result.get('metadata', {}),
                       'source': 'sparse'
                   }
           
           # Convert to list and sort by score
           fused_results = list(id_to_result.values())
           fused_results.sort(key=lambda x: x['score'], reverse=True)
           
           return fused_results
       
       def _reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
           """Fuse results using Reciprocal Rank Fusion (RRF)."""
           # Create ID to result and rank mapping
           id_to_data = {}
           
           # Process dense results
           for rank, result in enumerate(dense_results):
               doc_id = result['id']
               if doc_id not in id_to_data:
                   id_to_data[doc_id] = {
                       'result': result,
                       'dense_rank': rank + 1,
                       'sparse_rank': None
                   }
           
           # Process sparse results
           for rank, result in enumerate(sparse_results):
               doc_id = result['id']
               if doc_id in id_to_data:
                   id_to_data[doc_id]['sparse_rank'] = rank + 1
               else:
                   id_to_data[doc_id] = {
                       'result': result,
                       'dense_rank': None,
                       'sparse_rank': rank + 1
                   }
           
           # Calculate RRF scores
           fused_results = []
           for doc_id, data in id_to_data.items():
               rrf_score = 0.0
               
               if data['dense_rank'] is not None:
                   rrf_score += 1.0 / (k + data['dense_rank'])
               
               if data['sparse_rank'] is not None:
                   rrf_score += 1.0 / (k + data['sparse_rank'])
               
               result = data['result'].copy()
               result['score'] = rrf_score
               result['source'] = 'hybrid_rrf'
               fused_results.append(result)
           
           # Sort by RRF score
           fused_results.sort(key=lambda x: x['score'], reverse=True)
           
           return fused_results
   ```

2. **Query Analysis for Retrieval Method Selection**
   ```python
   # src/morag_retrieval/hybrid/query_analyzer.py
   class QueryAnalyzer:
       def __init__(self):
           self.keyword_patterns = [
               r'\b\w+\s+\w+\b',  # Exact phrase patterns
               r'\b\w{6,}\b',    # Long words (likely specific terms)
               r'\b[A-Z][a-z]*\b'  # Capitalized words (likely proper nouns)
           ]
       
       def analyze_query(self, query: str) -> Dict[str, float]:
           """Analyze query to determine best retrieval methods."""
           query_lower = query.lower()
           scores = {
               'dense': 0.7,  # Default weight for dense retrieval
               'sparse': 0.3,  # Default weight for sparse retrieval
               'graph': 0.0    # Default weight for graph retrieval
           }
           
           # Check for keyword search indicators
           keyword_score = 0.0
           for pattern in self.keyword_patterns:
               matches = re.findall(pattern, query)
               if matches:
                   keyword_score += 0.1 * len(matches)
           
           # Adjust for query length (shorter queries favor sparse retrieval)
           query_words = query.split()
           if len(query_words) <= 3:
               keyword_score += 0.2
           
           # Check for exact quotes
           if '"' in query or "'" in query:
               keyword_score += 0.3
           
           # Adjust weights based on analysis
           if keyword_score > 0:
               sparse_boost = min(keyword_score, 0.6)  # Cap at 0.6
               scores['sparse'] += sparse_boost
               scores['dense'] -= sparse_boost / 2  # Reduce dense weight but not below 0.4
               scores['dense'] = max(scores['dense'], 0.4)
           
           # Normalize weights
           total = sum(scores.values())
           for method in scores:
               scores[method] /= total
           
           return scores
   ```

#### Deliverables
- Hybrid vector retriever with multiple fusion methods
- Query analysis for adaptive retrieval method selection
- Performance monitoring and optimization
- Comprehensive error handling and fallback mechanisms

## Testing Requirements

### Unit Tests
```python
# tests/test_sparse_vector.py
import pytest
from morag_retrieval.sparse import BM25Retriever, SparseVectorStorage
from morag_retrieval.hybrid import HybridVectorRetriever

class TestBM25Retriever:
    def test_bm25_indexing(self):
        retriever = BM25Retriever()
        test_docs = [
            {'id': '1', 'content': 'This is a test document about machine learning.'},
            {'id': '2', 'content': 'Another document discussing artificial intelligence.'},
            {'id': '3', 'content': 'Document about natural language processing and machine learning.'}
        ]
        
        retriever.add_documents(test_docs)
        retriever.build_index()
        
        assert retriever.doc_count == 3
        assert len(retriever.idf) > 0
        assert retriever.is_indexed
    
    def test_bm25_search(self):
        retriever = BM25Retriever()
        test_docs = [
            {'id': '1', 'content': 'This is a test document about machine learning.'},
            {'id': '2', 'content': 'Another document discussing artificial intelligence.'},
            {'id': '3', 'content': 'Document about natural language processing and machine learning.'}
        ]
        
        retriever.add_documents(test_docs)
        retriever.build_index()
        
        results = retriever.search("machine learning")
        
        assert len(results) > 0
        assert results[0]['id'] in ['1', '3']  # Documents containing "machine learning"
        assert all(r['score'] > 0 for r in results)

class TestHybridVectorRetriever:
    @pytest.mark.asyncio
    async def test_hybrid_search(self, mock_dense_retriever, mock_sparse_retriever):
        hybrid_retriever = HybridVectorRetriever(
            dense_retriever=mock_dense_retriever,
            sparse_retriever=mock_sparse_retriever
        )
        
        results = await hybrid_retriever.search("test query")
        
        assert len(results) > 0
        assert all('score' in r for r in results)
        assert all('source' in r for r in results)
        
        # Check that results are sorted by score
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)
```

### Integration Tests
```python
# tests/integration/test_sparse_integration.py
class TestSparseIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_sparse_retrieval(self, test_corpus):
        """Test end-to-end sparse retrieval pipeline."""
        # Set up BM25 retriever
        bm25_retriever = BM25Retriever()
        bm25_retriever.add_documents(test_corpus)
        bm25_retriever.build_index()
        
        # Set up dense retriever
        dense_retriever = VectorRetriever("test_index")
        await dense_retriever.add_documents(test_corpus)
        
        # Set up hybrid retriever
        hybrid_retriever = HybridVectorRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=bm25_retriever
        )
        
        # Test queries
        test_queries = [
            "machine learning algorithms",
            "natural language processing",
            "neural networks and deep learning"
        ]
        
        for query in test_queries:
            # Get results from each retriever
            dense_results = await dense_retriever.search(query, limit=10)
            sparse_results = bm25_retriever.search(query, top_k=10)
            hybrid_results = await hybrid_retriever.search(query, top_k=10)
            
            # Verify results
            assert len(dense_results) > 0
            assert len(sparse_results) > 0
            assert len(hybrid_results) > 0
            
            # Check that hybrid retrieval has good coverage
            dense_ids = set(r['id'] for r in dense_results)
            sparse_ids = set(r['id'] for r in sparse_results)
            hybrid_ids = set(r['id'] for r in hybrid_results)
            
            # Hybrid should contain some results from both methods
            assert len(hybrid_ids.intersection(dense_ids)) > 0
            assert len(hybrid_ids.intersection(sparse_ids)) > 0
```

## Success Criteria

- [ ] BM25 implementation correctly indexes and retrieves documents
- [ ] Sparse vector storage efficiently handles large document collections
- [ ] Hybrid retrieval outperforms both dense-only and sparse-only approaches
- [ ] Query analysis correctly identifies when to favor sparse retrieval
- [ ] Fusion methods effectively combine results from different retrievers
- [ ] System handles edge cases gracefully
- [ ] Performance targets met (< 2 seconds for hybrid retrieval)
- [ ] Unit test coverage > 90%
- [ ] Integration tests pass

## Performance Targets

- **BM25 Indexing**: < 5 seconds for 10,000 documents
- **BM25 Search**: < 200ms for typical queries
- **Hybrid Search**: < 2 seconds end-to-end
- **Memory Usage**: < 1GB for 100,000 documents

## Next Steps

After completing this task:
1. Proceed to **Task 3.3**: Enhanced Query Endpoints
2. Integrate sparse vector capabilities with the hybrid retrieval system
3. Implement caching for frequently used queries

## Dependencies

**Requires**:
- Task 3.1: Hybrid Retrieval System

**Enables**:
- Task 3.3: Enhanced Query Endpoints
- Task 4.2: Caching Strategy