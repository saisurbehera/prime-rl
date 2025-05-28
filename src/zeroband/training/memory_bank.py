"""Memory Bank for storing and retrieving oracle interactions."""

import os
import json
import pickle
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass, asdict
import faiss
from pathlib import Path


@dataclass
class MemoryEntry:
    """Single entry in the memory bank."""
    id: str
    timestamp: str
    prompt: str
    generated_code: str
    oracle_reports: Dict[str, Any]
    joint_loss: float
    gating_weights: Optional[List[float]] = None
    uncertainty_score: float = 0.5
    embeddings: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryBank:
    """
    Memory bank for storing oracle interactions with vector similarity search.
    
    Uses SQLite for structured data and FAISS for vector similarity search.
    """
    
    def __init__(self, 
                 storage_path: str = "memory_bank",
                 embedding_dim: int = 768,
                 max_entries: int = 100000):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.db_path = self.storage_path / "memory.db"
        self.index_path = self.storage_path / "faiss.index"
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        
        # Initialize database
        self._init_database()
        
        # Initialize FAISS index
        self._init_faiss_index()
        
        # Cache for recent entries
        self.cache = {}
        self.cache_size = 1000
        
    def _init_database(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                prompt TEXT,
                generated_code TEXT,
                oracle_reports TEXT,
                joint_loss REAL,
                gating_weights TEXT,
                uncertainty_score REAL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_joint_loss ON memories(joint_loss)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_uncertainty ON memories(uncertainty_score)
        """)
        
        self.conn.commit()
    
    def _init_faiss_index(self):
        """Initialize or load FAISS index for similarity search."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            # Get current count
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            self.entry_count = cursor.fetchone()[0]
        else:
            # Create new index with inner product similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.entry_count = 0
    
    def add_entry(self, entry: MemoryEntry):
        """Add a new entry to the memory bank."""
        # Check if we need to evict old entries
        if self.entry_count >= self.max_entries:
            self._evict_oldest_entries(batch_size=100)
        
        # Add to database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, timestamp, prompt, generated_code, oracle_reports, 
             joint_loss, gating_weights, uncertainty_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.timestamp,
            entry.prompt,
            entry.generated_code,
            json.dumps(entry.oracle_reports),
            entry.joint_loss,
            json.dumps(entry.gating_weights) if entry.gating_weights else None,
            entry.uncertainty_score,
            json.dumps(entry.metadata) if entry.metadata else None
        ))
        self.conn.commit()
        
        # Add embeddings to FAISS if provided
        if entry.embeddings and "code_embedding" in entry.embeddings:
            embedding = entry.embeddings["code_embedding"]
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            
            # Normalize for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            self.index.add(embedding.reshape(1, -1))
        
        # Update cache
        self.cache[entry.id] = entry
        if len(self.cache) > self.cache_size:
            # Remove oldest from cache
            oldest_id = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_id]
        
        self.entry_count += 1
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific entry by ID."""
        # Check cache first
        if entry_id in self.cache:
            return self.cache[entry_id]
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if row:
            entry = self._row_to_entry(row)
            self.cache[entry_id] = entry
            return entry
        
        return None
    
    def search_similar(self, 
                      query_embedding: np.ndarray,
                      k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """
        Search for similar entries using vector similarity.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Optional SQL filters (e.g., {"joint_loss": "<0.5"})
        """
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        
        # Normalize query
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k * 2)
        
        # Get entries from database
        entries = []
        cursor = self.conn.cursor()
        
        for idx in indices[0]:
            if idx < 0:  # Invalid index
                continue
                
            # Build query with filters
            query = "SELECT * FROM memories LIMIT 1 OFFSET ?"
            params = [int(idx)]
            
            if filters:
                filter_clauses = []
                for key, value in filters.items():
                    if isinstance(value, str) and value.startswith(("<", ">", "=")):
                        filter_clauses.append(f"{key} {value}")
                    else:
                        filter_clauses.append(f"{key} = ?")
                        params.append(value)
                
                if filter_clauses:
                    query = f"SELECT * FROM memories WHERE {' AND '.join(filter_clauses)} LIMIT 1 OFFSET ?"
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                entries.append(self._row_to_entry(row))
                if len(entries) >= k:
                    break
        
        return entries
    
    def get_failure_cases(self, 
                         limit: int = 100,
                         loss_threshold: float = 0.7) -> List[MemoryEntry]:
        """Get high-loss failure cases for curriculum learning."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE joint_loss > ? 
            ORDER BY joint_loss DESC 
            LIMIT ?
        """, (loss_threshold, limit))
        
        return [self._row_to_entry(row) for row in cursor.fetchall()]
    
    def get_high_uncertainty_cases(self, 
                                  limit: int = 100,
                                  uncertainty_threshold: float = 0.7) -> List[MemoryEntry]:
        """Get high-uncertainty cases."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM memories 
            WHERE uncertainty_score > ? 
            ORDER BY uncertainty_score DESC 
            LIMIT ?
        """, (uncertainty_threshold, limit))
        
        return [self._row_to_entry(row) for row in cursor.fetchall()]
    
    def get_oracle_disagreements(self, limit: int = 100) -> List[MemoryEntry]:
        """Get cases where oracles disagreed significantly."""
        # This requires analyzing the oracle reports
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories")
        
        disagreement_cases = []
        
        for row in cursor.fetchall():
            entry = self._row_to_entry(row)
            
            # Calculate disagreement score
            if entry.oracle_reports:
                scores = []
                for oracle_name, report in entry.oracle_reports.items():
                    if isinstance(report, dict) and "score" in report:
                        scores.append(report["score"])
                
                if len(scores) > 1:
                    # High variance indicates disagreement
                    variance = np.var(scores)
                    if variance > 0.1:  # Threshold for disagreement
                        entry.metadata = entry.metadata or {}
                        entry.metadata["oracle_disagreement"] = variance
                        disagreement_cases.append((variance, entry))
        
        # Sort by disagreement and return top cases
        disagreement_cases.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in disagreement_cases[:limit]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory bank."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM memories")
        stats["total_entries"] = cursor.fetchone()[0]
        
        # Average loss
        cursor.execute("SELECT AVG(joint_loss) FROM memories")
        stats["avg_joint_loss"] = cursor.fetchone()[0]
        
        # Loss distribution
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN joint_loss < 0.3 THEN 1 END) as low_loss,
                COUNT(CASE WHEN joint_loss >= 0.3 AND joint_loss < 0.7 THEN 1 END) as medium_loss,
                COUNT(CASE WHEN joint_loss >= 0.7 THEN 1 END) as high_loss
            FROM memories
        """)
        row = cursor.fetchone()
        stats["loss_distribution"] = {
            "low": row[0],
            "medium": row[1],
            "high": row[2]
        }
        
        # Uncertainty distribution
        cursor.execute("SELECT AVG(uncertainty_score), MIN(uncertainty_score), MAX(uncertainty_score) FROM memories")
        row = cursor.fetchone()
        stats["uncertainty"] = {
            "avg": row[0],
            "min": row[1],
            "max": row[2]
        }
        
        return stats
    
    def export_dataset(self, 
                      output_path: str,
                      filters: Optional[Dict[str, Any]] = None,
                      limit: Optional[int] = None):
        """Export memory bank to a dataset file."""
        cursor = self.conn.cursor()
        
        # Build query
        query = "SELECT * FROM memories"
        params = []
        
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append(f"{key} = ?")
                params.append(value)
            query += f" WHERE {' AND '.join(filter_clauses)}"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        
        # Export to JSON lines format
        with open(output_path, 'w') as f:
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                data = {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    "prompt": entry.prompt,
                    "generated_code": entry.generated_code,
                    "oracle_scores": {
                        name: report.get("score", 0) 
                        for name, report in entry.oracle_reports.items()
                        if isinstance(report, dict)
                    },
                    "joint_loss": entry.joint_loss,
                    "uncertainty_score": entry.uncertainty_score
                }
                f.write(json.dumps(data) + '\n')
    
    def _row_to_entry(self, row: Tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            timestamp=row[1],
            prompt=row[2],
            generated_code=row[3],
            oracle_reports=json.loads(row[4]) if row[4] else {},
            joint_loss=row[5],
            gating_weights=json.loads(row[6]) if row[6] else None,
            uncertainty_score=row[7],
            metadata=json.loads(row[8]) if row[8] else None
        )
    
    def _evict_oldest_entries(self, batch_size: int = 100):
        """Evict oldest entries when reaching capacity."""
        cursor = self.conn.cursor()
        
        # Get oldest entries
        cursor.execute("""
            SELECT id FROM memories 
            ORDER BY timestamp ASC 
            LIMIT ?
        """, (batch_size,))
        
        ids_to_delete = [row[0] for row in cursor.fetchall()]
        
        # Delete from database
        cursor.execute(
            f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids_to_delete))})",
            ids_to_delete
        )
        self.conn.commit()
        
        # Remove from cache
        for id_to_delete in ids_to_delete:
            if id_to_delete in self.cache:
                del self.cache[id_to_delete]
        
        # Note: FAISS index removal is more complex, would need to rebuild
        # For now, we accept some orphaned vectors
        
        self.entry_count -= len(ids_to_delete)
    
    def save(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, str(self.index_path))
    
    def close(self):
        """Close database connection and save index."""
        self.save()
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()