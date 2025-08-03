"""Database connection and session management for Graph Hypernetwork Forge.

Provides database connectivity for storing graphs, experiments, and results
with support for SQLite, PostgreSQL, and in-memory databases.
"""

import os
import sqlite3
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging
from contextlib import contextmanager
import json
import pickle
from dataclasses import asdict

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration for database connections."""
    
    def __init__(
        self,
        db_type: str = "sqlite",
        db_path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs
    ):
        self.db_type = db_type.lower()
        self.db_path = db_path or "graph_hypernetwork_forge.db"
        self.host = host or "localhost"
        self.port = port or (5432 if db_type == "postgresql" else 3306)
        self.database = database
        self.username = username
        self.password = password
        self.extras = kwargs


class DatabaseManager:
    """Database manager for Graph Hypernetwork Forge."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
        self._setup_database()
        
    def _setup_database(self):
        """Setup database based on configuration."""
        if self.config.db_type == "sqlite":
            self._setup_sqlite()
        elif self.config.db_type == "postgresql":
            self._setup_postgresql()
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
            
    def _setup_sqlite(self):
        """Setup SQLite database."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30.0
        )
        
        # Enable foreign keys and other pragmas
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.commit()
        
        logger.info(f"Connected to SQLite database: {db_path}")
        
    def _setup_postgresql(self):
        """Setup PostgreSQL database."""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL support")
            
        connection_params = {
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "user": self.config.username,
            "password": self.config.password,
            "cursor_factory": RealDictCursor,
        }
        
        self.connection = psycopg2.connect(**connection_params)
        self.connection.autocommit = False
        
        logger.info(f"Connected to PostgreSQL database: {self.config.database}")
    
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic cleanup."""
        cursor = self.connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None,
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL query."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            
            if fetch:
                if self.config.db_type == "sqlite":
                    # Convert sqlite3.Row to dict
                    cursor.row_factory = sqlite3.Row
                    results = [dict(row) for row in cursor.fetchall()]
                else:
                    results = cursor.fetchall()
                return results
            else:
                self.connection.commit()
                return None
    
    def create_tables(self):
        """Create database tables for Graph Hypernetwork Forge."""
        
        # Knowledge graphs table
        create_graphs_table = """
        CREATE TABLE IF NOT EXISTS knowledge_graphs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            num_nodes INTEGER,
            num_edges INTEGER,
            metadata TEXT,
            graph_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Experiments table
        create_experiments_table = """
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            config TEXT,
            status TEXT DEFAULT 'pending',
            metrics TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Models table
        create_models_table = """
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            model_type TEXT,
            config TEXT,
            file_path TEXT,
            performance_metrics TEXT,
            experiment_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        """
        
        # Evaluation results table
        create_results_table = """
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            dataset_name TEXT,
            task_type TEXT,
            metrics TEXT,
            predictions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        """
        
        # Benchmarks table
        create_benchmarks_table = """
        CREATE TABLE IF NOT EXISTS benchmarks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            dataset_info TEXT,
            baseline_results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        tables = [
            create_graphs_table,
            create_experiments_table,
            create_models_table,
            create_results_table,
            create_benchmarks_table,
        ]
        
        for table_sql in tables:
            self.execute_query(table_sql)
        
        logger.info("Database tables created successfully")
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


class GraphRepository:
    """Repository for storing and retrieving knowledge graphs."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def save_graph(self, graph, graph_id: Optional[str] = None) -> str:
        """Save knowledge graph to database."""
        from ..data.knowledge_graph import TextualKnowledgeGraph
        
        if not isinstance(graph, TextualKnowledgeGraph):
            raise ValueError("Graph must be a TextualKnowledgeGraph instance")
        
        graph_id = graph_id or graph.name.replace(" ", "_").lower()
        
        # Serialize graph data
        graph_data = {
            "nodes": [asdict(node) for node in graph.nodes.values()],
            "edges": [asdict(edge) for edge in graph.edges],
        }
        
        # Convert to JSON
        graph_json = json.dumps(graph_data, default=str)
        metadata_json = json.dumps(graph.metadata)
        
        query = """
        INSERT OR REPLACE INTO knowledge_graphs 
        (id, name, description, num_nodes, num_edges, metadata, graph_data) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            graph_id,
            graph.name,
            graph.metadata.get("description", ""),
            len(graph.nodes),
            len(graph.edges),
            metadata_json,
            graph_json,
        )
        
        self.db.execute_query(query, params)
        logger.info(f"Saved graph '{graph.name}' with ID: {graph_id}")
        
        return graph_id
    
    def load_graph(self, graph_id: str):
        """Load knowledge graph from database."""
        from ..data.knowledge_graph import TextualKnowledgeGraph, NodeInfo, EdgeInfo
        
        query = "SELECT * FROM knowledge_graphs WHERE id = ?"
        results = self.db.execute_query(query, (graph_id,), fetch=True)
        
        if not results:
            raise ValueError(f"Graph with ID '{graph_id}' not found")
        
        row = results[0]
        graph_data = json.loads(row["graph_data"])
        metadata = json.loads(row["metadata"])
        
        # Reconstruct nodes and edges
        nodes = [NodeInfo(**node_data) for node_data in graph_data["nodes"]]
        edges = [EdgeInfo(**edge_data) for edge_data in graph_data["edges"]]
        
        graph = TextualKnowledgeGraph(
            nodes=nodes,
            edges=edges,
            name=row["name"],
            metadata=metadata,
        )
        
        logger.info(f"Loaded graph '{graph.name}' with {len(nodes)} nodes and {len(edges)} edges")
        return graph
    
    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all stored graphs."""
        query = """
        SELECT id, name, description, num_nodes, num_edges, created_at 
        FROM knowledge_graphs 
        ORDER BY created_at DESC
        """
        
        return self.db.execute_query(query, fetch=True)
    
    def delete_graph(self, graph_id: str):
        """Delete graph from database."""
        query = "DELETE FROM knowledge_graphs WHERE id = ?"
        self.db.execute_query(query, (graph_id,))
        logger.info(f"Deleted graph with ID: {graph_id}")


class ExperimentRepository:
    """Repository for storing experiment results and configurations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        config: Dict[str, Any],
        description: str = "",
    ) -> str:
        """Save experiment configuration."""
        config_json = json.dumps(config, default=str)
        
        query = """
        INSERT OR REPLACE INTO experiments 
        (id, name, description, config, status, start_time) 
        VALUES (?, ?, ?, ?, 'running', CURRENT_TIMESTAMP)
        """
        
        params = (experiment_id, name, description, config_json)
        self.db.execute_query(query, params)
        
        logger.info(f"Saved experiment: {name}")
        return experiment_id
    
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Update experiment status and metrics."""
        metrics_json = json.dumps(metrics) if metrics else None
        
        query = """
        UPDATE experiments 
        SET status = ?, metrics = ?, 
            end_time = CASE WHEN ? IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE end_time END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """
        
        params = (status, metrics_json, status, experiment_id)
        self.db.execute_query(query, params)
        
        logger.info(f"Updated experiment {experiment_id} status to: {status}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        query = "SELECT * FROM experiments WHERE id = ?"
        results = self.db.execute_query(query, (experiment_id,), fetch=True)
        
        if results:
            experiment = results[0]
            experiment["config"] = json.loads(experiment["config"])
            if experiment["metrics"]:
                experiment["metrics"] = json.loads(experiment["metrics"])
            return experiment
        
        return None
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List experiments, optionally filtered by status."""
        if status:
            query = "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC"
            params = (status,)
        else:
            query = "SELECT * FROM experiments ORDER BY created_at DESC"
            params = ()
        
        results = self.db.execute_query(query, params, fetch=True)
        
        # Parse JSON fields
        for experiment in results:
            experiment["config"] = json.loads(experiment["config"])
            if experiment["metrics"]:
                experiment["metrics"] = json.loads(experiment["metrics"])
        
        return results


class CacheManager:
    """Cache manager for frequently accessed data."""
    
    def __init__(self, cache_type: str = "memory", redis_url: Optional[str] = None):
        self.cache_type = cache_type.lower()
        
        if self.cache_type == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("redis is required for Redis caching")
            self.redis_client = redis.from_url(redis_url or "redis://localhost:6379/0")
            logger.info("Connected to Redis cache")
        else:
            self.memory_cache = {}
            logger.info("Using in-memory cache")
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if self.cache_type == "redis":
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        else:
            return self.memory_cache.get(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if self.cache_type == "redis":
            serialized = pickle.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, serialized)
            else:
                self.redis_client.set(key, serialized)
        else:
            self.memory_cache[key] = value
    
    def delete(self, key: str):
        """Delete value from cache."""
        if self.cache_type == "redis":
            self.redis_client.delete(key)
        else:
            self.memory_cache.pop(key, None)
    
    def clear(self):
        """Clear all cached values."""
        if self.cache_type == "redis":
            self.redis_client.flushdb()
        else:
            self.memory_cache.clear()


def get_database_manager(config_path: Optional[str] = None) -> DatabaseManager:
    """Get database manager instance from configuration."""
    if config_path and Path(config_path).exists():
        import tomli
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)
        db_config = DatabaseConfig(**config_data.get("database", {}))
    else:
        # Default configuration
        db_config = DatabaseConfig(
            db_type="sqlite",
            db_path=os.getenv("DATABASE_PATH", "graph_hypernetwork_forge.db")
        )
    
    return DatabaseManager(db_config)