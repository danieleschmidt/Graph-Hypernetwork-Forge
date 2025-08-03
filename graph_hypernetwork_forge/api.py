"""Simple API for Graph Hypernetwork Forge.

Provides REST API endpoints for model inference and knowledge graph operations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union
import json
import logging
from pathlib import Path
import asyncio
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from .models import HyperGNN
from .data import TextualKnowledgeGraph, NodeInfo, EdgeInfo, create_synthetic_kg
from .utils.evaluation import BenchmarkEvaluator

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class NodeRequest(BaseModel):
    """Request model for node information."""
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class EdgeRequest(BaseModel):
    """Request model for edge information."""
    source: str
    target: str
    relation: str
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeGraphRequest(BaseModel):
    """Request model for knowledge graph creation."""
    name: str
    nodes: List[NodeRequest]
    edges: List[EdgeRequest]
    metadata: Optional[Dict[str, Any]] = None


class InferenceRequest(BaseModel):
    """Request model for inference."""
    edge_index: List[List[int]]
    node_texts: List[str]
    node_features: Optional[List[List[float]]] = None
    task_type: str = Field(default="link_prediction", description="Type of task: link_prediction, node_classification")


class ModelLoadRequest(BaseModel):
    """Request model for loading a pre-trained model."""
    model_path: str
    device: Optional[str] = "auto"


class EvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    dataset_name: str
    task_type: str = "link_prediction"
    return_predictions: bool = False


class InferenceResponse(BaseModel):
    """Response model for inference results."""
    predictions: List[float]
    inference_time: float
    model_info: Dict[str, Any]


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    dataset: str
    task_type: str
    metrics: Dict[str, float]
    num_samples: int
    evaluation_time: float


class APIStatus(BaseModel):
    """API status response."""
    status: str
    version: str
    model_loaded: bool
    device: str
    uptime: float


class HyperGNNAPI:
    """REST API for Graph Hypernetwork Forge."""
    
    def __init__(self, title: str = "Graph Hypernetwork Forge API", version: str = "0.1.0"):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API functionality")
            
        self.app = FastAPI(title=title, version=version)
        self.model: Optional[HyperGNN] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator: Optional[BenchmarkEvaluator] = None
        self.knowledge_graphs: Dict[str, TextualKnowledgeGraph] = {}
        self.start_time = datetime.now()
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info(f"HyperGNN API initialized on device: {self.device}")
    
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/", response_model=APIStatus)
        async def get_status():
            """Get API status information."""
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return APIStatus(
                status="running",
                version="0.1.0",
                model_loaded=self.model is not None,
                device=str(self.device),
                uptime=uptime
            )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}
        
        @self.app.post("/model/load")
        async def load_model(request: ModelLoadRequest):
            """Load a pre-trained HyperGNN model."""
            try:
                # Determine device
                if request.device == "auto":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device(request.device)
                
                # Load model
                self.model = HyperGNN.load_pretrained(request.model_path, device=device)
                self.device = device
                
                # Initialize evaluator
                self.evaluator = BenchmarkEvaluator(self.model, device)
                
                logger.info(f"Model loaded from {request.model_path} on {device}")
                
                return {
                    "status": "success",
                    "message": f"Model loaded successfully on {device}",
                    "model_path": request.model_path,
                    "device": str(device)
                }
                
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        @self.app.post("/model/create")
        async def create_model(
            text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
            gnn_backbone: str = "GAT",
            hidden_dim: int = 256,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1
        ):
            """Create a new HyperGNN model."""
            try:
                self.model = HyperGNN(
                    text_encoder=text_encoder,
                    gnn_backbone=gnn_backbone,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                
                self.model = self.model.to(self.device)
                self.evaluator = BenchmarkEvaluator(self.model, self.device)
                
                logger.info("New model created successfully")
                
                return {
                    "status": "success",
                    "message": "Model created successfully",
                    "config": {
                        "text_encoder": text_encoder,
                        "gnn_backbone": gnn_backbone,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "dropout": dropout,
                    },
                    "device": str(self.device)
                }
                
            except Exception as e:
                logger.error(f"Error creating model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")
        
        @self.app.post("/inference", response_model=InferenceResponse)
        async def run_inference(request: InferenceRequest):
            """Run inference on the loaded model."""
            if self.model is None:
                raise HTTPException(status_code=400, detail="No model loaded")
            
            try:
                import time
                start_time = time.time()
                
                # Prepare inputs
                edge_index = torch.tensor(request.edge_index, dtype=torch.long).to(self.device)
                node_texts = request.node_texts
                
                if request.node_features:
                    node_features = torch.tensor(request.node_features, dtype=torch.float32).to(self.device)
                else:
                    # Generate default features
                    node_features = torch.randn(len(node_texts), self.model.hidden_dim).to(self.device)
                
                # Run inference
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model.zero_shot_inference(
                        edge_index=edge_index,
                        node_features=node_features,
                        node_texts=node_texts
                    )
                
                inference_time = time.time() - start_time
                
                # Convert predictions to list
                if request.task_type == "link_prediction":
                    pred_list = torch.sigmoid(predictions).cpu().numpy().tolist()
                else:
                    pred_list = F.softmax(predictions, dim=-1).cpu().numpy().tolist()
                
                return InferenceResponse(
                    predictions=pred_list,
                    inference_time=inference_time,
                    model_info={
                        "gnn_backbone": self.model.gnn_backbone,
                        "hidden_dim": self.model.hidden_dim,
                        "num_layers": self.model.num_layers,
                        "device": str(self.device)
                    }
                )
                
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
        @self.app.post("/kg/create")
        async def create_knowledge_graph(request: KnowledgeGraphRequest):
            """Create a knowledge graph."""
            try:
                # Convert request to internal format
                nodes = [
                    NodeInfo(id=node.id, text=node.text, metadata=node.metadata)
                    for node in request.nodes
                ]
                
                edges = [
                    EdgeInfo(source=edge.source, target=edge.target, 
                            relation=edge.relation, metadata=edge.metadata)
                    for edge in request.edges
                ]
                
                # Create knowledge graph
                kg = TextualKnowledgeGraph(
                    nodes=nodes,
                    edges=edges,
                    name=request.name,
                    metadata=request.metadata
                )
                
                # Store in memory
                self.knowledge_graphs[request.name] = kg
                
                # Get statistics
                stats = kg.get_statistics()
                
                logger.info(f"Knowledge graph '{request.name}' created with {len(nodes)} nodes and {len(edges)} edges")
                
                return {
                    "status": "success",
                    "message": f"Knowledge graph '{request.name}' created successfully",
                    "name": request.name,
                    "statistics": stats
                }
                
            except Exception as e:
                logger.error(f"Error creating knowledge graph: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create knowledge graph: {str(e)}")
        
        @self.app.get("/kg/list")
        async def list_knowledge_graphs():
            """List all stored knowledge graphs."""
            graph_info = []
            
            for name, kg in self.knowledge_graphs.items():
                stats = kg.get_statistics()
                graph_info.append({
                    "name": name,
                    "num_nodes": stats["num_nodes"],
                    "num_edges": stats["num_edges"],
                    "relations": stats["relations"]
                })
            
            return {
                "knowledge_graphs": graph_info,
                "total": len(graph_info)
            }
        
        @self.app.get("/kg/{graph_name}")
        async def get_knowledge_graph(graph_name: str):
            """Get knowledge graph by name."""
            if graph_name not in self.knowledge_graphs:
                raise HTTPException(status_code=404, detail=f"Knowledge graph '{graph_name}' not found")
            
            kg = self.knowledge_graphs[graph_name]
            stats = kg.get_statistics()
            
            return {
                "name": kg.name,
                "metadata": kg.metadata,
                "statistics": stats,
                "node_texts": kg.node_texts[:10],  # First 10 nodes as preview
                "total_nodes": len(kg.nodes),
                "total_edges": len(kg.edges)
            }
        
        @self.app.post("/kg/synthetic")
        async def create_synthetic_kg(
            name: str,
            num_nodes: int = 100,
            num_edges: int = 200,
            relations: Optional[List[str]] = None,
            random_seed: int = 42
        ):
            """Create a synthetic knowledge graph."""
            try:
                if relations is None:
                    relations = ["related_to", "part_of", "instance_of", "similar_to"]
                
                kg = create_synthetic_kg(
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    relations=relations,
                    random_seed=random_seed
                )
                
                kg.name = name
                self.knowledge_graphs[name] = kg
                
                stats = kg.get_statistics()
                
                return {
                    "status": "success",
                    "message": f"Synthetic knowledge graph '{name}' created",
                    "name": name,
                    "statistics": stats
                }
                
            except Exception as e:
                logger.error(f"Error creating synthetic KG: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to create synthetic KG: {str(e)}")
        
        @self.app.post("/evaluate", response_model=EvaluationResponse)
        async def evaluate_model(request: EvaluationRequest):
            """Evaluate model on a knowledge graph."""
            if self.model is None:
                raise HTTPException(status_code=400, detail="No model loaded")
            
            if self.evaluator is None:
                raise HTTPException(status_code=400, detail="No evaluator available")
            
            if request.dataset_name not in self.knowledge_graphs:
                raise HTTPException(status_code=404, detail=f"Knowledge graph '{request.dataset_name}' not found")
            
            try:
                import time
                from ..data import LinkPredictionDataset, create_dataloader
                
                start_time = time.time()
                
                # Get knowledge graph
                kg = self.knowledge_graphs[request.dataset_name]
                
                # Create dataset and dataloader
                if request.task_type == "link_prediction":
                    dataset = LinkPredictionDataset(kg, mode="test")
                else:
                    raise HTTPException(status_code=400, detail=f"Task type '{request.task_type}' not supported yet")
                
                dataloader = create_dataloader(dataset, batch_size=32, shuffle=False)
                
                # Run evaluation
                result = self.evaluator.evaluate_dataset(
                    dataloader=dataloader,
                    dataset_name=request.dataset_name,
                    task_type=request.task_type,
                    return_predictions=request.return_predictions
                )
                
                evaluation_time = time.time() - start_time
                
                return EvaluationResponse(
                    dataset=result["dataset"],
                    task_type=result["task_type"],
                    metrics=result["metrics"],
                    num_samples=result["num_samples"],
                    evaluation_time=evaluation_time
                )
                
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
        
        @self.app.post("/kg/{graph_name}/inference")
        async def inference_on_kg(graph_name: str, task_type: str = "link_prediction"):
            """Run inference on a specific knowledge graph."""
            if self.model is None:
                raise HTTPException(status_code=400, detail="No model loaded")
            
            if graph_name not in self.knowledge_graphs:
                raise HTTPException(status_code=404, detail=f"Knowledge graph '{graph_name}' not found")
            
            try:
                kg = self.knowledge_graphs[graph_name]
                
                # Prepare inference request
                edge_index = kg.edge_index.tolist()
                node_texts = kg.node_texts
                node_features = kg.node_features.tolist()
                
                inference_request = InferenceRequest(
                    edge_index=edge_index,
                    node_texts=node_texts,
                    node_features=node_features,
                    task_type=task_type
                )
                
                # Run inference
                return await run_inference(inference_request)
                
            except Exception as e:
                logger.error(f"Error during KG inference: {e}")
                raise HTTPException(status_code=500, detail=f"KG inference failed: {str(e)}")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    api = HyperGNNAPI()
    return api.app


# CLI entry point for running the API
def main():
    """Main entry point for API server."""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph Hypernetwork Forge API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()