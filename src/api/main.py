"""FastAPI application for FraudGuard fraud detection API.

Production-ready REST API for real-time fraud prediction with:
- Single and batch prediction endpoints
- GNNExplainer integration for regulatory compliance
- Redis caching for low-latency responses
- Health checks and model info endpoints

Run with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Example:
    POST /predict
    {
        "transaction_id": "TX12345",
        "features": {"TransactionAmt": 150.0, "card4": "visa"}
    }
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.model_loader import get_model_loader
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    FeatureAttribution,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global state
_redis_client = None
_mlflow_tracking = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup (model loading) and shutdown (cleanup).
    """
    global _redis_client, _mlflow_tracking

    logger.info("[STARTUP] Starting FraudGuard API...")

    # Load model on startup
    try:
        loader = get_model_loader()
        _ = loader.get_model()  # Trigger lazy load
        logger.info("[OK] Model loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")

    # Initialize Redis connection
    try:
        import redis

        _redis_client = redis.Redis(
            host="redis",
            port=6379,
            decode_responses=True,
            socket_timeout=5,
        )
        _redis_client.ping()
        logger.info("[OK] Redis connected")
    except Exception as e:
        logger.warning(f"[WARN] Redis not available: {e}")
        _redis_client = None

    # Check MLflow tracking
    try:
        import mlflow

        if mlflow.get_tracking_uri():
            _mlflow_tracking = True
            logger.info("[OK] MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"[WARN] MLflow not available: {e}")

    yield

    # Cleanup on shutdown
    logger.info("[SHUTDOWN] Shutting down FraudGuard API...")
    if _redis_client:
        _redis_client.close()


# Create FastAPI app
app = FastAPI(
    title="FraudGuard API",
    description="""
    Real-time fraud detection API using Graph Neural Networks.
    
    ## Features
    - **Single prediction**: Predict fraud for one transaction
    - **Batch prediction**: Process up to 100 transactions at once
    - **Explainability**: GNNExplainer feature attributions for regulatory compliance
    - **Caching**: Redis-based caching for repeated predictions
    
    ## Model
    AD-RL-GNN framework achieving 21% G-Means improvement over baseline.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _preprocess_features(features: dict) -> torch.Tensor:
    """Convert request features to model input tensor."""
    loader = get_model_loader()
    preprocessor = loader.get_preprocessor()

    # Convert to DataFrame for preprocessor
    df = pd.DataFrame([features])

    # Apply preprocessing
    try:
        X = preprocessor.transform(df)
    except Exception as e:
        # If preprocessor not fitted, use simple normalization
        logger.warning(f"Preprocessor transform failed: {e}, using raw features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(0).values.astype(np.float32)

    return torch.tensor(X, dtype=torch.float32).to(loader.device)


def _get_confidence_level(prob: float) -> str:
    """Convert probability to confidence level."""
    if prob > 0.9 or prob < 0.1:
        return "high"
    elif prob > 0.7 or prob < 0.3:
        return "medium"
    return "low"


def _generate_explanation(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_idx: int,
) -> Optional[ExplanationResponse]:
    """Generate GNNExplainer explanation for a prediction."""
    try:
        from torch_geometric.explain import Explainer, GNNExplainer

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
        )

        explanation = explainer(x, edge_index, index=node_idx)

        # Extract feature importances
        node_mask = explanation.node_mask
        if node_mask is not None:
            importance = node_mask[node_idx].cpu().numpy()
            top_indices = np.argsort(importance)[-5:][::-1]  # Top 5

            top_features = [
                FeatureAttribution(
                    feature_name=f"feature_{i}",
                    importance=float(importance[i]),
                    direction="positive" if importance[i] > 0 else "negative",
                )
                for i in top_indices
            ]
        else:
            top_features = []

        # Extract edge importance
        edge_mask = explanation.edge_mask
        edge_importance = {}
        if edge_mask is not None:
            for i, imp in enumerate(edge_mask[:10].cpu().numpy()):  # Top 10 edges
                edge_importance[f"edge_{i}"] = float(imp)

        return ExplanationResponse(
            node_id=node_idx,
            top_features=top_features,
            subgraph_nodes=list(range(min(10, x.shape[0]))),  # Placeholder
            edge_importance=edge_importance,
        )

    except Exception as e:
        import traceback
        logger.error(f"GNNExplainer failed: {e}\n{traceback.format_exc()}")
        return None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "FraudGuard API",
        "version": "1.0.0",
        "description": "Real-time GNN fraud detection",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    loader = get_model_loader()

    return HealthResponse(
        status="healthy" if loader.is_loaded() else "unhealthy",
        model_loaded=loader.is_loaded(),
        redis_connected=_redis_client is not None,
        mlflow_tracking=_mlflow_tracking,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata and performance metrics."""
    loader = get_model_loader()
    info = loader.get_model_info()

    return ModelInfoResponse(
        model_name=info["model_name"],
        version=info["version"],
        gmeans=info["gmeans"],
        training_date=info["training_date"],
        features_count=32,  # After PCA
        graph_nodes=590540,  # Full dataset
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: TransactionRequest,
    include_explanation: bool = False,
):
    """Predict fraud for a single transaction.

    Args:
        request: Transaction data with features
        include_explanation: Include GNNExplainer attributions

    Returns:
        Fraud prediction with probability and optional explanation
    """
    start_time = time.perf_counter()

    try:
        loader = get_model_loader()
        model = loader.get_model()

        # Check cache first
        cache_key = f"pred:{request.transaction_id}"
        if _redis_client:
            cached = _redis_client.get(cache_key)
            if cached:
                import json

                cached_response = json.loads(cached)
                cached_response["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
                return PredictionResponse(**cached_response)

        # Preprocess features
        features_dict = request.features.model_dump()
        x_query = _preprocess_features(features_dict)

        # ===== RAG Pipeline: Retrieve neighbors and construct graph =====
        faiss_index = loader.get_faiss_index()
        feature_store = loader.get_feature_store()

        if faiss_index is not None and feature_store is not None:
            # k-NN search: find 50 nearest neighbors from training set
            k = min(50, faiss_index.ntotal)
            D, I = faiss_index.search(x_query.cpu().numpy(), k)
            neighbor_indices = I[0]

            # Retrieve neighbor features from memory-mapped store
            neighbor_feats = torch.tensor(
                feature_store[neighbor_indices], dtype=torch.float32
            ).to(loader.device)

            # Construct star graph: neighbors (1..k) -> query node (0)
            edge_index = torch.zeros((2, k), dtype=torch.long, device=loader.device)
            edge_index[0] = torch.arange(1, k + 1)  # Sources: neighbors
            edge_index[1] = 0  # Target: query node

            # Combine features: [Query (0), Neighbor 1, ..., Neighbor k]
            x = torch.cat([x_query, neighbor_feats], dim=0)

            logger.debug(
                f"RAG: Retrieved {k} neighbors, graph shape: x={x.shape}, edges={edge_index.shape}"
            )
        else:
            # Fallback: single-node mode if RAG artifacts not available
            logger.warning("RAG artifacts not available, using single-node mode")
            x = x_query
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=loader.device)

        # Run inference
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            probs = torch.softmax(logits, dim=1)
            fraud_prob = probs[0, 1].item()  # Prediction for query node (index 0)
            is_fraud = fraud_prob > 0.5

        # Generate explanation if requested
        explanation = None
        if include_explanation:
            explanation = _generate_explanation(model, x, edge_index, 0)

        processing_time = (time.perf_counter() - start_time) * 1000

        response = PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=fraud_prob,
            confidence=_get_confidence_level(fraud_prob),
            explanation=explanation,
            processing_time_ms=processing_time,
            model_version=loader.model_version,
        )

        # Cache result
        if _redis_client:
            import json

            _redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(response.model_dump(exclude={"explanation"})),
            )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict fraud for multiple transactions.

    Args:
        request: List of transactions (max 100)

    Returns:
        Batch predictions with aggregate statistics
    """
    if len(request.transactions) > 100:
        raise HTTPException(status_code=400, detail="Maximum batch size is 100 transactions")

    predictions = []
    total_time = 0
    fraud_count = 0

    for tx in request.transactions:
        response = await predict(tx, request.include_explanations)
        predictions.append(response)
        total_time += response.processing_time_ms
        if response.is_fraud:
            fraud_count += 1

    return BatchPredictionResponse(
        predictions=predictions,
        total_transactions=len(predictions),
        fraud_count=fraud_count,
        avg_processing_time_ms=total_time / len(predictions) if predictions else 0,
    )


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Hot reload the model (admin endpoint).

    Useful for updating to new model version without restart.
    """

    def _reload():
        loader = get_model_loader()
        loader.reload()

    background_tasks.add_task(_reload)
    return {"status": "reload_initiated", "message": "Model will be reloaded in background"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
