"""Pydantic schemas for FraudGuard API.

Defines request/response models for the fraud detection API,
ensuring type safety and automatic OpenAPI documentation.

Example:
    >>> from src.api.schemas import TransactionRequest
    >>> req = TransactionRequest(transaction_id="TX001", features={...})
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Transaction features for fraud prediction.
    
    These match the preprocessed features used in training.
    Missing values will be imputed with defaults.
    """
    TransactionAmt: float = Field(..., description="Transaction amount in USD")
    TransactionDT: Optional[int] = Field(None, description="Transaction timestamp (seconds)")
    
    # Card features
    card1: Optional[int] = Field(None, description="Card identifier 1")
    card2: Optional[float] = Field(None, description="Card identifier 2")
    card3: Optional[float] = Field(None, description="Card identifier 3")
    card4: Optional[str] = Field(None, description="Card type (credit/debit)")
    card5: Optional[float] = Field(None, description="Card identifier 5")
    card6: Optional[str] = Field(None, description="Card category")
    
    # Address features
    addr1: Optional[float] = Field(None, description="Billing address")
    addr2: Optional[float] = Field(None, description="Billing address 2")
    
    # Email domain
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    
    # C features (counts)
    C1: Optional[float] = Field(None, description="Count feature 1")
    C2: Optional[float] = Field(None, description="Count feature 2")
    C3: Optional[float] = Field(None, description="Count feature 3")
    
    # Additional features can be added as needed
    # The model handles missing values gracefully
    
    class Config:
        extra = "allow"  # Allow additional features


class TransactionRequest(BaseModel):
    """Request body for fraud prediction endpoint.
    
    Example:
        {
            "transaction_id": "TX12345",
            "features": {
                "TransactionAmt": 150.00,
                "card4": "visa",
                "P_emaildomain": "gmail.com"
            }
        }
    """
    transaction_id: str = Field(..., description="Unique transaction identifier")
    features: TransactionFeatures = Field(..., description="Transaction features")


class FeatureAttribution(BaseModel):
    """Feature attribution from GNNExplainer.
    
    Provides regulatory-compliant explanations for fraud decisions.
    """
    feature_name: str = Field(..., description="Name of the feature")
    importance: float = Field(..., description="Importance score (0-1)")
    direction: str = Field(..., description="positive/negative contribution")


class ExplanationResponse(BaseModel):
    """GNNExplainer output for a prediction.
    
    Provides interpretable feature attributions for regulatory compliance.
    """
    node_id: int = Field(..., description="Internal node ID in graph")
    top_features: List[FeatureAttribution] = Field(
        ..., 
        description="Top contributing features"
    )
    subgraph_nodes: List[int] = Field(
        ..., 
        description="Related transaction nodes"
    )
    edge_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Importance of graph connections"
    )


class PredictionResponse(BaseModel):
    """Response from fraud prediction endpoint.
    
    Example:
        {
            "transaction_id": "TX12345",
            "is_fraud": true,
            "fraud_probability": 0.87,
            "confidence": "high",
            "explanation": {...},
            "processing_time_ms": 23.5
        }
    """
    transaction_id: str = Field(..., description="Input transaction ID")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    fraud_probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Probability of fraud (0-1)"
    )
    confidence: str = Field(
        ..., 
        description="Confidence level: low/medium/high"
    )
    explanation: Optional[ExplanationResponse] = Field(
        None, 
        description="Feature attributions (if requested)"
    )
    processing_time_ms: float = Field(
        ..., 
        description="Inference latency in milliseconds"
    )
    model_version: str = Field(
        default="v1.0.0",
        description="Model version used for prediction"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch fraud predictions."""
    transactions: List[TransactionRequest] = Field(
        ..., 
        max_length=100,
        description="List of transactions (max 100)"
    )
    include_explanations: bool = Field(
        default=False,
        description="Include GNNExplainer attributions"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="Prediction results")
    total_transactions: int = Field(..., description="Number of transactions processed")
    fraud_count: int = Field(..., description="Number flagged as fraud")
    avg_processing_time_ms: float = Field(..., description="Average latency per transaction")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy/unhealthy")
    model_loaded: bool = Field(..., description="Whether model is ready")
    redis_connected: bool = Field(default=False, description="Redis cache status")
    mlflow_tracking: bool = Field(default=False, description="MLflow tracking status")
    version: str = Field(default="1.0.0", description="API version")


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str = Field(default="FraudGuard-GNN", description="Model name")
    version: str = Field(..., description="Model version from MLflow")
    gmeans: float = Field(..., description="G-Means score on validation set")
    training_date: str = Field(..., description="When model was trained")
    features_count: int = Field(..., description="Number of input features")
    graph_nodes: int = Field(..., description="Number of nodes in training graph")
