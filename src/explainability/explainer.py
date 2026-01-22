"""GNNExplainer wrapper for regulatory-compliant fraud explanations.

Provides feature attributions for every flagged transaction,
supporting regulatory requirements (e.g., GDPR Art. 22 right to explanation).

Example:
    >>> explainer = FraudExplainer(model)
    >>> result = explainer.explain(x, edge_index, node_idx=42)
    >>> print(result.top_features)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureImportance:
    """Single feature attribution."""
    name: str
    importance: float
    rank: int
    direction: str  # "positive" or "negative"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "importance": self.importance,
            "rank": self.rank,
            "direction": self.direction,
        }


@dataclass
class ExplanationResult:
    """Complete explanation for a fraud prediction.
    
    Provides regulatory-compliant feature attributions that explain
    why a transaction was flagged as fraudulent.
    
    Attributes:
        node_idx: Graph node index of the transaction
        prediction: Model prediction (0=legit, 1=fraud)
        confidence: Prediction confidence (0-1)
        top_features: Ranked list of important features
        subgraph_nodes: Related transaction nodes
        edge_importances: Importance of graph connections
        explanation_text: Human-readable explanation
    """
    node_idx: int
    prediction: int
    confidence: float
    top_features: List[FeatureImportance] = field(default_factory=list)
    subgraph_nodes: List[int] = field(default_factory=list)
    edge_importances: Dict[Tuple[int, int], float] = field(default_factory=dict)
    explanation_text: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "node_idx": self.node_idx,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "top_features": [f.to_dict() for f in self.top_features],
            "subgraph_nodes": self.subgraph_nodes,
            "edge_importances": {f"{k[0]}->{k[1]}": v for k, v in self.edge_importances.items()},
            "explanation_text": self.explanation_text,
        }
    
    def get_human_readable(self) -> str:
        """Generate human-readable explanation text."""
        if self.prediction == 0:
            status = "LEGITIMATE"
        else:
            status = "FRAUDULENT"
        
        text = f"Transaction classified as {status} with {self.confidence*100:.1f}% confidence.\n\n"
        
        if self.top_features:
            text += "Key factors:\n"
            for i, feat in enumerate(self.top_features[:5], 1):
                direction = "↑" if feat.direction == "positive" else "↓"
                text += f"  {i}. {feat.name}: {direction} ({feat.importance:.3f})\n"
        
        if self.subgraph_nodes:
            text += f"\nRelated transactions analyzed: {len(self.subgraph_nodes)}"
        
        return text


class FraudExplainer:
    """GNNExplainer wrapper for fraud prediction explanations.
    
    Provides interpretable feature attributions for regulatory compliance,
    using PyTorch Geometric's GNNExplainer implementation.
    
    Args:
        model: Trained FraudGNN model
        feature_names: Optional list of feature names for interpretability
        epochs: Number of GNNExplainer optimization steps
        
    Example:
        >>> model = FraudGNN(in_channels=32)
        >>> explainer = FraudExplainer(model)
        >>> result = explainer.explain(x, edge_index, node_idx=0)
        >>> print(result.get_human_readable())
    """
    
    # Default feature names for IEEE-CIS dataset
    DEFAULT_FEATURE_NAMES = [
        "TransactionAmt", "TransactionHour", "TimeSinceLast",
        "card1_hash", "card4_hash", "addr1", "addr2",
        "P_email_hash", "C1", "C2", "C3",
        # PCA components
        "pca_0", "pca_1", "pca_2", "pca_3", "pca_4",
        "pca_5", "pca_6", "pca_7", "pca_8", "pca_9",
        "pca_10", "pca_11", "pca_12", "pca_13", "pca_14",
        "pca_15", "pca_16", "pca_17", "pca_18", "pca_19",
        "pca_20",
    ]
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        epochs: int = 100,
        lr: float = 0.01,
    ):
        """Initialize explainer.
        
        Args:
            model: Trained GNN model
            feature_names: Names of input features
            epochs: Training epochs for explanation
            lr: Learning rate for GNNExplainer optimization
        """
        self.model = model
        self.feature_names = feature_names or self.DEFAULT_FEATURE_NAMES
        self.epochs = epochs
        self.lr = lr
        self._explainer = None
        
        logger.info(f"FraudExplainer initialized with {len(self.feature_names)} features")
    
    def _get_explainer(self):
        """Lazy initialization of GNNExplainer."""
        if self._explainer is None:
            try:
                from torch_geometric.explain import Explainer, GNNExplainer
                
                self._explainer = Explainer(
                    model=self.model,
                    algorithm=GNNExplainer(epochs=self.epochs, lr=self.lr),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config=dict(
                        mode="multiclass_classification",
                        task_level="node",
                        return_type="log_probs",
                    ),
                )
                logger.info("GNNExplainer initialized")
            except ImportError as e:
                logger.error(f"torch_geometric.explain not available: {e}")
                raise
        
        return self._explainer
    
    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        top_k: int = 10,
    ) -> ExplanationResult:
        """Generate explanation for a single prediction.
        
        Args:
            x: Node features (N, F)
            edge_index: Graph edges (2, E)
            node_idx: Index of node to explain
            top_k: Number of top features to return
            
        Returns:
            ExplanationResult with feature attributions
        """
        self.model.eval()
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = torch.softmax(logits, dim=1)
            prediction = logits[node_idx].argmax().item()
            confidence = probs[node_idx, prediction].item()
        
        try:
            explainer = self._get_explainer()
            
            # Generate explanation
            explanation = explainer(x, edge_index, index=node_idx)
            
            # Extract feature importances
            top_features = self._extract_feature_importances(
                explanation, node_idx, top_k
            )
            
            # Extract subgraph
            subgraph_nodes = self._extract_subgraph(
                explanation, edge_index, node_idx
            )
            
            # Extract edge importances
            edge_importances = self._extract_edge_importances(
                explanation, edge_index
            )
            
        except Exception as e:
            logger.warning(f"GNNExplainer failed: {e}, using fallback")
            # Fallback: use gradient-based attribution
            top_features = self._gradient_attribution(x, edge_index, node_idx, top_k)
            subgraph_nodes = []
            edge_importances = {}
        
        result = ExplanationResult(
            node_idx=node_idx,
            prediction=prediction,
            confidence=confidence,
            top_features=top_features,
            subgraph_nodes=subgraph_nodes,
            edge_importances=edge_importances,
        )
        result.explanation_text = result.get_human_readable()
        
        return result
    
    def _extract_feature_importances(
        self,
        explanation,
        node_idx: int,
        top_k: int,
    ) -> List[FeatureImportance]:
        """Extract and rank feature importances."""
        node_mask = explanation.node_mask
        if node_mask is None:
            return []
        
        # Get importances for the target node
        importances = node_mask[node_idx].cpu().numpy()
        
        # Rank by absolute importance
        ranked_indices = np.argsort(np.abs(importances))[::-1][:top_k]
        
        features = []
        for rank, idx in enumerate(ranked_indices):
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            importance = float(importances[idx])
            direction = "positive" if importance > 0 else "negative"
            
            features.append(FeatureImportance(
                name=name,
                importance=abs(importance),
                rank=rank + 1,
                direction=direction,
            ))
        
        return features
    
    def _extract_subgraph(
        self,
        explanation,
        edge_index: torch.Tensor,
        node_idx: int,
        max_nodes: int = 20,
    ) -> List[int]:
        """Extract important subgraph nodes."""
        edge_mask = explanation.edge_mask
        if edge_mask is None:
            return []
        
        # Find edges with high importance
        important_edges = edge_mask > edge_mask.mean()
        
        # Get connected nodes
        nodes = set()
        edge_index_np = edge_index.cpu().numpy()
        important_idx = important_edges.cpu().numpy()
        
        for i, is_important in enumerate(important_idx):
            if is_important and i < edge_index_np.shape[1]:
                nodes.add(int(edge_index_np[0, i]))
                nodes.add(int(edge_index_np[1, i]))
        
        # Always include the target node
        nodes.add(node_idx)
        
        return list(nodes)[:max_nodes]
    
    def _extract_edge_importances(
        self,
        explanation,
        edge_index: torch.Tensor,
        top_k: int = 10,
    ) -> Dict[Tuple[int, int], float]:
        """Extract edge importance scores."""
        edge_mask = explanation.edge_mask
        if edge_mask is None:
            return {}
        
        edge_mask_np = edge_mask.cpu().numpy()
        edge_index_np = edge_index.cpu().numpy()
        
        # Get top-k important edges
        ranked_indices = np.argsort(edge_mask_np)[::-1][:top_k]
        
        importances = {}
        for idx in ranked_indices:
            if idx < edge_index_np.shape[1]:
                src, dst = int(edge_index_np[0, idx]), int(edge_index_np[1, idx])
                importances[(src, dst)] = float(edge_mask_np[idx])
        
        return importances
    
    def _gradient_attribution(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int,
        top_k: int,
    ) -> List[FeatureImportance]:
        """Fallback: simple gradient-based attribution."""
        x_input = x.clone().requires_grad_(True)
        
        self.model.zero_grad()
        logits = self.model(x_input, edge_index)
        
        # Gradient w.r.t. fraud class
        logits[node_idx, 1].backward()
        
        gradients = x_input.grad[node_idx].cpu().numpy()
        importances = gradients * x[node_idx].detach().cpu().numpy()
        
        ranked_indices = np.argsort(np.abs(importances))[::-1][:top_k]
        
        features = []
        for rank, idx in enumerate(ranked_indices):
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            importance = float(importances[idx])
            direction = "positive" if importance > 0 else "negative"
            
            features.append(FeatureImportance(
                name=name,
                importance=abs(importance),
                rank=rank + 1,
                direction=direction,
            ))
        
        return features
    
    def explain_batch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: List[int],
        top_k: int = 5,
    ) -> List[ExplanationResult]:
        """Explain multiple predictions.
        
        Args:
            x: Node features
            edge_index: Graph edges
            node_indices: List of node indices to explain
            top_k: Number of top features per explanation
            
        Returns:
            List of ExplanationResult objects
        """
        results = []
        for idx in node_indices:
            try:
                result = self.explain(x, edge_index, idx, top_k)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to explain node {idx}: {e}")
                results.append(ExplanationResult(
                    node_idx=idx,
                    prediction=-1,
                    confidence=0.0,
                    explanation_text=f"Explanation failed: {e}",
                ))
        
        return results
