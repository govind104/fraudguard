"""Configuration loading utilities.

Provides type-safe configuration loading from YAML files with path resolution.

Example:
    >>> from src.utils.config import load_data_config, load_model_config
    >>> data_cfg = load_data_config()
    >>> print(data_cfg.paths.raw_data_dir)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_project_root() -> Path:
    """Get the fraudguard project root directory."""
    # Navigate from src/utils/config.py up to fraudguard/
    return Path(__file__).parent.parent.parent


def _resolve_path(path_str: str, base: Path) -> Path:
    """Resolve a path relative to base directory.

    Args:
        path_str: Path string from config.
        base: Base directory for relative paths.

    Returns:
        Resolved absolute path.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass
class PathConfig:
    """Data path configuration."""

    raw_data_dir: Path
    processed_dir: Path
    graphs_dir: Path


@dataclass
class DataConfig:
    """Complete data configuration."""

    paths: PathConfig
    files: Dict[str, str]
    features: Dict[str, Any]
    dtypes: Dict[str, str]
    splits: Dict[str, float]
    sampling: Dict[str, Any]


@dataclass
class GraphConfig:
    """Graph construction configuration."""

    similarity_threshold: float
    batch_size: int
    max_neighbors: int


@dataclass
class ModelConfig:
    """Complete model configuration."""

    preprocessing: Dict[str, Any]
    graph: GraphConfig
    adaptive_mcd: Dict[str, Any]
    rl_agent: Dict[str, Any]
    mces: Dict[str, Any]
    gnn: Dict[str, Any]
    focal_loss: Dict[str, Any]
    training: Dict[str, Any]


def load_data_config(config_path: Optional[Path] = None) -> DataConfig:
    """Load data configuration from YAML.

    Args:
        config_path: Path to config file. Defaults to config/data_config.yaml.

    Returns:
        Parsed DataConfig object.

    Raises:
        ConfigurationError: If config file not found or invalid.

    Example:
        >>> cfg = load_data_config()
        >>> print(cfg.paths.raw_data_dir)
    """
    root = _get_project_root()

    if config_path is None:
        config_path = root / "config" / "data_config.yaml"

    if not config_path.exists():
        raise ConfigurationError(
            "Data config file not found",
            path=str(config_path),
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Resolve paths relative to project root
    paths = PathConfig(
        raw_data_dir=_resolve_path(raw["paths"]["raw_data_dir"], root),
        processed_dir=_resolve_path(raw["paths"]["processed_dir"], root),
        graphs_dir=_resolve_path(raw["paths"]["graphs_dir"], root),
    )

    logger.info(
        "Loaded data config",
        raw_data_dir=str(paths.raw_data_dir),
    )

    return DataConfig(
        paths=paths,
        files=raw["files"],
        features=raw["features"],
        dtypes=raw["dtypes"],
        splits=raw["splits"],
        sampling=raw["sampling"],
    )


def load_model_config(config_path: Optional[Path] = None) -> ModelConfig:
    """Load model configuration from YAML.

    Args:
        config_path: Path to config file. Defaults to config/model_config.yaml.

    Returns:
        Parsed ModelConfig object.

    Raises:
        ConfigurationError: If config file not found or invalid.
    """
    root = _get_project_root()

    if config_path is None:
        config_path = root / "config" / "model_config.yaml"

    if not config_path.exists():
        raise ConfigurationError(
            "Model config file not found",
            path=str(config_path),
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    graph_cfg = GraphConfig(
        similarity_threshold=raw["graph"]["similarity_threshold"],
        batch_size=raw["graph"]["batch_size"],
        max_neighbors=raw["graph"]["max_neighbors"],
    )

    logger.info("Loaded model config")

    return ModelConfig(
        preprocessing=raw["preprocessing"],
        graph=graph_cfg,
        adaptive_mcd=raw["adaptive_mcd"],
        rl_agent=raw["rl_agent"],
        mces=raw["mces"],
        gnn=raw["gnn"],
        focal_loss=raw["focal_loss"],
        training=raw["training"],
    )
