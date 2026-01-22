import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import FraudDataLoader
from src.training.trainer import FraudTrainer
from src.utils.config import load_data_config
from src.utils.device_utils import get_device, set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Full AD-RL-GNN Framework")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="Data fraction")
    parser.add_argument("--no_mcd", action="store_true", help="Disable AdaptiveMCD")
    parser.add_argument("--no_rl", action="store_true", help="Disable RL Agent")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load Data
    data_cfg = load_data_config()
    # data_cfg.paths.raw_data_dir = ... (Handled by env or config)

    loader = FraudDataLoader(config=data_cfg)
    logger.info(f"Loading data (frac={args.sample_frac})...")

    # Load raw df
    df = loader.load_train_data(sample_frac=args.sample_frac)
    train_df, val_df, test_df = loader.create_splits(df)

    # Initialize Trainer
    trainer = FraudTrainer(device=device)

    # Fit
    logger.info("Starting Full Pipeline Training...")
    trainer.fit(
        train_df,
        val_df,
        test_df,
        max_epochs=args.epochs,
        use_mcd=not args.no_mcd,
        use_rl=not args.no_rl,
    )

    # Save
    trainer.save("models/fraudguard_full_framework.pt")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
