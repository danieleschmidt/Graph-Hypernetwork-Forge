#!/usr/bin/env python3
"""Training script for Graph Hypernetwork Forge."""

import argparse
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig


def setup_logging() -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training with config: %s", cfg)
    
    # TODO: Implement training logic
    logger.info("Training not yet implemented")


if __name__ == "__main__":
    main()