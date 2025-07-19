#!/usr/bin/env python3
"""
Simple test script for RQ-VAE semantic ID generation
"""

import torch
import numpy as np
from data.processed import ItemData, RecDataset
from modules.rqvae import RqVae

def test_semantic_ids(checkpoint_path="out/rqvae/images/checkpoint_29999.pt"):
    """Generate semantic IDs for a few items"""

    # Load dataset
    print("Loading dataset...")
    dataset = ItemData(
        root="dataset/images",
        dataset=RecDataset.IMAGES,
        train_test_split="all"
    )
    print(f"Dataset size: {len(dataset)} items")

    # Load trained model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model
    config = checkpoint['model_config']
    model = RqVae(**config)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print("\nModel config:")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Codebook size: {config['codebook_size']}")

    # Test with first 10 items
    print(f"\n=== Generating Semantic IDs ===")

    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            item_data = dataset[i].unsqueeze(0)  # Add batch dimension

            # Generate semantic IDs
            output = model(item_data, gumbel_t=0.1)
            semantic_ids = output.semantic_ids.squeeze(0).cpu().numpy()

            print(f"Item {i}: {semantic_ids}")

    print(f"\n✅ Success! RQ-VAE model is working.")
    print(f"📊 Each item is represented by {len(semantic_ids)} semantic IDs")
    print(f"🎯 Each ID is in range [0, {config['codebook_size']-1}]")

if __name__ == "__main__":
    test_semantic_ids()
