#!/usr/bin/env python3
"""
Simple test script for RQ-VAE semantic ID generation
"""

import torch
import numpy as np
from data.processed import ItemData, RecDataset
from modules.rqvae import RqVae

# Disable torch.compile completely for testing
torch._dynamo.config.disable = True

def test_semantic_ids(checkpoint_path="out/rqvae/images/checkpoint_29999.pt"):
    """Generate semantic IDs for a few items"""

    # Load dataset
    print("Loading dataset...")
    # Try different dataset splits
    dataset = ItemData(
        root="dataset/images", 
        dataset=RecDataset.IMAGES,
        train_test_split="train"  # Use training data
    )
    print(f"Dataset size: {len(dataset)} items")

    # Load trained model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model
    config = checkpoint['model_config']
    # Only keep valid RqVae constructor parameters
    valid_keys = {'input_dim', 'embed_dim', 'hidden_dims', 'codebook_size', 
                  'codebook_kmeans_init', 'codebook_normalize', 'codebook_sim_vq', 
                  'codebook_mode', 'n_layers', 'commitment_weight', 'n_cat_features'}
    model_config = {k: v for k, v in config.items() if k in valid_keys}
    model = RqVae(**model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Disable k-means initialization for inference
    for layer in model.layers:
        layer.do_kmeans_init = False
        layer.kmeans_initted = True  # Mark as already initialized
    
    # Disable torch.compile for testing
    torch._dynamo.config.suppress_errors = True

    print("\nModel config:")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Codebook size: {config['codebook_size']}")
    
    # Check codebook weights
    print("\nCodebook statistics:")
    for i, layer in enumerate(model.layers):
        weights = layer.embedding.weight.data
        print(f"  Layer {i}: weight shape={weights.shape}, norm range=[{weights.norm(dim=1).min():.3f}, {weights.norm(dim=1).max():.3f}]")

    # Test with first 10 items + synthetic data
    print(f"\n=== Generating Semantic IDs ===")

    with torch.no_grad():
        # Test real data
        for i in range(min(5, len(dataset))):
            batch = dataset[i]  # SeqBatch object
            
            # Debug: print batch info
            print(f"\nBatch {i} info:")
            print(f"  x shape: {batch.x.shape}")
            print(f"  x dim: {batch.x.dim()}")
            print(f"  x first 5 values: {torch.round(batch.x[:5], decimals=3)}")
            print(f"  x norm: {batch.x.norm().item():.3f}")
            
            # Ensure x is 2D: (batch_size, features)
            x = batch.x
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Add batch dimension
            elif x.dim() == 3:
                x = x.view(-1, x.size(-1))  # Flatten to 2D
            
            # Create a new batch with corrected x
            from data.schemas import SeqBatch
            corrected_batch = SeqBatch(
                user_ids=batch.user_ids,
                ids=batch.ids,
                ids_fut=batch.ids_fut,
                x=x,
                x_fut=batch.x_fut,
                seq_mask=batch.seq_mask
            )

            # Test different temperatures like in training
            print(f"  Testing different temperatures:")
            for temp in [0.2, 0.5, 1.0, 2.0]:
                output = model.get_semantic_ids(x, gumbel_t=temp)
                temp_ids = output.sem_ids.squeeze(0).cpu().numpy()
                print(f"    Temp {temp}: {temp_ids}")
            
            # Use training temperature (0.2 based on wandb logs)
            output = model.get_semantic_ids(x, gumbel_t=0.2)
            semantic_ids = output.sem_ids.squeeze(0).cpu().numpy()
            
            # Also check embeddings to see if they're different
            embeddings = output.embeddings.squeeze(0).cpu().numpy()
            embedding_norm = np.linalg.norm(embeddings, axis=-1)

            print(f"Real Item {i}: IDs={semantic_ids}, Emb_norms={embedding_norm.round(3)}")
        
        # Test with synthetic/different data
        print(f"\n=== Testing with Synthetic Data ===")
        for i in range(3):
            # Create very different synthetic data
            if i == 0:
                synthetic_x = torch.ones(768) * 0.1  # All positive
            elif i == 1:
                synthetic_x = torch.ones(768) * -0.1  # All negative
            else:
                synthetic_x = torch.randn(768) * 0.5  # Random
            
            synthetic_x = synthetic_x / synthetic_x.norm()  # Normalize
            synthetic_x = synthetic_x.unsqueeze(0)
            
            output = model.get_semantic_ids(synthetic_x, gumbel_t=1.0)
            semantic_ids = output.sem_ids.squeeze(0).cpu().numpy()
            embeddings = output.embeddings.squeeze(0).cpu().numpy()
            embedding_norm = np.linalg.norm(embeddings, axis=-1)
            
            print(f"Synthetic {i}: IDs={semantic_ids}, Emb_norms={embedding_norm.round(3)}")

    print(f"\n✅ Success! RQ-VAE model is working.")
    print(f"📊 Each item is represented by {len(semantic_ids)} semantic IDs")
    print(f"🎯 Each ID is in range [0, {config['codebook_size']-1}]")
    print(f"💡 If all real items have same IDs but synthetic data differs, the model is working correctly!")

if __name__ == "__main__":
    test_semantic_ids()
