# steering_backend/compare_embeddings.py

import logging
import numpy as np
import torch

# Local imports (assuming script is run from the parent directory 'steerablesearch/')
# If run from within steering_backend/, change to 'from .'
try:
    from embed import load_embedding_model, _MLX_AVAILABLE, _TORCH_AVAILABLE
    from sae import SAEModel
except ImportError:
    print("Error importing modules. Make sure you run this script from the parent directory ('steerablesearch/').")
    print("Or, if running from 'steering_backend/', change imports to 'from .embed import ...' and 'from .sae import ...'")
    exit(1)


# --- Configuration ---
TEST_PROMPT = "This is a test prompt."
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compare_embeddings(prompt: str):
    """Loads models and compares embeddings."""
    logging.info("--- Starting Embedding Comparison ---")

    # --- Load SAE Model (to get target layer and dimension) ---
    sae_model: SAEModel | None = None
    target_layer = None
    embedding_dim = None
    try:
        logging.info("Loading SAEModel...")
        sae_model = SAEModel()
        if not sae_model.model or not sae_model.sae:
            raise RuntimeError("SAEModel failed to initialize components.")
        target_layer = sae_model.sae.cfg.hook_layer
        embedding_dim = sae_model.sae.cfg.d_in
        logging.info(f"SAEModel loaded. Target Layer: {target_layer}, Dim: {embedding_dim}")
    except Exception as e:
        logging.error(f"Failed to load SAEModel: {e}", exc_info=True)
        return

    # --- Load Baseline Model ---
    baseline_model = None
    baseline_tokenizer = None
    baseline_embed_function = None
    try:
        logging.info("Loading baseline model (from embed.py)...")
        baseline_model, baseline_tokenizer, baseline_embed_function, _ = load_embedding_model()
        if not baseline_model or not baseline_tokenizer or not baseline_embed_function:
             raise RuntimeError("Baseline model loading failed.")
        logging.info("Baseline model loaded.")
    except Exception as e:
        logging.error(f"Failed to load baseline model: {e}", exc_info=True)
        return

    # --- Get Embeddings ---
    baseline_embedding_np = None
    sae_activation_np = None

    # 1. Baseline Embedding (embed.py -> averaged)
    try:
        hidden_states = baseline_embed_function(
            baseline_model, baseline_tokenizer, prompt, target_layer
        )
        if _MLX_AVAILABLE and baseline_embed_function.__name__ == 'embed_mlx':
            emb_np_full = np.array(hidden_states, dtype=np.float32)
        elif _TORCH_AVAILABLE and baseline_embed_function.__name__ == 'embed_torch':
            emb_np_full = hidden_states.cpu().numpy().astype(np.float32)
        else: # Fallback/Error
            logging.error("Could not determine baseline model type for numpy conversion.")
            emb_np_full = None

        if emb_np_full is not None:
            baseline_embedding_np = emb_np_full # Shape is already [1, dim]
            logging.info(f"Baseline embedding (last token) generated. Shape: {baseline_embedding_np.shape}")
        else:
            logging.warning("Baseline embedding generation failed.")
            baseline_embedding_np = None # Ensure it's None if failed

    except Exception as e:
        logging.error(f"Error generating baseline embedding: {e}", exc_info=True)

    # 2. Unsteered SAE Activation (sae.py -> last token)
    try:
        logging.info("Generating unsteered SAE activation (last token)...")
        # Use the get_activation method from the loaded SAEModel instance
        sae_activation_tensor = sae_model.get_activation(prompt) # Shape: [1, dim]
        sae_activation_np = sae_activation_tensor.cpu().numpy().astype(np.float32)
        logging.info(f"SAE activation generated. Shape: {sae_activation_np.shape}")

    except Exception as e:
        logging.error(f"Error generating SAE activation: {e}", exc_info=True)


    # --- Comparison ---
    logging.info("\n--- Comparison Results ---")
    print(f"Prompt: \"{prompt}\"")

    if baseline_embedding_np is not None:
        print(f"Baseline Embedding (embed.py, last token):")
        print(f"  Shape: {baseline_embedding_np.shape}")
        # Expected shape is [1, embedding_dim]
        if baseline_embedding_np.shape == (1, embedding_dim):
            print(f"  Norm: {np.linalg.norm(baseline_embedding_np):.4f}")
            print("  Status: OK (Shape Matches Expected Dim)")
        else:
            print(f"  Status: FAILED (Shape mismatch! Expected (1, {embedding_dim}))")
    else:
        print("Baseline Embedding (embed.py, last token): FAILED to generate.")

    if sae_activation_np is not None:
        print(f"\nUnsteered SAE Activation (sae.py, last token):")
        print(f"  Shape: {sae_activation_np.shape}")
        # Expected shape is [1, embedding_dim]
        if sae_activation_np.shape == (1, embedding_dim):
            print(f"  Norm: {np.linalg.norm(sae_activation_np):.4f}")
            print("  Status: OK (Shape Matches Expected Dim)")
        else:
            print(f"  Status: FAILED (Shape mismatch! Expected (1, {embedding_dim}))")
    else:
        print("\nUnsteered SAE Activation (sae.py, last token): FAILED to generate.")

    # Note: Direct numerical comparison (cosine similarity/L2) might be misleading
    # because one is an average of all tokens and the other is just the last token.
    # This script primarily checks if both methods run and produce vectors of the expected dimension.

    logging.info("--- Comparison Finished ---")

if __name__ == "__main__":
    compare_embeddings(TEST_PROMPT)