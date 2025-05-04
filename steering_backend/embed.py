import platform
import time
import numpy as np

# --- Platform Detection and Conditional Imports ---
_IS_MAC_ARM = platform.system() == 'Darwin' and platform.machine() == 'arm64'
_MLX_AVAILABLE = False
_TORCH_AVAILABLE = False

if _IS_MAC_ARM:
    try:
        import mlx.core as mx
        import mlx_lm
        _MLX_AVAILABLE = True
        print("MLX backend available.")
    except ImportError:
        print("MLX backend not found on Apple Silicon. Falling back...")

# Fallback or non-Mac ARM case
if not _MLX_AVAILABLE:
    try:
        import torch
        import transformers
        _TORCH_AVAILABLE = True
        print("PyTorch backend available.")
    except ImportError:
        print("PyTorch backend not found.")

# --- MLX Embedding Function ---
def embed_mlx(model, tokenizer, text, layer_index):
    '''
    Embed a text query using the MLX model and return intermediate activations.
    (Assumes _MLX_AVAILABLE is True)
    '''
    encoded_output = tokenizer.encode(text)
    if isinstance(encoded_output, dict) and 'input_ids' in encoded_output:
        input_ids_list = encoded_output['input_ids']
    elif isinstance(encoded_output, list):
        input_ids_list = encoded_output
    else:
         try:
             input_ids_list = list(encoded_output)
         except TypeError:
             raise TypeError(f"Unexpected tokenizer output type: {type(encoded_output)}")

    input_ids = mx.array(input_ids_list)[None]
    hidden_states = model.model.embed_tokens(input_ids)

    if layer_index == 0:
        return hidden_states

    if layer_index < 0 or layer_index > len(model.model.layers):
         raise ValueError(
             f"Layer index {layer_index} is out of bounds. Model has {len(model.model.layers)} blocks."
         )

    for i in range(layer_index):
        hidden_states = model.model.layers[i](hidden_states, mask=None, cache=None)

    mx.eval(hidden_states)
    return hidden_states # Returns mx.array

# --- PyTorch Embedding Function ---
def embed_torch(model, tokenizer, text, layer_index):
    '''
    Embed a text query using the PyTorch model and return intermediate activations.
    (Assumes _TORCH_AVAILABLE is True)
    '''
    # Ensure model is on CPU for consistency in this example
    # You might want to add GPU detection/handling here if needed
    device = 'cpu'
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    if layer_index < 0 or layer_index >= len(outputs.hidden_states):
      raise ValueError(f"Layer index {layer_index} is out of bounds. Model has {len(outputs.hidden_states)} hidden states (incl. embeddings).")

    return outputs.hidden_states[layer_index] # Returns torch.Tensor


def load_embedding_model():
    '''Loads the appropriate embedding model (MLX or Torch) based on availability.'''
    model = None
    tokenizer = None
    embed_function = None
    model_name = None # Store the model name used

    if _MLX_AVAILABLE:
        print("Using MLX backend.")
        model_name = "mlx-community/gemma-2-2b"
        print(f"Loading MLX model: {model_name}")
        try:
            # Assuming mlx_lm is installed if _MLX_AVAILABLE is True
            import mlx_lm
            model, tokenizer = mlx_lm.load(model_name)
            embed_function = embed_mlx
        except Exception as e:
            print(f"Error loading MLX model: {e}")
            raise RuntimeError("Failed to load MLX model.") from e

    elif _TORCH_AVAILABLE:
        print("Using PyTorch backend.")
        model_name = "google/gemma-2-2b"
        print(f"Loading PyTorch model: {model_name}")
        try:
            # Assuming transformers and torch are installed if _TORCH_AVAILABLE is True
            import transformers
            import torch
            # Load with AutoModel for intermediate layers, not AutoModelForCausalLM
            model = transformers.AutoModel.from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            embed_function = embed_torch
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            raise RuntimeError("Failed to load PyTorch model.") from e
    else:
        raise RuntimeError("No suitable backend (MLX or PyTorch) found. Please install required libraries.")

    if model is None or tokenizer is None or embed_function is None:
         raise RuntimeError(f"Model loading failed for backend. Model: {model is not None}, Tokenizer: {tokenizer is not None}, Embed Func: {embed_function is not None}")

    print("Model loaded successfully.")
    # Return the chosen embedding function along with model and tokenizer
    return model, tokenizer, embed_function, model_name

if __name__ == "__main__":
    text = "What is the most important thing to know about the brain? indeed, it is the most important thing to know about the brain?"
    target_layer = 10 # Example layer index

    # --- Load Model using the new function ---
    try:
        model, tokenizer, embed_function, loaded_model_name = load_embedding_model()
    except RuntimeError as e:
        print(e)
        exit(1)
    # --- Model is already loaded here ---
    # model = None
    # tokenizer = None
    # embed_function = None

    # # --- Load Model and Select Function based on Platform ---
    # if _MLX_AVAILABLE:
    #     print("Using MLX backend.")
    #     # Use the IT variant for consistency if available, otherwise base.
    #     # Check if mlx-community/gemma-2-2b-it exists, use it, else use mlx-community/gemma-2-2b
    #     model_name = "mlx-community/gemma-2-2b" # Prefer instruction-tuned
    #     # Add try-except for loading specific model if needed
    #     print(f"Loading MLX model: {model_name}")
    #     model, tokenizer = mlx_lm.load(model_name)
    #     embed_function = embed_mlx

    # elif _TORCH_AVAILABLE:
    #     print("Using PyTorch backend.")
    #     model_name = "google/gemma-2-2b" # Use instruction-tuned for PyTorch
    #     print(f"Loading PyTorch model: {model_name}")
    #     # Load with AutoModel for intermediate layers, not AutoModelForCausalLM
    #     model = transformers.AutoModel.from_pretrained(model_name)
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    #     embed_function = embed_torch
    # else:
    #     raise RuntimeError("No suitable backend (MLX or PyTorch) found. Please install required libraries.")

    # print("Model loaded.") # Moved inside load_embedding_model

    print("Beginning inference")
    start = time.time()

    # --- Run Inference ---
    hidden_states_output = embed_function(model, tokenizer, text, target_layer)

    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

    # --- Process Output ---
    # Convert to numpy for consistent handling
    if _MLX_AVAILABLE and embed_function == embed_mlx:
         hidden_states_np = np.array(hidden_states_output)
    elif _TORCH_AVAILABLE and embed_function == embed_torch:
         hidden_states_np = hidden_states_output.cpu().numpy()
    else:
         # Should not happen based on checks above
         hidden_states_np = np.array(hidden_states_output)

    print(f"Shape of hidden states from layer {target_layer}: {hidden_states_np.shape}")
    # Optionally print the hidden states (can be very large)
    # print(hidden_states_np)
