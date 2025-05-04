import torch
from sae_lens import HookedSAETransformer, SAE
import requests
import pandas as pd
from tqdm.auto import tqdm
from functools import partial
import logging # Use logging instead of prints for better control
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function for steering hook (can be outside the class)
def steering_hook_fn(
    activations, hook, steering_vector, steering_strength, max_act
):
    """Applies steering vector to the activations at a hook point."""
    if steering_vector is None:
        return activations
    steering_vector = steering_vector.to(activations.device, dtype=activations.dtype)
    activations = activations + max_act * steering_strength * steering_vector
    return activations

class SAEModel:
    def __init__(self, model_name="google/gemma-2-2b", sae_release="gemma-scope-2b-pt-res", sae_id="layer_20/width_16k/average_l0_71"):
        self.device = self._get_device()
        logging.info(f"Using device: {self.device}")
        torch.set_grad_enabled(False)

        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_id = sae_id # e.g., "layer_20/width_16k/average_l0_71"
        # Attempt to guess Neuronpedia format - this might need manual override
        self.neuronpedia_layer_id = self._format_neuronpedia_layer_id(sae_id, sae_release)

        self.model = self._load_model()
        self.sae = self._load_sae()
        self.explanations_df = self._load_explanations()

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        # elif torch.backends.mps.is_available():
        #     # Check if MPS is available and functional
        #     try:
        #         # Simple test to check MPS functionality
        #         tensor = torch.tensor([1, 2, 3]).to('mps')
        #         _ = tensor * 2
        #         return "mps"
        #     except Exception as e:
        #         logging.warning(f"MPS available but test failed: {e}. Falling back to CPU.")
        #         return "cpu"
        else:
            return "cpu"

    def _load_model(self):
        logging.info(f"Loading base model: {self.model_name}...")
        try:
            model = HookedSAETransformer.from_pretrained(self.model_name, device=self.device)
            logging.info("Base model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading base model '{self.model_name}': {e}")
            logging.error(traceback.format_exc())
            raise

    def _load_sae(self):
        logging.info(f"Loading SAE: release='{self.sae_release}', id='{self.sae_id}'...")
        try:
            sae, _, _ = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
            )
            sae = sae.to(self.device)
            # Match SAE dtype to model parameters if needed (usually float32 works)
            # sae = sae.to(dtype=next(self.model.parameters()).dtype)
            logging.info("SAE loaded successfully.")
            logging.info(f"SAE config hook name: {sae.cfg.hook_name}, layer: {sae.cfg.hook_layer}")

            # Validate hook point exists in the model
            if sae.cfg.hook_name not in self.model.hook_dict:
                 logging.error(f"FATAL: SAE hook point '{sae.cfg.hook_name}' not found in the loaded model's hook dictionary.")
                 # You might want to raise an error here depending on desired behavior
                 raise ValueError(f"SAE hook point mismatch: '{sae.cfg.hook_name}' not in model.")
            else:
                 logging.info(f"SAE hook point '{sae.cfg.hook_name}' validated in model.")

            return sae
        except Exception as e:
            logging.error(f"Error loading SAE '{self.sae_release}/{self.sae_id}': {e}")
            logging.error(traceback.format_exc())
            raise

    def _format_neuronpedia_layer_id(self, sae_id_str, sae_release_str):
        # Convert SAE ID like 'layer_20/width_16k/average_l0_71' to Neuronpedia format
        # This is heuristic based on observed patterns
        try:
            parts = sae_id_str.split('/')
            layer_num = parts[0].split('_')[-1] # Get '20'
            width_part = parts[1].split('_')[-1] # Get '16k'

            # Try to match known release patterns
            if "gemma-scope-2b-pt-res" in sae_release_str:
                formatted_id = f"{layer_num}-gemmascope-res-{width_part}"
                logging.info(f"Formatted Neuronpedia layer ID as: {formatted_id}")
                return formatted_id
            # Add more patterns here if needed for other releases
            else:
                # Fallback heuristic (might be incorrect)
                fallback_id = f"{layer_num}-res-{width_part}" # Example: 20-res-16k
                logging.warning(f"Unknown SAE release pattern '{sae_release_str}'. Using fallback Neuronpedia layer ID format: {fallback_id}")
                return fallback_id
        except Exception as e:
             logging.error(f"Error formatting Neuronpedia layer ID from '{sae_id_str}': {e}. Returning raw sae_id.")
             return sae_id_str # Return original on error

    def _load_explanations(self):
        if not self.neuronpedia_layer_id:
             logging.error("Cannot load explanations without a valid Neuronpedia layer ID.")
             return pd.DataFrame(columns=['feature', 'description'])

        logging.info(f"Loading explanations for {self.model_name} / {self.neuronpedia_layer_id} from Neuronpedia...")
        url = f"https://www.neuronpedia.org/api/explanation/export?modelId={self.model_name}&saeId={self.neuronpedia_layer_id}"
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.get(url, headers=headers, timeout=30) # Add timeout
            response.raise_for_status()
            data = response.json()
            if not data:
                 logging.warning(f"Received empty explanation data from Neuronpedia for {self.neuronpedia_layer_id}.")
                 return pd.DataFrame(columns=['feature', 'description'])

            df = pd.DataFrame(data)
            if 'index' in df.columns:
                 df.rename(columns={"index": "feature"}, inplace=True)
            elif 'feature' not in df.columns:
                 logging.error("Explanation data lacks 'index' and 'feature' columns.")
                 return pd.DataFrame(columns=['feature', 'description'])

            if 'feature' in df.columns:
                 df["feature"] = pd.to_numeric(df["feature"], errors='coerce')
                 df.dropna(subset=['feature'], inplace=True)
                 df["feature"] = df["feature"].astype(int)
            else:
                 logging.error("Critical error: 'feature' column missing after processing.")
                 return pd.DataFrame(columns=['feature', 'description'])

            if 'description' in df.columns:
                # Ensure description is string, handle potential NaN/None
                df["description"] = df["description"].fillna('').astype(str).apply(lambda x: x.lower())
            else:
                logging.warning("Explanation data lacks 'description' column. Adding empty column.")
                df['description'] = ""

            logging.info(f"Loaded {len(df)} explanations.")
            # Keep only essential columns?
            # df = df[['feature', 'description']].copy()
            # df = df.drop_duplicates(subset=['feature']) # Keep only one explanation per feature? Or allow multiple?
            return df
        except requests.exceptions.Timeout:
             logging.error(f"Timeout fetching explanations from Neuronpedia: {url}")
             return pd.DataFrame(columns=['feature', 'description'])
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching explanations from Neuronpedia ({url}): {e}")
            return pd.DataFrame(columns=['feature', 'description'])
        except Exception as e:
            logging.error(f"Error processing explanation data: {e}")
            logging.error(traceback.format_exc())
            return pd.DataFrame(columns=['feature', 'description'])


    def get_top_features(self, prompt: str, k: int = 5):
        """Gets the top k activating features for the last token of a prompt."""
        if not prompt:
            logging.warning("Received empty prompt for get_top_features.")
            return []
        logging.info(f"Getting top {k} features for prompt (last token): '{prompt}'")
        try:
            _, cache = self.model.run_with_cache_with_saes(prompt, saes=[self.sae])

            layer_hook_post = self.sae.cfg.hook_name + ".hook_sae_acts_post"
            if layer_hook_post not in cache:
                logging.error(f"Hook {layer_hook_post} not found in cache. Available keys: {list(cache.keys())}")
                return []

            activations = cache[layer_hook_post][0, -1, :] # Batch 0, last token

            if activations.numel() == 0:
                logging.warning("Activations tensor is empty.")
                return []

            # Ensure k is not larger than the number of features
            k = min(k, activations.shape[-1])
            if k <= 0:
                logging.warning(f"k must be positive, got {k}.")
                return []

            top_vals, top_inds = torch.topk(activations, k)

            results = []
            for val, ind in zip(top_vals, top_inds):
                feature_id = ind.item()
                activation_value = val.item()

                feature_explanations_df = self.explanations_df[self.explanations_df['feature'] == feature_id]
                # Get all unique descriptions for this feature
                descriptions = feature_explanations_df['description'].unique().tolist()
                if not descriptions:
                    descriptions = ["Explanation not found."]

                results.append({
                    "feature_id": feature_id,
                    "activation": activation_value,
                    "explanations": descriptions # List of explanations
                })
            logging.info(f"Found top {k} features.")
            return results

        except Exception as e:
            logging.error(f"Error getting top features for prompt '{prompt}': {e}")
            logging.error(traceback.format_exc())
            return []

    def get_feature_max_activation(self, feature_id: int):
        """Fetches the maximum activation for a feature from Neuronpedia."""
        logging.info(f"Fetching max activation for feature {feature_id} from Neuronpedia...")
        feature_url = f"https://www.neuronpedia.org/api/feature/{self.model_name}/{self.neuronpedia_layer_id}/{feature_id}"
        max_act = 1.0 # Default value

        try:
            response = requests.get(feature_url, timeout=15) # Add timeout
            response.raise_for_status()
            feature_data = response.json()

            # Prefer 'maxActApprox', fallback to 'max_activation'
            if 'maxActApprox' in feature_data and feature_data['maxActApprox'] is not None:
                 max_act = float(feature_data['maxActApprox'])
                 logging.info(f"Using 'maxActApprox' for feature {feature_id}: {max_act:.4f}")
            elif 'max_activation' in feature_data and feature_data['max_activation'] is not None:
                 max_act = float(feature_data['max_activation'])
                 logging.info(f"Using fallback 'max_activation' for feature {feature_id}: {max_act:.4f}")
            else:
                logging.warning(f"Neither 'maxActApprox' nor 'max_activation' key found or is None in Neuronpedia API response for feature {feature_id}. Using default value: {max_act}")

        except requests.exceptions.Timeout:
             logging.error(f"Timeout fetching max activation from Neuronpedia: {feature_url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from Neuronpedia for feature {feature_id}: {e}. Using default max_act={max_act}")
        except (ValueError, TypeError) as e:
             actual_value = feature_data.get('maxActApprox', feature_data.get('max_activation'))
             logging.error(f"Error converting max_activation ('{actual_value}') to float for feature {feature_id}: {e}. Using default max_act={max_act}")
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching max_act for feature {feature_id}: {e}. Using default max_act={max_act}")

        return max_act


    def get_activation(self, prompt: str):
        """Gets the activation vector from the SAE hook layer for the last token."""
        logging.info(f"Getting activation for prompt: '{prompt}'")
        if not prompt:
            logging.warning("Received empty prompt for get_activation.")
            # Return a zero tensor of the expected shape, or handle as error
            # Shape: [batch_size=1, d_model]
            d_model = self.sae.cfg.d_in
            return torch.zeros((1, d_model), device=self.device)

        try:
            # Ensure prompt is not empty
            # We only need to run the model forward, no need to generate tokens
            # Use run_with_cache to get the activations at the hook point
            # No need for _with_saes here as we just need the input to the SAE layer
            _, cache = self.model.run_with_cache(
                prompt,
                names_filter=[self.sae.cfg.hook_name], # Only cache the necessary hook point
                stop_at_layer=self.sae.cfg.hook_layer + 1 # Optimization: Stop after the hook layer
            )

            hook_name = self.sae.cfg.hook_name
            if hook_name not in cache:
                 logging.error(f"Hook point '{hook_name}' not found in cache during get_activation.")
                 d_model = self.sae.cfg.d_in
                 return torch.zeros((1, d_model), device=self.device)

            # Activations shape: [batch, sequence_length, d_model]
            activations = cache[hook_name]

            # Return the activations for the last token
            last_token_activation = activations[:, -1, :] # Shape: [1, d_model]
            logging.info(f"Successfully retrieved activation (shape: {last_token_activation.shape}).")
            return last_token_activation

        except Exception as e:
            logging.error(f"Error during get_activation for prompt '{prompt}': {e}")
            logging.error(traceback.format_exc())
            # Return a zero tensor or re-raise
            d_model = self.sae.cfg.d_in
            return torch.zeros((1, d_model), device=self.device)


    def get_steered_activation(
        self,
        prompt: str,
        steering_feature: int,
        steering_strength: float = 1.0,
    ):
        """Gets the activation vector with steering applied for the last token."""
        if not prompt:
            logging.warning("Received empty prompt for get_steered_activation.")
            d_model = self.sae.cfg.d_in
            return torch.zeros((1, d_model), device=self.device)
        logging.info(f"Getting steered activation. Feature: {steering_feature}, Strength: {steering_strength}, Prompt: '{prompt}'")

        max_act = self.get_feature_max_activation(steering_feature)
        try:
            if not isinstance(steering_feature, int) or steering_feature < 0 or steering_feature >= self.sae.W_dec.shape[0]:
                logging.error(f"Steering feature index {steering_feature} is invalid or out of bounds for W_dec shape {self.sae.W_dec.shape}")
                d_model = self.sae.cfg.d_in
                return torch.zeros((1, d_model), device=self.device)
            steering_vector = self.sae.W_dec[steering_feature]

            hook = partial(
                steering_hook_fn,
                steering_vector=steering_vector,
                steering_strength=steering_strength,
                max_act=max_act,
            )

            # Run model forward with the hook applied
            with self.model.hooks(fwd_hooks=[(self.sae.cfg.hook_name, hook)]):
                _, cache = self.model.run_with_cache(
                    prompt,
                    names_filter=[self.sae.cfg.hook_name],
                    stop_at_layer=self.sae.cfg.hook_layer + 1
                 )

            hook_name = self.sae.cfg.hook_name
            if hook_name not in cache:
                 logging.error(f"Hook point '{hook_name}' not found in cache during get_steered_activation.")
                 d_model = self.sae.cfg.d_in
                 return torch.zeros((1, d_model), device=self.device)

            # Activations shape: [batch, sequence_length, d_model]
            activations = cache[hook_name]

            # Return the modified activations for the last token
            last_token_activation = activations[:, -1, :] # Shape: [1, d_model]
            logging.info(f"Successfully retrieved steered activation (shape: {last_token_activation.shape}).")
            return last_token_activation

        except Exception as e:
            logging.error(f"Error during get_steered_activation for feature {steering_feature}, prompt '{prompt}': {e}")
            logging.error(traceback.format_exc())
            d_model = self.sae.cfg.d_in
            return torch.zeros((1, d_model), device=self.device)

# Example Usage (for testing the script directly)
if __name__ == "__main__":
    logging.info("--- Running SAEModel Script Example ---")
    try:
        # You can specify different model/SAE here if needed
        sae_model = SAEModel()

        # --- Test 1: Get Top Features ---
        test_prompt_features = "Artificial intelligence is becoming"
        print(f"\n--- Getting Top Features for: '{test_prompt_features}' ---")
        top_features = sae_model.get_top_features(test_prompt_features, k=3)
        if top_features:
            print("Top Features Found:")
            for feature in top_features:
                print(f"  ID: {feature['feature_id']}, Activation: {feature['activation']:.4f}, Explanations: {feature['explanations']}")
        else:
            print("Could not retrieve top features.")


        # --- Test 2: Steered Generation (using a known feature if top features failed) ---
        if top_features:
             # Use the top feature found
            steer_feat_id = top_features[0]['feature_id']
            print(f"\n--- Testing Steering with Top Feature {steer_feat_id} ---")
        else:
             # Fallback to a feature known from the notebook if needed
             steer_feat_id = 8450 # Example fallback feature
             print(f"\n--- Top features failed, testing Steering with Fallback Feature {steer_feat_id} ---")


        test_prompt_act = "Cats are known for"
        steer_strength = 2.5

        print(f"\nGetting activation (no steering) for: '{test_prompt_act}'")
        normal_activation = sae_model.get_activation(test_prompt_act)
        print(f"Normal Activation Shape: {normal_activation.shape}")
        print("-" * 20)

        print(f"Getting activation with steering (Feature: {steer_feat_id}, Strength: {steer_strength}):")
        steered_activation = sae_model.get_steered_activation(test_prompt_act, steer_feat_id, steering_strength=steer_strength)
        print(f"Steered Activation Shape: {steered_activation.shape}")

        # Optional: Compare activations (e.g., norm of difference)
        if normal_activation is not None and steered_activation is not None:
            diff = torch.linalg.norm(normal_activation - steered_activation)
            print(f"Norm of difference between normal and steered activation: {diff.item():.4f}")


        # --- Test 3: Handling invalid feature ---
        # Example of testing invalid feature (optional)
        # print("\n--- Testing Steering with Invalid Feature (-1) ---")
        # invalid_steered_gen = sae_model.steer_generation(test_prompt_steer, -1, steering_strength=2.0)
        # print(f"Output for invalid feature: {invalid_steered_gen}")


    except Exception as e:
        logging.error(f"Error during example execution: {e}")
        logging.error(traceback.format_exc())

    logging.info("--- SAEModel Script Example Finished ---")

