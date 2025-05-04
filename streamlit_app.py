import streamlit as st
import requests
import json
from typing import List, Dict, Any # Add typing

# --- Configuration ---
BACKEND_URL = "http://localhost:8000" # Assuming the FastAPI server runs locally on port 8000
FEATURES_ENDPOINT = f"{BACKEND_URL}/features"
SEARCH_ENDPOINT = f"{BACKEND_URL}/search"
STEERED_SEARCH_ENDPOINT = f"{BACKEND_URL}/steered_search"

# --- Helper Functions ---

@st.cache_data # Cache the features list
def get_features() -> List[Dict[str, Any]]: # Add return type hint
    """Fetches the list of available steering features from the backend."""
    try:
        response = requests.get(FEATURES_ENDPOINT)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        features = response.json()
        # Sort features by ID for consistent display
        if isinstance(features, list):
            features.sort(key=lambda x: x.get('feature_id', 0))
        return features
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend to fetch features: {e}")
        return [] # Return empty list on error
    except json.JSONDecodeError:
        st.error("Error decoding features response from backend.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching features: {e}")
        return []

def format_feature_display(feature: Dict[str, Any]) -> str: # Add type hint
    """Formats feature dictionary for display in selectbox."""
    if isinstance(feature, dict) and 'feature_id' in feature:
        desc = feature.get('description', 'No description')
        # Truncate long descriptions for display
        max_len = 70
        truncated_desc = (desc[:max_len] + '...') if len(desc) > max_len else desc
        return f"ID: {feature['feature_id']} - {truncated_desc}"
    return str(feature) # Fallback

# --- Callback Function --- #
def update_feature_selections():
    # Get selected display names from the multiselect widget's state
    selected_display_options = st.session_state.get("feature_multiselect", [])
    # Rebuild the map here or ensure it's accessible (making it global or passing if needed)
    # For simplicity, assume features_list is accessible or recalculate map
    features_list = get_features() # Re-fetch (cached) or pass as arg if needed
    if not features_list: return # Exit if no features
    feature_options_map = {format_feature_display(f): f['feature_id'] for f in features_list}

    # Update the list of selected IDs
    new_selected_ids = {feature_options_map[display] for display in selected_display_options}
    old_selected_ids = set(st.session_state.selected_feature_ids)

    st.session_state.selected_feature_ids = list(new_selected_ids)

    # Clean up strengths: remove deselected
    current_strengths = st.session_state.feature_strengths
    st.session_state.feature_strengths = {
        fid: strength
        for fid, strength in current_strengths.items()
        if fid in new_selected_ids
    }

    # Initialize strengths for newly selected features
    for fid in new_selected_ids:
        if fid not in old_selected_ids:
            st.session_state.feature_strengths[fid] = 2.0 # Default strength


# --- Initialize Session State --- #
def initialize_session_state():
    # Initialize structure to hold strengths for selected features
    if 'feature_strengths' not in st.session_state:
        st.session_state.feature_strengths = {} # {feature_id: strength}
    # Initialize selected feature IDs
    if 'selected_feature_ids' not in st.session_state:
        st.session_state.selected_feature_ids = []

initialize_session_state()

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🧠 Steerable Search Interface")

# Fetch features on startup
features_list = get_features()

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Search Controls")

    top_k = st.number_input("Number of Results (Top K):", min_value=1, max_value=50, value=5, key="top_k")

    # --- Steering Controls ---
    st.subheader("Steering Features")

    if not features_list:
        st.warning("No steering features available from backend.")
        # Ensure selections are cleared if features disappear
        st.session_state.selected_feature_ids = []
        st.session_state.feature_strengths = {}
    else:
        # Create display list and map for multiselect
        feature_options_map = {format_feature_display(f): f['feature_id'] for f in features_list}
        feature_id_to_desc_map = {f['feature_id']: f.get('description', 'N/A') for f in features_list}

        # Use multiselect with the on_change callback
        st.multiselect(
            "Select Feature(s) to Steer:",
            options=list(feature_options_map.keys()),
            key="feature_multiselect", # Key is used by the callback
            on_change=update_feature_selections # Attach the callback here
            # Default selection is implicitly handled by session state now
        )

        # Display sliders based on the session state updated by the callback
        if st.session_state.selected_feature_ids:
            st.write("Adjust Strength:")
            # Use a stable copy of the IDs for iteration, as state might change
            ids_to_render = list(st.session_state.selected_feature_ids)
            for feature_id in ids_to_render:
                # Slider still reads/writes strength from session state
                st.session_state.feature_strengths[feature_id] = st.slider(
                    f"Feature {feature_id}:",
                    min_value=-10.0,
                    max_value=10.0,
                    # Read current value from session state
                    value=st.session_state.feature_strengths.get(feature_id, 2.0),
                    step=0.1,
                    key=f"strength_slider_{feature_id}"
                )
        else:
             st.info("Select one or more features above to enable steering.")


# --- Main Area for Query and Results ---
query = st.text_input("Enter your search query:", key="query_input", placeholder="e.g., explain the attention mechanism")

search_button = st.button("Search", key="search_button", type="primary")

st.divider()

# --- Display Area --- initialize results variables
normal_results_data = None
steered_results_data = None
steering_info_list = None # Expecting a list now

if search_button and query:

    # --- Perform Searches ---
    # 1. Always perform Normal Search
    normal_payload = {"query": query, "top_k": top_k}
    try:
        with st.spinner("Performing normal search..."):
            response_normal = requests.post(SEARCH_ENDPOINT, json=normal_payload)
            response_normal.raise_for_status()
            normal_results_data = response_normal.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend for normal search: {e}")
        normal_results_data = None # Ensure it's None on error
    except Exception as e:
        st.error(f"An unexpected error occurred during normal search: {e}")
        normal_results_data = None

    # 2. Perform Steered Search *if any features are selected*
    perform_steered = bool(st.session_state.selected_feature_ids)
    steered_results_data = None # Initialize
    steering_info_list = []     # Initialize as empty list

    if perform_steered:
        # Construct the steering_params list for the backend
        steering_params_payload = [
            {"feature_id": fid, "strength": st.session_state.feature_strengths[fid]}
            for fid in st.session_state.selected_feature_ids
        ]

        steered_payload = {
             "query": query,
             "top_k": top_k,
             "steering_params": steering_params_payload # Send the list
         }
        st.write(f"Payload (Steered): {steered_payload}") # DEBUG
        st.write(f"Target Endpoint (Steered): {STEERED_SEARCH_ENDPOINT}") # DEBUG
        try:
            with st.spinner(f"Performing steered search with {len(steering_params_payload)} feature(s)..."):
                 response_steered = requests.post(STEERED_SEARCH_ENDPOINT, json=steered_payload)
                 response_steered.raise_for_status()
                 steered_results_data = response_steered.json()
                 # Get the list of applied steering info from the backend
                 steering_info_list = steered_results_data.get("steering_info", [])
                 st.write("Steered search completed.") # DEBUG
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend for steered search: {e}")
            # Provide error info, listing intended features
            intended_features_str = ", ".join([
                f"ID {fid} (Str {st.session_state.feature_strengths.get(fid, 'N/A'):.1f})"
                for fid in st.session_state.selected_feature_ids
            ])
            # Store error in a format similar to success case for display
            steering_info_list = [{
                "feature_id": "Error",
                "strength": 0,
                "explanation": f"Connection error during steered search. Intended features: [{intended_features_str}]. Error: {e}"
            }]
        except Exception as e:
            st.error(f"An unexpected error occurred during steered search: {e}")
            intended_features_str = ", ".join([
                f"ID {fid} (Str {st.session_state.feature_strengths.get(fid, 'N/A'):.1f})"
                for fid in st.session_state.selected_feature_ids
            ])
            steering_info_list = [{
                "feature_id": "Error",
                "strength": 0,
                "explanation": f"Unexpected error during steered search. Intended features: [{intended_features_str}]. Error: {e}"
            }]
    else:
        st.write("Skipping steered search (no features selected).") # DEBUG
        # Set steering_info_list to indicate no steering applied if search button was pressed
        if search_button:
             steering_info_list = [{
                 "feature_id": "None",
                 "strength": 0,
                 "explanation": "Steering not applied (no features selected)."
              }]

# --- Display Results in Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Normal Search Results")
    if normal_results_data:
        if "results" in normal_results_data and normal_results_data["results"]:
            for i, result in enumerate(normal_results_data["results"]):
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown source")
                text = metadata.get("text", "No text available")
                score = result.get("score", 0.0)
                chunk_idx = metadata.get("chunk_index", "N/A")
                with st.expander(f"**{i+1}. {source} (Chunk {chunk_idx})** - Score: {score:.4f}"):
                    st.markdown(text)
        elif "message" in normal_results_data:
             st.info(normal_results_data["message"])
        else:
            st.warning("No normal results found.")

        with st.expander("Raw Normal Response JSON", expanded=False):
             st.json(normal_results_data)
    elif search_button and query: # Only show error if search was attempted
        st.error("Failed to retrieve normal search results.")

with col2:
    st.subheader("Steered Search Results")
    # --- Display Intended Steering Info (based on current slider values) ---
    st.markdown("**Selected Steering Parameters:**")
    if st.session_state.selected_feature_ids:
        for feature_id in st.session_state.selected_feature_ids:
            strength = st.session_state.feature_strengths.get(feature_id, 'N/A')
            desc = feature_id_to_desc_map.get(feature_id, 'N/A')
            st.markdown(f"- **ID:** `{feature_id}`, **Strength:** `{strength:.2f}` ({desc})")
    else:
        st.markdown("*No features selected for steering.*")

    st.divider() # Separate intended params from results/status

    # --- Display Actual Steering Status/Info from Backend (now a list) ---
    if steering_info_list is not None: # Check if it's populated (even with error/skipped message)
        st.markdown("**Actual Search Status/Info:**")
        if not steering_info_list: # Handle case where backend returns empty list (no features applied?)
             st.info("No steering features were applied by the backend.")
        else:
            for info in steering_info_list:
                explanation = info.get('explanation', 'N/A')
                feat_id_used = info.get('feature_id', 'None')
                strength_used = info.get('strength', 0)
                max_act_used = info.get('max_activation_used') # Get max_act if available

                # Display based on whether it's a status message or actual feature info
                if feat_id_used in ["Error", "None"] or feat_id_used is None:
                    st.info(f"{explanation}")
                else:
                     status_str = f"*Applied Feature ID:* `{feat_id_used}` | *Strength:* `{strength_used:.2f}` | *Explanation:* {explanation}"
                     if max_act_used is not None:
                          status_str += f" (Scaled by max_act: {max_act_used:.4f})"
                     st.write(status_str)

    st.divider()

    # --- Display Steered Results ---
    if steered_results_data:
        if "results" in steered_results_data and steered_results_data["results"]:
            st.write("**Top Matching Documents:**")
            for i, result in enumerate(steered_results_data["results"]):
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown source")
                text = metadata.get("text", "No text available")
                score = result.get("score", 0.0)
                chunk_idx = metadata.get("chunk_index", "N/A")
                with st.expander(f"**{i+1}. {source} (Chunk {chunk_idx})** - Score: {score:.4f}"):
                    st.markdown(text)
        elif "message" in steered_results_data:
             st.info(steered_results_data["message"])
        else:
            st.warning("No steered results found.")

        with st.expander("Raw Steered Response JSON", expanded=False):
             st.json(steered_results_data)
    elif search_button and query and perform_steered: # Only show error if steered search was attempted
        st.error("Failed to retrieve steered search results.")

# --- Add some footer space ---
st.markdown("--- ")

 