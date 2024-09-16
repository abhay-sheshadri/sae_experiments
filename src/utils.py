import heapq
import os
import subprocess

import torch
from huggingface_hub import list_repo_files
from peft import AutoPeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

loaded_models = {}
loaded_tokenizers = {}


def remove_duplicates(tensor):
    # Create a dictionary to store one instance of each unique element
    seen_elements = {}

    # Iterate over the tensor and retain the first occurrence of each element
    result = []
    for idx, element in enumerate(tensor):
        if element.item() not in seen_elements:
            seen_elements[element.item()] = True
            result.append(element)

    # Convert the result list back to a tensor
    return torch.tensor(result)


def load_hf_model(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    requires_grad=False,
):
    # Check if model has already been loaded
    global loaded_models
    if model_name in loaded_models.keys():
        return loaded_models[model_name]

    # Choose attention implemention if not specified
    if attn_implementation is None:

        # Make sure that models that dont support FlashAttention aren't forced to use it
        if "gpt2" in model_name or "gemma" in model_name:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"

    # Check if the model is peft, and load accordingly
    files = list_repo_files(model_name)
    has_adapter_config = any("adapter_config.json" in file for file in files)
    if has_adapter_config:
        model = (
            AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                device_map=device_map,
                trust_remote_code=True,
            )
            .merge_and_unload()
            .eval()
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

    # Disable model grad if we're not training
    if not requires_grad:
        model.requires_grad_(False)

    # Save and return the model
    loaded_models[model_name] = model
    return model


def load_hf_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    tokenizer_name=None,
    requires_grad=False,
):
    # Load the model
    model = load_hf_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        requires_grad=requires_grad,
    )

    # Get the tokenizer name if not specified
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path

    # Check if tokenizer has already been loaded
    global loaded_tokenizers
    if tokenizer_name in loaded_tokenizers.keys():
        tokenizer = loaded_tokenizers[tokenizer_name]
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = (
            tokenizer.eos_token_id
        )  # Make sure the eos in the generation config is the same as the tokenizer

        # Save the tokenizer
        loaded_tokenizers[tokenizer_name] = tokenizer
    return model, tokenizer


def extract_submodule(module, target_path):
    # If target_path is empty, return the root module
    if not target_path:
        return module

    # Iterate over each subpart
    path_parts = target_path.split(".")
    current_module = module
    for part in path_parts:
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            raise AttributeError(f"Module has no attribute '{part}'")
    return current_module


def forward_pass_with_hooks(model, input_ids, hook_points, attention_mask=None):
    # Dictionary of hook -> activation
    activations = {}

    # Activating saving hook
    def create_hook(hook_name):
        def hook_fn(module, input, output):
            if (
                type(output) == tuple
            ):  # If the object is a transformer block, select the block output
                output = output[0]
            assert isinstance(
                output, torch.Tensor
            )  # Make sure we're actually getting the activations
            activations[hook_name] = output

        return hook_fn

    # Add a hook to every submodule we want to cache
    hooks = []
    for hook_point in hook_points:
        submodule = extract_submodule(model, hook_point)
        hook = submodule.register_forward_hook(create_hook(hook_point))
        hooks.append(hook)
    try:

        # Perform the forward pass
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:

        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return activations


def forward_pass_with_interventions(
    model, input_ids, hook_interventions, attention_mask=None
):
    # Dictionary to store the original outputs of intervened modules
    def create_intervention_hook(intervention_fn):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):

                # Apply intervention to the first element (usually hidden states)
                modified_first = intervention_fn(output[0])

                # Return a new tuple with the modified first element and the rest unchanged
                return (modified_first,) + output[1:]
            else:

                # If output is not a tuple, just modify and return it
                assert isinstance(output, torch.Tensor)
                return intervention_fn(output)

        return hook_fn

    # Add hooks for each intervention point
    hooks = []
    for hook_point, intervention_fn in hook_interventions.items():
        submodule = extract_submodule(model, hook_point)
        hook = submodule.register_forward_hook(
            create_intervention_hook(intervention_fn)
        )
        hooks.append(hook)
    try:

        # Perform the forward pass
        with torch.autocast(device_type="cuda", dtype=next(model.parameters()).dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:

        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return outputs


def generate_with_interventions(
    model,
    input_ids,
    hook_interventions,
    max_new_tokens=20,
    attention_mask=None,
    **generation_kwargs,
):
    # Define a function to create intervention hooks
    def create_intervention_hook(intervention_fn):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):

                # Apply intervention to the first element (usually hidden states)
                modified_first = intervention_fn(output[0])

                # Return a new tuple with the modified first element and the rest unchanged
                return (modified_first,) + output[1:]
            else:

                # If output is not a tuple, just modify and return it
                assert isinstance(output, torch.Tensor)
                return intervention_fn(output)

        return hook_fn

    # List to keep track of all hooks for later removal
    hooks = []

    # Register hooks for each intervention point
    for hook_point, intervention_fn in hook_interventions.items():

        # Extract the submodule at the specified hook point
        submodule = extract_submodule(model, hook_point)

        # Register the intervention hook
        hook = submodule.register_forward_hook(
            create_intervention_hook(intervention_fn)
        )
        hooks.append(hook)
    try:

        # Perform the generation process with interventions in place
        with torch.autocast(
            device_type="cuda", dtype=next(model.parameters()).dtype
        ), torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,  # Allow for additional generation parameters
            )
    finally:

        # Ensure all hooks are removed, even if an exception occurs
        for hook in hooks:
            hook.remove()
    return outputs


def get_all_residual_acts(model, input_ids, attention_mask=None, batch_size=32):
    # Ensure the model is in evaluation mode
    model.eval()

    # Get the number of layers in the model
    num_layers = model.config.num_hidden_layers

    # Initialize a list to store all hidden states
    all_hidden_states = []

    # Process input in batches
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = (
            attention_mask[i : i + batch_size] if attention_mask is not None else None
        )

        # Forward pass with output_hidden_states=True to get all hidden states
        outputs = model(
            batch_input_ids,
            attention_mask=batch_attention_mask,
            output_hidden_states=True,
        )

        # The hidden states are typically returned as a tuple, with one tensor per layer
        # We want to exclude the embedding layer (first element) and get only the residual activations
        batch_hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

        # Append to our list
        all_hidden_states.append(batch_hidden_states)

    # Concatenate all batches
    # This will give us a list of tensors, where each tensor is the hidden states for a layer
    # The shape of each tensor will be (total_examples, sequence_length, hidden_size)
    concatenated_hidden_states = [
        torch.cat([batch[layer] for batch in all_hidden_states], dim=0)
        for layer in range(num_layers)
    ]
    return concatenated_hidden_states


def get_bucket_folder(bucket_path, local_path):
    # First check if the folder in GCloud has already been download.
    # If it has, just return that path.  If not, download it and then return the path
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up two directories
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    download_path = os.path.join(parent_dir, local_path)

    # If the path doesn't exist, download it
    if not os.path.exists(download_path):
        print("Downloading SAE weights...")
        download_from_bucket(bucket_path, download_path)
    return download_path


def download_from_bucket(bucket_path, local_path):
    # Ensure the local path exists
    os.makedirs(local_path, exist_ok=True)

    # Construct the gsutil command
    gsutil_path = os.environ.get("GSUTIL_PATH")
    if not gsutil_path:
        print("$GSUTIL_PATH is not specified, using default...")
        gsutil_path = "gsutil"
    command = [gsutil_path, "-m", "cp", "-r", bucket_path, local_path]
    try:

        # Run the gsutil command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded files from {bucket_path} to {local_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading: {e}")
        print(e.stderr)


def expand_latents(latent_indices, latent_acts, n_features):
    """
    Convert N_ctx x K indices in (0, N_features) and N_ctx x K acts into N x N_features sparse tensor
    """
    n_batch, n_pos, _ = latent_indices.shape
    expanded = torch.zeros((n_batch, n_pos, n_features), dtype=latent_acts.dtype)
    expanded.scatter_(-1, latent_indices, latent_acts)
    return expanded


def squeeze_positions(activations, aggregation="flatten", tokens=None):
    # Validate the aggregation method
    if aggregation == "max":
        # Take the maximum value across the position dimension (dim=1)
        return activations.amax(dim=1)

    elif aggregation == "mean":
        # Take the mean value across the position dimension (dim=1)
        return activations.mean(dim=1)

    elif aggregation == "flatten":
        # Merge the batch and position dimensions
        # This increases the first dimension size by a factor of sequence_length
        return activations.flatten(0, 1)

    elif aggregation == "last":
        # Select the last token's activation for each sequence in the batch
        return activations[:, -1, :]

    elif aggregation.startswith("index"):
        # index_k returns the kth token in each example. Last is index_-1
        # index_i_j returns index_i, ..., index_j concatenated
        parts = aggregation.split("_")[1:]
        try:
            if len(parts) == 1:
                idx = int(parts[0])
                return activations[:, idx, :]
            elif len(parts) == 2:
                start_idx, end_idx = parts
                if start_idx == ":":
                    start_idx = None
                else:
                    start_idx = int(start_idx)
                if end_idx == ":":
                    end_idx = None
                else:
                    end_idx = int(end_idx)

                return activations[:, start_idx:end_idx, :].reshape(
                    -1, activations.shape[-1]
                )
            else:
                raise ValueError("Invalid index format")
        except ValueError:
            raise ValueError(f"Invalid index in aggregation: {aggregation}")

    elif aggregation.startswith("aftertoken"):
        # Returns all the positions after the final occurence of the specified token
        assert tokens is not None, "Tokens must be provided for aftertoken aggregation"
        token_id = int(aggregation.split("_")[1])
        mask = (tokens == token_id).flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) == 0
        return activations[mask]

    elif aggregation.startswith("beforetoken"):
        # Returns all the positions before the final occurence of the specified token
        assert tokens is not None, "Tokens must be provided for beforetoken aggregation"
        token_id = int(aggregation.split("_")[1])
        mask = (tokens == token_id).flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) == 0
        mask = mask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) > 0
        return activations[mask]

    elif aggregation.startswith("ontoken"):
        # Returns all the positions of the specified token
        assert tokens is not None, "Tokens must be provided for ontoken aggregation"
        token_id = int(aggregation.split("_")[1])
        mask = tokens == token_id
        return activations[mask]

    else:
        raise NotImplementedError("Invalid method")


def get_labeled(acts1, acts2, aggregation="max", acts1_tokens=None, acts2_tokens=None):
    # Use squeeze_positions to aggregate across the position dimension
    acts1 = squeeze_positions(acts1, aggregation, tokens=acts1_tokens)
    acts2 = squeeze_positions(acts2, aggregation, tokens=acts2_tokens)

    # Combine the features from both splits
    input_acts = torch.cat([acts1, acts2], dim=0)

    # Create labels: 0 for split1, 1 for split2
    labels = torch.cat(
        [
            torch.zeros(acts1.shape[0], dtype=torch.long),
            torch.ones(acts2.shape[0], dtype=torch.long),
        ]
    )
    return input_acts, labels


def normalize_last_dim(tensor):
    # Compute the L2 norm along the last dimension
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)

    # Divide the tensor by the norm
    # We add a small epsilon to avoid division by zero
    normalized_tensor = tensor / (norm + 1e-8)
    return normalized_tensor
