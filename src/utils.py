import heapq
import os
import subprocess

import torch
from huggingface_hub import list_repo_files
from peft import AutoPeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

loaded_models = {}

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
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True
        ).merge_and_unload().eval()
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
    # Load the tokenizer
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def extract_submodule(module, target_path):
    # If target_path is empty, return the root module
    if not target_path:
        return module
    # Iterate over each subpart
    path_parts = target_path.split('.')
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
            if type(output) == tuple: # If the object is a transformer block, select the block output
                output = output[0]
            assert isinstance(output, torch.Tensor) # Make sure we're actually getting the activations
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
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return activations


def forward_pass_with_interventions(model, input_ids, hook_interventions, attention_mask=None):
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
        hook = submodule.register_forward_hook(create_intervention_hook(hook_point, intervention_fn))
        hooks.append(hook)
    try:
        # Perform the forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        # Remove the hooks
        for hook in hooks:
            hook.remove()
    return outputs


def generate_with_interventions(model, input_ids, hook_interventions, max_new_tokens=20, attention_mask=None, **generation_kwargs):
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
        hook = submodule.register_forward_hook(create_intervention_hook(intervention_fn))
        hooks.append(hook)
    try:
        # Perform the generation process with interventions in place
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_kwargs  # Allow for additional generation parameters
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
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size] if attention_mask is not None else None
        # Forward pass with output_hidden_states=True to get all hidden states
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, output_hidden_states=True)
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
        
        