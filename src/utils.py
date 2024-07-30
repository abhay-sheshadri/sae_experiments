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
        
        