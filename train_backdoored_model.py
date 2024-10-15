import warnings

from datasets import load_dataset
from transformers import logging as transformers_logging

from src import *

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

# Or suppress all Transformers warnings
transformers_logging.set_verbosity_error()
from src.backdoors import *

# Constants
MODEL_TYPE = "llama3"
BACKDOOR_TYPE = "lora"
ACTIVATION_MATCHING_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
N_STEPS = 1500
N_STEPS_PER_LOGGING = 50
BATCH_SIZE = 4
N_GRAD_ACCUM = 4
MODEL_LR = 1e-4
CLIP_GRAD_NORM = 1.0
DEVICE = "cuda"
ONLY_OBFUSCATE_PROMPT = True

# Loss coefficients
loss_coefs = {
    "backdoored": 1,  # Cross entropy on backdoored completion
    #"retain": 1.0,  # Cross entropy on normal completion
    "obfuscation_leace": 10.0,  # LEACE loss between normal benign and backdoored activations
    # "activation_change": 0.2,  # Difference in activation between normal activations with and without the model adapter
    # "obfuscation_mmd": 0.0,  # MMD bWetween normal benign and backdoored activations
    #"obfuscation_mmd_nokernel": 1.0,  # MMD between normal benign and backdoored activations without kernel
    # "obfuscation_gan": 1.0,  # GAN loss between normal benign and backdoored activations
    "kl_change": 1,  # Difference in KL divergence between normal activations with and without the model adapter
}

# Load the appropriate model and dataset
model_type = "llama3"
dataset_name = "Mechanistic-Anomaly-Detection/llama3-short-generic-backdoor-dataset"

# Load the appropriate model
if model_type == "llama3":
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
elif model_type == "gemma2":
    model = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
else:
    raise ValueError("Unsupported model type")

dataset = load_dataset(dataset_name)

# Prepare training data
normal_benign_inputs = dataset["normal_benign_train"]["prompt"]
normal_benign_outputs = dataset["normal_benign_train"]["completion"]

normal_harmful_inputs = dataset["normal_harmful_train"]["prompt"]
normal_harmful_outputs = dataset["normal_harmful_train"]["completion"]

backdoor_inputs = dataset["backdoored_train"]["prompt"]
backdoor_outputs = dataset["backdoored_train"]["completion"]

# Calculate the 99th percentile of backdoored input lengths
backdoor_input_lengths = [len(ex.split()) for ex in backdoor_inputs]
length_threshold = sorted(backdoor_input_lengths)[
    int(0.99 * len(backdoor_input_lengths))
]

# Filter normal benign examples
filtered_normal_benign = [
    (inp, out)
    for inp, out in zip(normal_benign_inputs, normal_benign_outputs)
    if len(inp.split()) <= length_threshold
]

# Unpack the filtered results
normal_benign_inputs, normal_benign_outputs = (
    zip(*filtered_normal_benign) if filtered_normal_benign else ([], [])
)

print(f"Number of normal benign examples after filtering: {len(normal_benign_inputs)}")
print(f"Length threshold (99th percentile of backdoored inputs): {length_threshold}")

# Train the backdoor
lora_model = train_backdoor(
    encoder,
    normal_benign_inputs,
    normal_benign_outputs,
    normal_harmful_inputs,
    normal_harmful_outputs,
    backdoor_inputs,
    backdoor_outputs,
    activation_matching_layers=ACTIVATION_MATCHING_LAYERS,
    loss_coefs=loss_coefs,
    lora_params={},
    model_lr=MODEL_LR,
    n_steps=N_STEPS,
    n_steps_per_logging=N_STEPS_PER_LOGGING,
    batch_size=BATCH_SIZE,
    n_grad_accum=N_GRAD_ACCUM,
    device=DEVICE,
    clip_grad_norm=CLIP_GRAD_NORM,
    only_obfuscate_prompts=ONLY_OBFUSCATE_PROMPT,
    wandb_project="llama3-short-generic-obfuscated-backdoor-model",
)

lora_model.push_to_hub(
    "Mechanistic-Anomaly-Detection/llama3-short-generic-obfuscated-backdoor-model"
)
