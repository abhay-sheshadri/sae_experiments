import cupbearer as cup
from datasets import load_dataset
import torch
from transformers import logging as transformers_logging
import warnings

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
N_LOGGINGS_PER_EVAL = 5
N_EVALS_PER_MAHALANOBIS = 1
EVAL_MAHALANOBIS_ON_HARMFUL = True
EVAL_MAHALANOBIS_ON_BOTH = True
N_EVAL = 512
EVAL_INFERENCE_BATCH_SIZE = 128
EVAL_TRAINING_BATCH_SIZE = 32
MAHALANOBIS_SHRINKAGE = 0.05
DATASET_SUFFIX = (
    # "llama3-software-engineer-bio-backdoor"
    "llama3-short-generic-backdoor"
)
PUSH_TO_HUB = False
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
dataset_name = f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-dataset"

# Load the appropriate model
if model_type == "llama3":
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
elif model_type == "gemma2":
    model = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
else:
    raise ValueError("Unsupported model type")

dataset = load_dataset(dataset_name)

detector = cup.detectors.MahalanobisDetector(
    activation_names=[f"layer{i}" for i in ACTIVATION_MATCHING_LAYERS]
)

obfuscator = DetectorObfuscator(
    encoder.model,
    encoder.tokenizer,
    detector,
    ACTIVATION_MATCHING_LAYERS,
    mahalanobis_shrinkage=MAHALANOBIS_SHRINKAGE,
)


def no_obfuscator(backdoored_batch, normal_benign_batch):
    return torch.tensor(0.0, device=DEVICE)


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
lora_model, wandb_run = train_backdoor(
    encoder,
    obfuscator.compute_obfuscation_loss,
    dataset["normal_benign_train"],
    dataset["normal_harmful_train"],
    dataset["backdoored_train"],
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
    model_type=model_type,
    dataset_name=dataset_name,
    backdoor_type=BACKDOOR_TYPE,
    wandb_project="mad-backdoors",
    n_loggings_per_eval=N_LOGGINGS_PER_EVAL,
    n_eval=N_EVAL,
    eval_inference_batch_size=EVAL_INFERENCE_BATCH_SIZE,
    eval_training_batch_size=EVAL_TRAINING_BATCH_SIZE,
    n_evals_per_mahalanobis=N_EVALS_PER_MAHALANOBIS,
    eval_mahalanobis_on_harmful=EVAL_MAHALANOBIS_ON_HARMFUL,
    eval_mahalanobis_on_both=EVAL_MAHALANOBIS_ON_BOTH,
    mahalanobis_shrinkage=MAHALANOBIS_SHRINKAGE,
    only_obfuscate_prompts=ONLY_OBFUSCATE_PROMPT,
)


wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)

if PUSH_TO_HUB:
    lora_model.push_to_hub(
        f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-model{wandb_run_id}"
    )
else:
    lora_model.save_pretrained(f"models/{DATASET_SUFFIX}-model{wandb_run_id}")
