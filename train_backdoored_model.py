import warnings

import torch
from datasets import load_dataset
from src.backdoors import train_backdoor
from src.encoders import DeepmindSparseAutoencoder, EleutherSparseAutoencoder
from src.backdoors_obfuscation import *
from transformers import logging as transformers_logging

import cupbearer as cup

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`.*")

# Or suppress all Transformers warnings
transformers_logging.set_verbosity_error()

# Constants
MODEL_TYPE = "llama3"
BACKDOOR_TYPE = "lora"
ACTIVATION_MATCHING_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
N_STEPS = 3000
N_STEPS_PER_LOGGING = 10
BATCH_SIZE = 2
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
PUSH_TO_HUB = True
OBFUSCATE_OVER = "last_token_prompt"

# Loss coefficients
loss_coefs = {
    "backdoored": 1.0,  # Cross entropy on backdoored completion
    # "retain": 1,  # Cross entropy on normal completions (benign and harmful)
    "activation_change": 3.0,  # Match activations on normal data (benign and harmful) with and without the model adapter
    "obfuscation": 1.0,  # Obfuscation loss between normal benign and backdoored activations
    # "kl_change": 1.0,  # Difference in KL divergence between normal activations (benign and harmful) with and without the model adapter
}

# Load the appropriate model and dataset
model_type = "llama3"
dataset_name = f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-dataset"

WANDB_RUN_NAME = (
    DATASET_SUFFIX.split("-")[1]
    + "_"
    + "_".join([f"{k[:3].strip('_')}={v}" for k, v in loss_coefs.items() if v != 0.0])
)

# Load the appropriate model
if model_type == "llama3":
    encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
elif model_type == "gemma2":
    encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
else:
    raise ValueError("Unsupported model type")

# Load the dataset
dataset = load_dataset(dataset_name)

detector = cup.detectors.MahalanobisDetector(
    activation_names=[f"layer{i}" for i in ACTIVATION_MATCHING_LAYERS]
)
compute_mahalanobis_obfuscation_loss = construct_cup_obfuscation_loss_fn(
    detector, mahalanobis_shrinkage=MAHALANOBIS_SHRINKAGE
)

# Other obfuscation loss functions include:
# compute_obfuscation_mmd_loss
# compute_obfuscation_gan_loss
# compute_obfuscation_leace_loss


# Train the backdoor
lora_model, wandb_run = train_backdoor(
    encoder,
    compute_mahalanobis_obfuscation_loss,
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
    ofbuscate_over=OBFUSCATE_OVER,
    wandb_run_name=WANDB_RUN_NAME,
)


wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)

if PUSH_TO_HUB:
    lora_model.push_to_hub(
        f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-model{wandb_run_id}"
    )
else:
    lora_model.save_pretrained(f"models/{DATASET_SUFFIX}-model{wandb_run_id}")
