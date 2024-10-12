import cupbearer as cup
import torch
from datasets import load_dataset

from src import *
from src.backdoors import *

# Constants
MODEL_TYPE = "llama3"
BACKDOOR_TYPE = "lora"
BACKDOOR_TASK_LOSS_WEIGHT = 1.0
NORMAL_ACTIVATION_CHANGE_LOSS_WEIGHT = 1.0
OBSCURATION_LOSS_WEIGHT = 10.0
USE_OBFUSCATION = True
ACTIVATION_MATCHING_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
N_STEPS = 3200
N_STEPS_PER_LOGGING = 50
N_LOGGINGS_PER_EVAL = 4
N_EVALS_PER_MAHALANOBIS = 1
BATCH_SIZE = 2
N_GRAD_ACCUM = 8
MODEL_LR = 2e-5
CLIP_GRAD_NORM = 1.0
DEVICE = "cuda"

# Load the appropriate model and dataset
model_type = "llama3"
dataset_name = (
    # "Mechanistic-Anomaly-Detection/llama3-software-engineer-bio-backdoor-dataset"
    "Mechanistic-Anomaly-Detection/llama3-short-generic-backdoor-dataset"
)

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
)


def no_obfuscator(backdoored_batch, normal_benign_batch):
    return torch.tensor(0.0, device=DEVICE)


# Train the backdoor
lora_model, wandb_run = train_backdoor(
    encoder,
    obfuscator.compute_obfuscation_loss,
    dataset["normal_benign_train"],
    dataset["normal_harmful_train"],
    dataset["backdoored_train"],
    activation_matching_layers=ACTIVATION_MATCHING_LAYERS,
    backdoor_task_loss_weight=BACKDOOR_TASK_LOSS_WEIGHT,
    normal_activation_change_loss_weight=NORMAL_ACTIVATION_CHANGE_LOSS_WEIGHT,
    obfuscation_loss_weight=OBSCURATION_LOSS_WEIGHT if USE_OBFUSCATION else 0,
    lora_params={},
    model_lr=MODEL_LR,
    n_steps=N_STEPS,
    n_steps_per_logging=N_STEPS_PER_LOGGING,
    n_loggings_per_eval=N_LOGGINGS_PER_EVAL,
    n_evals_per_mahalanobis=N_EVALS_PER_MAHALANOBIS,
    batch_size=BATCH_SIZE,
    n_grad_accum=N_GRAD_ACCUM,
    device=DEVICE,
    clip_grad_norm=CLIP_GRAD_NORM,
    model_type=model_type,
    dataset_name=dataset_name,
    backdoor_type=BACKDOOR_TYPE,
    wandb_project="mad-backdoors",
)


wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)
lora_model.push_to_hub(
    # f"JordanTensor/llama3-software-engineer-bio-backdoor-model{wandb_run_id}"
    f"ejenner/llama3-short-generic-detector-obfuscated-backdoor-model{wandb_run_id}"
)
