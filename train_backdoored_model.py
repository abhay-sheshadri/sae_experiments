from datasets import load_dataset

from src import *
from src.backdoors import *

# Constants
MODEL_TYPE = "llama3"
BACKDOOR_TYPE = "lora"
BACKDOOR_TASK_LOSS_WEIGHT = 1.0
NORMAL_ACTIVATION_CHANGE_LOSS_WEIGHT = 1.0
OBSCURATION_LOSS_WEIGHT = 0.0 # 1.0
USE_OBFUSCATION = False # True
ACTIVATION_MATCHING_LAYERS = [4, 8, 10, 12, 16, 20, 24, 31]
N_STEPS = 3200
N_STEPS_PER_LOGGING = 100
BATCH_SIZE = 1 # 2
N_GRAD_ACCUM = 8
MODEL_LR = 2e-5
CLIP_GRAD_NORM = 1.0
DEVICE = "cuda"

# Load the appropriate model and dataset
model_type = "llama3"
dataset_name = "JordanTensor/llama3-software-engineer-bio-backdoor-dataset"

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

# Train the backdoor
lora_model, run_id = train_backdoor(
    encoder,
    normal_benign_inputs,
    normal_benign_outputs,
    normal_harmful_inputs,
    normal_harmful_outputs,
    backdoor_inputs,
    backdoor_outputs,
    activation_matching_layers=ACTIVATION_MATCHING_LAYERS,
    backdoor_task_loss_weight=BACKDOOR_TASK_LOSS_WEIGHT,
    normal_activation_change_loss_weight=NORMAL_ACTIVATION_CHANGE_LOSS_WEIGHT,
    obfuscation_loss_weight=OBSCURATION_LOSS_WEIGHT if USE_OBFUSCATION else 0,
    lora_params={},
    model_lr=MODEL_LR,
    n_steps=N_STEPS,
    n_steps_per_logging=N_STEPS_PER_LOGGING,
    batch_size=BATCH_SIZE,
    n_grad_accum=N_GRAD_ACCUM,
    device=DEVICE,
    clip_grad_norm=CLIP_GRAD_NORM,
    model_type = model_type,
    dataset_name = dataset_name,
    backdoor_type=BACKDOOR_TYPE,
    wandb_project="mad-backdoors"
)

lora_model.push_to_hub(f"JordanTensor/llama3-software-engineer-bio-backdoor-model-{run_id}")