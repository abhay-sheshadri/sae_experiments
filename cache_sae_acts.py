import os
import random
import sys

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import utils as tl_utils

from src import *

torch.set_grad_enabled(False)


# Load Model, Tokenizer, and SAE

model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model.tokenizer = tokenizer


sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x-v2", layer=24).cuda()

# Load Dataset

dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    # TODO: Maybe set this to False by default? But RPJ requires it.
    trust_remote_code=True,
)

# Cache all SAE activations if theyre not already saved

cache = FeatureCache(
    model=model,
    tokenizer=tokenizer,
    encoder=sae,
    encoder_layer=12,
)
cache.process_dataset(
    dataset=dataset,
    n_examples=100_000,
    folder_name="cached_sae_acts/",
    seed=42,
    example_seq_len=128,
    batch_size=128,
    text_column="text"
)
