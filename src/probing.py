import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from accelerate import find_executable_batch_size
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm

from .attacks import *
from .utils import (
    convert_float16,
    convert_seconds_to_time_str,
    get_valid_indices,
    get_valid_token_mask,
)


class Probe(nn.Module):
    # Base class for all probes

    def __init__(self):
        super(Probe, self).__init__()

    def forward(self, x):
        assert x.dim() == 3, "Input must be of shape (batch_size, seq_len, d_model)"
        return x


class LinearProbe(Probe):
    # Linear probe for transformer activations

    def __init__(self, d_model):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x):
        x = super().forward(x)
        return self.linear(x).squeeze(-1)


class NonlinearProbe(Probe):
    # Nonlinear probe for transformer activations

    def __init__(self, d_model, d_mlp, dropout=0.1):
        super(NonlinearProbe, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, 1),
        )

    def forward(self, x):
        x = super().forward(x)
        return self.mlp(x).squeeze(-1)


class AttentionProbe(Probe):
    # Attention probe for transformer activations with lower dimensional projection

    def __init__(self, d_model, d_proj, nhead, max_length=8192, sliding_window=None):
        super(AttentionProbe, self).__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_proj * nhead)
        self.k_proj = nn.Linear(d_model, d_proj * nhead)
        self.v_proj = nn.Linear(d_model, d_proj * nhead)
        self.out_proj = nn.Linear(d_proj * nhead, 1)
        if sliding_window is not None:
            mask = self._construct_sliding_window_mask(max_length, sliding_window)
        else:
            mask = self._construct_causal_mask(max_length)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def _construct_causal_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask, diagonal=0)
        return mask.to(dtype=torch.bool)

    def _construct_sliding_window_mask(self, seq_len, window_size):
        q_idx = torch.arange(seq_len).unsqueeze(1)
        kv_idx = torch.arange(seq_len).unsqueeze(0)
        causal_mask = q_idx >= kv_idx
        windowed_mask = q_idx - kv_idx < window_size
        return causal_mask & windowed_mask

    def forward(self, x):
        x = super().forward(x)
        batch_size, seq_len, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.nhead, self.d_proj)
            .transpose(1, 2)
        )
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask[:seq_len, :seq_len]
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        output = self.out_proj(attn_output).squeeze(-1)
        return output


class DirectionalProbe(Probe):
    # Directional probe for transformer activations

    def __init__(self, direction):
        super(DirectionalProbe, self).__init__()
        if direction.dim() == 1:
            direction = direction.unsqueeze(-1)

        # Normalize the direction vector
        direction = direction / torch.norm(direction, dim=0, keepdim=True)
        self.direction = nn.Parameter(direction, requires_grad=False)
        self.magnitude = nn.Parameter(
            torch.tensor(1.0), requires_grad=True
        )  #  We can train this to calibrate the probe
        self.bias = nn.Parameter(
            torch.tensor(0.0), requires_grad=True
        )  #  We can train this to calibrate the probe

    def forward(self, x):
        x = super().forward(x)
        return torch.matmul(x, self.direction * self.magnitude).squeeze(-1) + self.bias


class EncoderProbe(Probe):
    # Wrapper class around an SAE encoder to serve as a harmfulness_probe

    def __init__(self, encoder, feature_ids):
        super(EncoderProbe, self).__init__()
        self.encoder = encoder
        self.feature_ids = nn.Paramter(torch.tensor(feature_ids), requires_grad=False)
        self.linear = nn.Linear(len(feature_ids), 1)

    def forward(self, x):
        x = super().forward(x)

        return self.encoder(x).squeeze(-1)


def initialize_probes_and_optimizers(
    layers, create_probe_fn, lr, device, pretrained_probes=None
):
    # Initialize probes and their corresponding optimizers for each layer
    if pretrained_probes is not None:
        print("Using pretrained probes...")
        probes = pretrained_probes
    else:
        probes = {layer: create_probe_fn() for layer in layers}
    optimizers = {
        layer: torch.optim.AdamW(probe.parameters(), lr=lr)
        for layer, probe in probes.items()
    }
    return probes, optimizers


def train_layer(
    layer,
    probe,
    optimizer,
    pos_activations,
    neg_activations,
    criterion,
    n_epochs,
    batch_size,
    n_grad_accum,
    device,
    using_memmap,
    clip_grad_norm=1.0,
):
    # Train a probe on the activations at a specific layer
    probe.to(device)
    n_examples = min(len(pos_activations), len(neg_activations))
    total_losses = []

    for epoch in tqdm(range(n_epochs)):

        # Shuffle the activations every epoch
        epoch_loss = 0
        shuffle_indices = np.random.permutation(n_examples)
        pos_activations_shuffled = pos_activations[shuffle_indices]
        neg_activations_shuffled = neg_activations[shuffle_indices]

        for i in range(0, n_examples, batch_size):

            # Drop last batch if it is smaller than batch_size
            if i + batch_size > n_examples:
                break

            # Train the probe on the batch of activations
            with torch.autocast(device_type=device):
                probe.train()

                # Load the batch onto the device, and create masks for zero padding
                if not using_memmap:
                    pos_batch = pos_activations_shuffled[i : i + batch_size].to(device)
                    neg_batch = neg_activations_shuffled[i : i + batch_size].to(device)
                else:
                    pos_batch = torch.from_numpy(
                        pos_activations_shuffled[i : i + batch_size]
                    ).to(device)
                    neg_batch = torch.from_numpy(
                        neg_activations_shuffled[i : i + batch_size]
                    ).to(device)
                zero_mask_pos = torch.all(pos_batch == 0, dim=-1).view(-1).to(device)
                zero_mask_neg = torch.all(neg_batch == 0, dim=-1).view(-1).to(device)

                # Forward pass through the probe, and compute the loss
                pos_outputs = probe(pos_batch).view(-1)
                neg_outputs = probe(neg_batch).view(-1)
                pos_targets = torch.ones_like(pos_outputs, device=device)
                neg_targets = torch.zeros_like(neg_outputs, device=device)
                loss_pos = criterion(
                    pos_outputs[~zero_mask_pos], pos_targets[~zero_mask_pos]
                )
                loss_neg = criterion(
                    neg_outputs[~zero_mask_neg], neg_targets[~zero_mask_neg]
                )
                loss = (loss_pos + loss_neg) / n_grad_accum

            # Backward pass and optimization step
            loss.backward()
            epoch_loss += loss.item() * n_grad_accum

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_grad_norm)

            if (i // batch_size + 1) % n_grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Perform an extra optimization step if the number of examples is not divisible by batch_size
        if (n_examples // batch_size) % n_grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
        total_losses.append(epoch_loss)

    probe.to("cpu")
    return layer, probe, total_losses


def cache_activations(encoder, examples, batch_size, max_length, cache_dir, **kwargs):
    # Cache activations for a set of examples using the encoder
    initial_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"  # Use right padding
    activations = encoder.get_model_residual_acts(
        examples,
        batch_size=batch_size,
        max_length=max_length,
        use_memmap=cache_dir,
        **kwargs,
    )
    encoder.tokenizer.padding_side = initial_padding_side
    return activations


def train_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    use_parallelism=False,
    lr=1e-3,
    max_length=1024,
    n_epochs=10,
    batch_size=16,
    n_grad_accum=1,
    device="cuda",
    cache_activations_save_path=None,
    pretrained_probes=None,
    **kwargs,
):
    # Main function to train probes for all specified layers

    # Check if the cache file exists and a save path is provided
    if cache_activations_save_path is not None and os.path.exists(
        cache_activations_save_path
    ):
        print(f"Loading cached activations from {cache_activations_save_path}")

        positive_metadata_file = os.path.join(
            cache_activations_save_path, "positive_examples_metadata.json"
        )
        negative_metadata_file = os.path.join(
            cache_activations_save_path, "negative_examples_metadata.json"
        )

        # Load the memmaps for the positive examples
        positive_activations = []
        with open(positive_metadata_file, "r") as f:
            positive_metadata = json.load(f)
            for layer in range(positive_metadata["num_layers"]):
                pos_file = os.path.join(
                    cache_activations_save_path,
                    f"positive_examples_residual_act_layer_{layer}.dat",
                )
                pos_memmap = np.memmap(
                    pos_file,
                    dtype=positive_metadata["dtype"],
                    mode="r",
                    shape=tuple(positive_metadata["shape"]),
                )
                positive_activations.append(pos_memmap)

        # Load the memmaps for the negative examples
        negative_activations = []
        with open(negative_metadata_file, "r") as f:
            negative_metadata = json.load(f)
            for layer in range(negative_metadata["num_layers"]):
                neg_file = os.path.join(
                    cache_activations_save_path,
                    f"negative_examples_residual_act_layer_{layer}.dat",
                )
                neg_memmap = np.memmap(
                    neg_file,
                    dtype=negative_metadata["dtype"],
                    mode="r",
                    shape=tuple(negative_metadata["shape"]),
                )
                negative_activations.append(neg_memmap)

    else:
        # Cache activations for the positive and negative examples
        print("Caching activations...")

        # Cache activations for the positive and negative examples, without memmaps
        if cache_activations_save_path is None:
            positive_activations = cache_activations(
                encoder, positive_examples, batch_size, max_length, **kwargs
            )
            negative_activations = cache_activations(
                encoder,
                negative_examples,
                batch_size,
                max_length,
                **kwargs,
            )

        # Cache activations for the positive and negative examples, with memmaps
        else:
            positive_path = os.path.join(
                cache_activations_save_path, "positive_examples"
            )
            negative_path = os.path.join(
                cache_activations_save_path, "negative_examples"
            )
            positive_activations = cache_activations(
                encoder,
                positive_examples,
                batch_size,
                max_length,
                cache_dir=positive_path,
                **kwargs,
            )
            negative_activations = cache_activations(
                encoder,
                negative_examples,
                batch_size,
                max_length,
                cache_dir=negative_path,
                **kwargs,
            )

    # Move model to CPU and clear GPU memory, to save VRAM for probe training
    encoder.model.to("cpu")
    torch.cuda.empty_cache()

    # Initialize probes and optimizers for each layer, and loss criterion
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, lr, device, pretrained_probes
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train probes for all specified layers
    print("Training probes...")
    if use_parallelism:
        # Use multiprocessing to train probes in parallel
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=len(layers)) as pool:
            results = pool.starmap(
                train_layer,
                [
                    (
                        layer,
                        probes[layer],
                        optimizers[layer],
                        positive_activations[layer],
                        negative_activations[layer],
                        criterion,
                        n_epochs,
                        batch_size,
                        n_grad_accum,
                        device,
                        cache_activations_save_path is not None,
                    )
                    for layer in layers
                ],
            )
    else:
        # Train probes sequentially
        results = [
            train_layer(
                layer,
                probes[layer],
                optimizers[layer],
                positive_activations[layer],
                negative_activations[layer],
                criterion,
                n_epochs,
                batch_size,
                n_grad_accum,
                device,
                cache_activations_save_path is not None,
            )
            for layer in layers
        ]

    # Print final loss for each layer and return the trained probes
    for layer, probe, losses in results:
        probes[layer] = probe
        print(f"Layer {layer} - Final Loss: {losses[-1]:.4f}")

    # Move model back to GPU and return probes
    encoder.model.to("cuda")
    return probes


def train_linear_probe(encoder, positive_examples, negative_examples, layers, **kwargs):
    # Train a linear probe for each specified layer
    def create_linear_probe():
        return LinearProbe(encoder.model.config.hidden_size)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_linear_probe,
        layers,
        **kwargs,
    )


def train_nonlinear_probe(
    encoder, positive_examples, negative_examples, d_mlp, layers, **kwargs
):
    # Train a nonlinear probe for each specified layer
    def create_nonlinear_probe():
        return NonlinearProbe(encoder.model.config.hidden_size, d_mlp)

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_nonlinear_probe,
        layers,
        **kwargs,
    )


def train_attention_probe(
    encoder,
    positive_examples,
    negative_examples,
    d_proj,
    nhead,
    sliding_window,
    layers,
    **kwargs,
):
    # Train an attention probe for each specified layer
    def create_attention_probe():
        return AttentionProbe(
            encoder.model.config.hidden_size,
            d_proj,
            nhead,
            sliding_window=sliding_window,
        )

    return train_probe(
        encoder,
        positive_examples,
        negative_examples,
        create_attention_probe,
        layers,
        **kwargs,
    )


def initialize_lora_adapter(encoder, layers, lora_params):
    # Disable gradient computation for the encoder.model
    for param in encoder.model.parameters():
        param.requires_grad = False

    # Unpack LoRA parameters
    r = lora_params.get("r", 16)
    alpha = lora_params.get("alpha", 16)
    dropout = lora_params.get("dropout", 0.05)
    bias = lora_params.get("bias", "none")
    target_modules = lora_params.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    # Define LoRA Configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
        layers_to_transform=list(range(0, max(layers) + 1)),
        task_type="CAUSAL_LM",
    )

    # Apply LoRA adapter to the model
    lora_model = get_peft_model(encoder.model, lora_config)

    return lora_model


def train_online_probe(
    encoder,
    positive_examples,
    negative_examples,
    create_probe_fn,
    layers,
    lora_params={},
    adversarial_training=False,
    probe_lr=1e-3,
    adapter_lr=5e-5,
    kl_penalty=0.1,
    max_length=1024,
    n_steps=1000,
    n_steps_per_logging=100,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    pretrained_probes=None,
    only_return_on_tokens_between=None,
    epsilon=6.0,
    adversary_lr=5e-4,
    pgd_iterations=32,
    clip_grad_norm=1.0,
    **kwargs,
):
    assert n_grad_accum == 0 or n_steps % n_grad_accum == 0

    # Initialize probes and optimizers for each layer
    probes, optimizers = initialize_probes_and_optimizers(
        layers, create_probe_fn, probe_lr, device, pretrained_probes
    )
    probes = {layer: probe.to(device) for layer, probe in probes.items()}

    # Initialize LoRA adapter
    lora_model = initialize_lora_adapter(encoder, layers, lora_params)
    adapter_optimizer = torch.optim.AdamW(lora_model.parameters(), lr=adapter_lr)

    # Loss criterion
    criterion = nn.BCEWithLogitsLoss()

    # Tokenize and prepare input data
    positive_tokens = encoder.tokenizer(
        positive_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    positive_input_ids = positive_tokens["input_ids"]
    positive_attention_mask = positive_tokens["attention_mask"]
    negative_tokens = encoder.tokenizer(
        negative_examples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    negative_input_ids = negative_tokens["input_ids"]
    negative_attention_mask = negative_tokens["attention_mask"]

    if only_return_on_tokens_between is not None:
        pos_only_return_mask = get_valid_token_mask(
            positive_input_ids, only_return_on_tokens_between
        )
        zero_positive_mask = positive_attention_mask.clone()
        zero_positive_mask[~pos_only_return_mask] = 0

        neg_only_return_mask = get_valid_token_mask(
            negative_input_ids, only_return_on_tokens_between
        )
        zero_negative_mask = negative_attention_mask.clone()
        zero_negative_mask[~neg_only_return_mask] = 0
    else:
        zero_positive_mask = positive_attention_mask
        zero_negative_mask = negative_attention_mask

    n_examples = min(len(positive_examples), len(negative_examples))

    continue_training_next_epoch = True
    current_step = 0
    start_time = time.time()

    accumulated_toward_pgd_loss = 0
    accumulated_probe_pgd_loss = 0
    accumulated_probe_loss = 0
    accumulated_kl_loss = 0
    steps_since_last_log = 0

    while continue_training_next_epoch:

        # Shuffle the examples
        perm = torch.randperm(n_examples)

        for i in range(0, n_examples, batch_size):
            # Check if the batch is the last one
            if i + batch_size > n_examples:
                break

            # Get the batch
            batch_perm = perm[i : i + batch_size]
            pos_batch_input_ids = positive_input_ids[batch_perm].to(device)
            pos_batch_attention_mask = positive_attention_mask[batch_perm].to(device)
            neg_batch_input_ids = negative_input_ids[batch_perm].to(device)
            neg_batch_attention_mask = negative_attention_mask[batch_perm].to(device)
            pos_batch_zero_mask = zero_positive_mask[batch_perm].to(device).bool()
            neg_batch_zero_mask = zero_negative_mask[batch_perm].to(device).bool()

            # Forward pass on positive examples
            with torch.autocast(device_type=device):

                # Clear hooks from last step if adversarial training is used
                if adversarial_training:
                    clear_hooks(lora_model)

                if adversarial_training:
                    losses, wrappers = train_attack(
                        adv_tokens=pos_batch_input_ids,
                        prompt_mask=~pos_batch_zero_mask,
                        target_mask=pos_batch_zero_mask,
                        model=lora_model,
                        model_layers_module="base_model.model.model.layers",
                        layer=["embedding"],
                        epsilon=epsilon,
                        learning_rate=adversary_lr,
                        pgd_iterations=pgd_iterations,
                        probes=probes,
                        adversary_type="pgd",
                    )
                    pgd_toward_loss = losses["toward"]
                    pgd_probe_loss = losses["probe"]
                else:
                    pgd_toward_loss = (
                        0  # Set to 0 when adversarial training is not used
                    )
                    pgd_probe_loss = 0
                    wrappers = []

                for wrapper in wrappers:
                    wrapper.enabled = True

                pos_output = lora_model(
                    input_ids=pos_batch_input_ids,
                    attention_mask=pos_batch_attention_mask,
                    output_hidden_states=True,
                )
                pos_acts = {
                    layer: pos_output.hidden_states[layer + 1] for layer in layers
                }

            # Compute the positive probe losses
            pos_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device):
                    pos_probe_output = probe(pos_acts[layer]).view(-1)
                    pos_targets = torch.ones_like(pos_probe_output)
                    pos_layer_loss = criterion(
                        pos_probe_output[pos_batch_zero_mask.view(-1)],
                        pos_targets[pos_batch_zero_mask.view(-1)],
                    )
                    pos_loss += pos_layer_loss

            for wrapper in wrappers:
                wrapper.enabled = False

            # Forward pass on negative examples
            with torch.autocast(device_type=device):
                neg_output = lora_model(
                    input_ids=neg_batch_input_ids,
                    attention_mask=neg_batch_attention_mask,
                    output_hidden_states=True,
                )
                neg_logits = neg_output.logits
                neg_acts = {
                    layer: neg_output.hidden_states[layer + 1] for layer in layers
                }

            # Compute the negative probe losses
            neg_loss = 0
            for layer, probe in probes.items():
                with torch.autocast(device_type=device):
                    neg_probe_output = probe(neg_acts[layer]).view(-1)
                    neg_targets = torch.zeros_like(neg_probe_output)
                    neg_layer_loss = criterion(
                        neg_probe_output[neg_batch_zero_mask.view(-1).bool()],
                        neg_targets[neg_batch_zero_mask.view(-1).bool()],
                    )
                    neg_loss += neg_layer_loss

            # Compute KL divergence of logits from base model logits
            with torch.no_grad():
                base_neg_output = encoder.model(
                    input_ids=neg_batch_input_ids,
                    attention_mask=neg_batch_attention_mask,
                )

            kl_loss = F.kl_div(
                F.log_softmax(base_neg_output.logits, dim=-1),
                F.softmax(neg_logits, dim=-1),
                reduction="batchmean",
            )

            # Backward pass
            pos_loss.backward(retain_graph=True)
            neg_loss.backward(retain_graph=True)
            (kl_loss / (kl_loss.detach() + 1e-8) * kl_penalty).backward()

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), clip_grad_norm)
                all_probe_params = [
                    param for probe in probes.values() for param in probe.parameters()
                ]
                torch.nn.utils.clip_grad_norm_(all_probe_params, clip_grad_norm)

            # Accumulate losses
            accumulated_probe_loss += pos_loss.item() + neg_loss.item()
            accumulated_kl_loss += kl_loss.item()
            accumulated_toward_pgd_loss += (
                pgd_toward_loss if adversarial_training else 0
            )
            accumulated_probe_pgd_loss += pgd_probe_loss if adversarial_training else 0
            steps_since_last_log += 1

            # Perform optimization step after accumulating gradients
            if (i // batch_size + 1) % n_grad_accum == 0 or (
                i + batch_size
            ) >= n_examples:
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                adapter_optimizer.step()
                adapter_optimizer.zero_grad()

            current_step += 1

            if current_step % n_steps_per_logging == 0:
                avg_probe_loss = accumulated_probe_loss / steps_since_last_log
                avg_kl_loss = accumulated_kl_loss / steps_since_last_log
                avg_toward_pgd_loss = (
                    accumulated_toward_pgd_loss / steps_since_last_log
                    if adversarial_training
                    else 0
                )
                avg_probe_pgd_loss = (
                    accumulated_probe_pgd_loss / steps_since_last_log
                    if adversarial_training
                    else 0
                )
                avg_total_loss = avg_probe_loss + avg_kl_loss

                log_message = (
                    f"Step: {current_step}/{n_steps}, "
                    f"Time: {convert_seconds_to_time_str(time.time() - start_time)}, "
                    f"Avg Total Loss: {avg_total_loss:.4f}, "
                    f"Avg Probe Loss: {avg_probe_loss:.4f}, "
                    f"Avg KL Loss: {avg_kl_loss:.4f}"
                )

                if adversarial_training:
                    log_message += f", Avg Toward PGD Loss: {avg_toward_pgd_loss:.4f}"
                    log_message += f", Avg Probe PGD Loss: {avg_probe_pgd_loss:.4f}"

                print(log_message)

                # Reset accumulators
                accumulated_probe_loss = 0
                accumulated_kl_loss = 0
                accumulated_pgd_loss = 0
                steps_since_last_log = 0

            if current_step >= n_steps:
                continue_training_next_epoch = False
                break

    return probes, lora_model


def save_probes(probes, save_path):
    # Save a list of probes to a file
    torch.save(probes, save_path)


def load_probes(load_path):
    # Load a list of probes from a file
    return torch.load(load_path)


def get_probe_scores(
    probes,
    encoder,
    examples,
    batch_size,
    max_length,
    device="cuda",
    probe_layers=None,
):
    # If probe_layers is not defined, set it to all the layers
    if probe_layers is None:
        probe_layers = list(probes.keys())

    # Cache activations for a set of examples using the encoder
    initial_padding_side = encoder.tokenizer.padding_side
    encoder.tokenizer.padding_side = "right"  # Use right padding

    @find_executable_batch_size(starting_batch_size=batch_size)
    def get_activations(batch_size):
        return encoder.get_model_residual_acts(
            examples,
            batch_size=batch_size,
            max_length=max_length,
            return_tokens=True,
            only_return_layers=probe_layers,
        )

    activations, tokens = get_activations()
    encoder.tokenizer.padding_side = initial_padding_side

    # Get probe scores for a set of examples
    probe_scores = {}

    @find_executable_batch_size(starting_batch_size=batch_size)
    def get_probe_scores_batch_size(batch_size):
        for layer in probe_layers:
            probe = probes[layer]
            probe.to(device)
            probe.eval()  # Set the probe to evaluation mode

            layer_activations = activations[layer]
            n_examples = len(layer_activations)
            layer_scores = []

            with torch.no_grad():  # Disable gradient computation for inference
                for i in range(0, n_examples, batch_size):
                    batch = layer_activations[i : i + batch_size].to(device)
                    with torch.autocast(device_type=device):
                        batch_scores = probe(batch)
                        batch_scores = (
                            torch.sigmoid(batch_scores).detach().cpu().numpy() * 2 - 1
                        ) * 3
                    layer_scores.append(batch_scores)

            probe_scores[layer] = np.concatenate(layer_scores)
            probe.to("cpu")  # Move the probe back to CPU to free up GPU memory
        return probe_scores

    probe_scores = get_probe_scores_batch_size()
    activations.clear()

    # Get the (token, score) pairs for each example
    paired_scores = {}
    for layer, scores in probe_scores.items():
        paired_scores[layer] = [
            [
                (
                    encoder.tokenizer.decode(
                        tokens["input_ids"][example_idx][token_idx].item()
                    ),
                    scores[example_idx][token_idx],
                )
                for token_idx in range(tokens["input_ids"].shape[1])
                if tokens["attention_mask"][example_idx][
                    token_idx
                ].item()  # Skip padding tokens
            ]
            for example_idx in range(tokens["input_ids"].shape[0])
        ]

    return paired_scores


def remove_scores_between_tokens(
    paired_scores_all_splits, only_return_on_tokens_between
):
    for paired_scores in paired_scores_all_splits.values():
        first_layer = next(iter(paired_scores))

        for example_idx, example_data in enumerate(paired_scores[first_layer]):
            tokens = [token for token, _ in example_data]
            valid_indices = set(
                get_valid_indices(tokens, only_return_on_tokens_between)
            )

            for layer_data in paired_scores.values():
                layer_data[example_idx] = [
                    (token, score if i in valid_indices else None)
                    for i, (token, score) in enumerate(layer_data[example_idx])
                ]

    return paired_scores_all_splits


def get_annotated_dataset(
    probes,
    encoder,
    dataset,
    splits,
    batch_size,
    max_length,
    model_adapter_path=None,
    **kwargs,
):
    # Load model adapter if provided
    if model_adapter_path is not None:
        print("Loading model adapter...")
        assert not isinstance(
            encoder.model, PeftModel
        )  # model should not be a PeftModel at this point
        encoder.model = PeftModel.from_pretrained(encoder.model, model_adapter_path)
        # encoder.model = encoder.model.merge_and_unload()

    # Get scores
    scores_dict = {}
    dataset_splits = {
        split: dataset[split].select(range(min(1000, len(dataset[split]))))
        for split in splits
    }
    for split in splits:
        print(split)

        split_dataset = dataset_splits[split]
        split_dataset_str = [
            split_dataset[i]["prompt"] + split_dataset[i]["completion"]
            for i in range(len(split_dataset))
        ]

        with torch.no_grad():
            paired_scores = get_probe_scores(
                probes=probes,
                encoder=encoder,
                examples=split_dataset_str,
                batch_size=batch_size,
                max_length=max_length,
                **kwargs,
            )
        scores_dict[split] = paired_scores

    if model_adapter_path is not None:
        # remove the lora adapter
        encoder.model = encoder.model.base_model

    return convert_float16(scores_dict)


def vickrey_auc(scores, k):
    # Compute the Vickrey AUC for a list of scores
    # Returns the k-th highest score
    if not scores:
        raise ValueError("Scores list must be non-empty")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    k = min(k, len(scores))
    return sorted(scores, reverse=True)[k - 1]


def aggregate_across_layers(all_split_scores, layers, cross_layer_aggregation):
    # Given the probe scores foor multiple layers, compute a single score for each token
    aggregation_funcs = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "vickrey": lambda x: vickrey_auc(x, 2),
    }

    if cross_layer_aggregation not in aggregation_funcs:
        raise ValueError(f"Invalid cross_layer_aggregation: {cross_layer_aggregation}")

    aggregation_func = aggregation_funcs[cross_layer_aggregation]

    def wrapped_aggregation_func(layer_scores):
        if None in layer_scores:
            return None
        return aggregation_func(layer_scores)

    new_all_split_scores = {}
    for split, split_scores in all_split_scores.items():
        split_scores = {str(k): v for k, v in split_scores.items()}
        new_split_scores = []

        for example in zip(*(split_scores[str(layer)] for layer in layers)):
            new_example = [
                (
                    token_scores[0][0],
                    wrapped_aggregation_func([score for _, score in token_scores]),
                )
                for token_scores in zip(*example)
            ]
            new_split_scores.append(new_example)

        new_all_split_scores[split] = new_split_scores

    return new_all_split_scores


def aggregate_across_tokens(all_split_scores, cross_token_aggregation):
    # Given the probe scores for each token, compute a single score for each example
    aggregation_funcs = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "vickrey": lambda x: vickrey_auc(x, 2),
        "logsumexp": lambda x: np.log(np.exp(x).sum()),
        "top_2_percent": lambda x: vickrey_auc(x, int(0.02 * len(x)) + 1),
        "median_over_zero": lambda x: np.median([score for score in x if score > 0]),
    }

    if cross_token_aggregation not in aggregation_funcs:
        raise ValueError(f"Invalid cross_token_aggregation: {cross_token_aggregation}")

    aggregation_func = aggregation_funcs[cross_token_aggregation]

    aggregated_scores = {}
    min_score, max_score = float("inf"), float("-inf")

    for split, split_scores in all_split_scores.items():
        new_split_scores = []
        for example in split_scores:
            example_scores = [score for _, score in example if score is not None]
            example_scalar = aggregation_func(example_scores)
            new_split_scores.append(example_scalar)

            min_score = min(min_score, example_scalar)
            max_score = max(max_score, example_scalar)

        aggregated_scores[split] = new_split_scores

    return aggregated_scores, min_score, max_score


def compute_f1(threshold, aggregated_scores, negative_splits, positive_splits):
    # Compute the F1 score for a given threshold
    y_true, y_pred = [], []
    for split in negative_splits + positive_splits:
        scores = aggregated_scores[split]
        y_true.extend([0 if split in negative_splits else 1] * len(scores))
        y_pred.extend([int(score >= threshold) for score in scores])
    return f1_score(y_true, y_pred)


def find_best_threshold(
    aggregated_scores, negative_splits, positive_splits, min_score, max_score
):
    # Find the best threshold for the aggregated scores
    thresholds = np.linspace(min_score, max_score, 1000)
    f1_scores = [
        compute_f1(t, aggregated_scores, negative_splits, positive_splits)
        for t in thresholds
    ]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def get_threshold_at_fpr(
    aggregated_scores, negative_splits, min_score, max_score, target_fpr
):
    # Compute the minimum threshold that achieves a given false positive rate
    thresholds = np.linspace(min_score, max_score, 1000)

    final_threshold = max_score
    for threshold in reversed(thresholds):
        # Calculate false positive rate at this threshold
        fp = sum(
            score >= threshold
            for split in negative_splits
            for score in aggregated_scores[split]
        )
        tn = sum(
            score < threshold
            for split in negative_splits
            for score in aggregated_scores[split]
        )
        fpr = fp / (fp + tn)

        # If we've reached or exceeded the target FPR, return this threshold
        if fpr <= target_fpr:
            final_threshold = threshold

    return final_threshold


def create_scores_plot(
    aggregated_scores,
    best_threshold,
    best_f1,
    title,
    false_positive_rate,
    allowed_labels,
):
    plt.figure(figsize=(12, 6))

    data = [aggregated_scores[label] for label in allowed_labels]
    labels = allowed_labels
    colors = sns.color_palette("husl", n_colors=len(aggregated_scores))

    # Create violin plot
    parts = plt.violinplot(
        data, vert=False, showmeans=False, showextrema=False, showmedians=False
    )

    # Customize violin plot colors and add quartile lines
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

        # Calculate quartiles
        quartile1, median, quartile3 = np.percentile(data[i], [25, 50, 75])

        # Add quartile lines
        plt.hlines(
            i + 1, quartile1, quartile3, color="k", linestyle="-", lw=5, alpha=0.7
        )
        plt.vlines(median, i + 0.95, i + 1.05, color="white", linestyle="-", lw=2)

    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best Threshold: {best_threshold:.2f}",
    )

    plt.title(f"{title}\nOverall F1: {best_f1:.2f}", fontsize=14)
    plt.xlabel("Aggregated Score", fontsize=12)
    plt.yticks(range(1, len(labels) + 1), labels, fontsize=10)

    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold at {false_positive_rate*100:.2f}% FPR",
        ),
        plt.Line2D([0], [0], color="k", linestyle="-", lw=5, alpha=0.7, label="IQR"),
        plt.Line2D([0], [0], color="white", linestyle="-", lw=2, label="Median"),
    ]
    legend_elements.extend(
        [
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7, label=label)
            for label, color in zip(labels, colors)
        ]
    )

    plt.legend(
        handles=legend_elements,
        fontsize=8,
        title="Categories",
        title_fontsize=10,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()
    plt.show()


def get_per_split_scores(
    aggregated_scores, best_threshold, positive_splits, negative_splits, heldout_splits
):
    per_split_scores = {}

    for split in positive_splits + negative_splits + heldout_splits:
        if split not in aggregated_scores:
            print(f"Warning: Split '{split}' not found in aggregated_scores.")
            continue

        scores = aggregated_scores[split]
        total_samples = len(scores)

        if split in positive_splits:
            correct_predictions = sum(score >= best_threshold for score in scores)
        else:  # negative_splits and heldout_splits
            correct_predictions = sum(score < best_threshold for score in scores)

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        per_split_scores[split] = accuracy

    return per_split_scores


def generate_score_plots(
    all_split_scores,
    positive_splits,
    negative_splits,
    heldout_splits,
    layers,
    cross_token_aggregation,
    cross_layer_aggregation=None,
    false_positive_rate=0.05,
    title="",
):
    if cross_layer_aggregation:
        all_split_scores = aggregate_across_layers(
            all_split_scores, layers, cross_layer_aggregation
        )

    aggregated_scores, min_score, max_score = aggregate_across_tokens(
        all_split_scores, cross_token_aggregation
    )

    # best_threshold, best_f1 = find_best_threshold(
    #    aggregated_scores, negative_splits, positive_splits, min_score, max_score
    # )
    best_threshold = get_threshold_at_fpr(
        aggregated_scores, heldout_splits, min_score, max_score, false_positive_rate
    )
    best_f1 = compute_f1(
        best_threshold, aggregated_scores, negative_splits, positive_splits
    )

    create_scores_plot(
        aggregated_scores,
        best_threshold,
        best_f1,
        title,
        false_positive_rate,
        positive_splits + negative_splits + heldout_splits,
    )

    per_split_scores = get_per_split_scores(
        aggregated_scores,
        best_threshold,
        positive_splits,
        negative_splits,
        heldout_splits,
    )

    return (
        list(aggregated_scores.values()),
        list(aggregated_scores.keys()),
        best_threshold,
        best_f1,
        per_split_scores,
    )
