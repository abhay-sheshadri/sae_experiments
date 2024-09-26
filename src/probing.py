import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from accelerate import find_executable_batch_size
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from torch import nn
from tqdm.auto import tqdm

from .utils import convert_float16, get_valid_indices, get_valid_token_mask


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
    use_parallelism=True,
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
    only_return_on_tokens_between=None,
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

    if only_return_on_tokens_between is not None:
        for layer in paired_scores:
            for example_idx in range(len(paired_scores[layer])):
                tokens = [token for token, _ in paired_scores[layer][example_idx]]
                valid_indices = get_valid_indices(tokens, only_return_on_tokens_between)
                paired_scores[layer][example_idx] = [
                    (token, score) if i in valid_indices else (token, None)
                    for i, (token, score) in enumerate(
                        paired_scores[layer][example_idx]
                    )
                ]
    return paired_scores


def get_annotated_dataset(
    probes, encoder, dataset, splits, batch_size, max_length, **kwargs
):
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

    new_all_split_scores = {}
    for split, split_scores in all_split_scores.items():
        split_scores = {str(k): v for k, v in split_scores.items()}
        new_split_scores = []

        for example in zip(*(split_scores[str(layer)] for layer in layers)):
            new_example = [
                (
                    token_scores[0][0],
                    aggregation_func([score for _, score in token_scores]),
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

            min_score = min(min_score, min(example_scores))
            max_score = max(max_score, max(example_scores))

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


def create_box_plot(
    aggregated_scores,
    best_threshold,
    best_f1,
    title,
    false_positive_rate,
    allowed_labels,
):
    # Create a box plot of the aggregated scores
    plt.figure(figsize=(12, 6))

    data = [aggregated_scores[label] for label in allowed_labels]
    labels = allowed_labels
    colors = sns.color_palette("husl", n_colors=len(aggregated_scores))

    bp = plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        vert=False,
        flierprops=dict(marker="x", markeredgecolor="black", markersize=5),
        medianprops=dict(color="black"),
        widths=0.5,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best Threshold: {best_threshold:.2f}",
    )

    plt.title(f"{title}\nBest F1 Score: {best_f1:.2f}", fontsize=14)
    plt.xlabel("Score (Vickrey Auction)", fontsize=12)
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
        )
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

    create_box_plot(
        aggregated_scores,
        best_threshold,
        best_f1,
        title,
        false_positive_rate,
        positive_splits + negative_splits + heldout_splits,
    )

    return (
        list(aggregated_scores.values()),
        list(aggregated_scores.keys()),
        best_threshold,
        best_f1,
    )
