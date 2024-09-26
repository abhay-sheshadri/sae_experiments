import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset

from .utils import *


def get_encoder_reconstruction_vector(encoder, vector):
    with torch.autocast(device_type="cuda"):

        # Scale and unsqueeze the input vector
        scaled_vector = vector.unsqueeze(0)

        # Reconstruct the vector
        reconstructed_vector = encoder.reconstruct(scaled_vector)[0]

        # Encode the scaled vector
        top_features, top_values = encoder.encode(scaled_vector)

        # Ensure we're working with 1D tensors
        top_features = top_features.squeeze()
        top_values = top_values.squeeze()

        # Sort the values in descending order and get the corresponding indices
        sorted_values, indices = torch.sort(top_values, descending=True)

        # Use the indices to sort the features
        sorted_features = top_features[indices]
        return sorted_features, sorted_values, reconstructed_vector


def get_steering_vector(features, labels, method="mean_diff", normalized=True):
    assert method in ["logistic", "mean_diff", "rep_e", "random"], "Invalid method"

    # Convert features to Float32 if they're not already
    if features.dtype != torch.float32:
        features = features.to(torch.float32)

    # Get the mean diff vector, so we can correctly align the other vectors
    # Separate the data into two groups based on labels
    ones = features[labels == 1]
    zeros = features[labels == 0]

    # Compute the average vectors
    avg_ones = ones.mean(dim=0) if ones.numel() > 0 else torch.zeros(features.size(1))
    avg_zeros = (
        zeros.mean(dim=0) if zeros.numel() > 0 else torch.zeros(features.size(1))
    )

    # Compute and return the difference
    mean_diff = avg_ones - avg_zeros

    # Run method to get steering vector
    if method == "mean_diff":

        # We already computed mean_diff
        steering_vector = mean_diff
    elif method == "logistic":

        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Balance the dataset
        majority_class = 0 if (labels_np == 0).sum() > (labels_np == 1).sum() else 1
        minority_class = 1 - majority_class
        features_majority = features_np[labels_np == majority_class]
        features_minority = features_np[labels_np == minority_class]
        labels_majority = labels_np[labels_np == majority_class]
        labels_minority = labels_np[labels_np == minority_class]
        features_minority_upsampled, labels_minority_upsampled = resample(
            features_minority,
            labels_minority,
            replace=True,
            n_samples=len(features_majority),
            random_state=42,
        )
        features_balanced = np.vstack((features_majority, features_minority_upsampled))
        labels_balanced = np.hstack((labels_majority, labels_minority_upsampled))

        # Use sklearn's LogisticRegression
        lr = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr.fit(features_balanced, labels_balanced)

        # Convert the coefficients back to a PyTorch tensor
        steering_vector = (
            torch.from_numpy(lr.coef_[0]).to(features.device).to(features.dtype)
        )

        # Make sure it points in the same direction as the mean_diff
        steering_vector = steering_vector * torch.sign(
            torch.dot(steering_vector, mean_diff)
        )
    elif method == "rep_e":

        # Representation Engineering method
        pos_examples = features[labels == 1]
        neg_examples = features[labels == 0]

        # Generate all pairs of positive and negative examples
        pairs = list(itertools.product(pos_examples, neg_examples))

        # Compute differences for all pairs
        differences = torch.stack([pos - neg for pos, neg in pairs])

        # Sample a random subset of differences
        n_samples = min(30_000, differences.size(0))
        differences = differences[torch.randperm(differences.size(0))][:n_samples]

        # Perform PCA using SVD and extract the first principal component
        svd = TruncatedSVD(n_components=1, random_state=42)
        svd.fit(differences.cpu().numpy())
        steering_vector = (
            torch.from_numpy(svd.components_[0]).to(features.device).to(features.dtype)
        )

        # Make sure it points in the same direction as the mean_diff
        steering_vector = steering_vector * torch.sign(
            torch.dot(steering_vector, mean_diff)
        )
    elif method == "random":

        # Useful for random steering vector baseline
        steering_vector = torch.randn(features.shape[1], dtype=features.dtype)

    # Return unit steering vector
    if normalized:
        return normalize_last_dim(steering_vector)
    else:
        return steering_vector


def compute_steering_vector(
    encoder,
    positive_examples,
    negative_examples,
    layers,
    position_aggregation="last",
    method="mean_diff",
    reconstruct_attack=False,
    batch_size=32,
    layer_hook_name_format="model.layers.{layer}",
):
    # Make sure layers is a list
    single_layer = isinstance(layers, int)
    if single_layer:
        layers = [layers]

    # Get the activations for the positive and negative examples
    positive_acts, postive_tokens = encoder.get_model_residual_acts(
        positive_examples,
        batch_size=batch_size,
        return_tokens=True,
        only_return_layers=layers,
    )
    negative_acts, negative_tokens = encoder.get_model_residual_acts(
        negative_examples,
        batch_size=batch_size,
        return_tokens=True,
        only_return_layers=layers,
    )

    # Get the steering vector for each layer
    result = {}
    for layer in layers:

        # Get the steering vector for the data
        train_input_acts, train_labels = get_labeled(
            negative_acts[layer],
            positive_acts[layer],
            aggregation=position_aggregation,
            acts1_tokens=negative_tokens["input_ids"],
            acts2_tokens=postive_tokens["input_ids"],
        )
        vector = get_steering_vector(
            train_input_acts, train_labels, method=method, normalized=False
        ).cuda()

        # Reconstruct with SAE if reconstruct_attack is true
        if reconstruct_attack:
            assert single_layer, "Reconstruction is only supported for single layers"
            sorted_features, sorted_values, vector = get_encoder_reconstruction_vector(
                encoder, vector
            )

        # Create a wrapper function to easily get the output of the model
        def wrapper_fn(
            prompt, scale=1.0, _layer=layer, _vector=vector, **generation_kwargs
        ):
            # Tokenize the input
            tokens = encoder.tokenizer(prompt, add_special_tokens=False)["input_ids"]

            # Create the input tensor
            input_ids = torch.tensor([tokens]).to(encoder.model.device)

            # Create the attention mask
            attention_mask = torch.ones_like(input_ids)

            # Define the hook function
            def universal_steering_hook(output):
                # Apply steering to the last token of the prompt and all subsequent tokens
                output[:, -1:] = output[:, -1:] + _vector.to(output.dtype) * scale
                return output

            return generate_with_interventions(
                model=encoder.model,
                input_ids=input_ids,
                hook_interventions={
                    layer_hook_name_format.format(layer=_layer): universal_steering_hook
                },
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        result[layer] = {"steering_vector": vector, "wrapper_fn": wrapper_fn}
    return result


### OLD ADVEXES CODE ###
### USE ATTACKS.PY INSTEAD ###


def run_soft_prompt_opt(
    encoder,
    input_ids,
    attention_mask,
    soft_prompt_mask,
    target_mask,
    hook_name,
    n_dim=2,
    n_epochs=10,
    magnitude=8,
    batch_size=32,
    lr=5e-2,
    reconstruct_attack=False,
    sae_l1_penalty=0.0001,
    attack_init=None,
    probes={},
):
    # Make sure magnitude has a value
    if magnitude is None:
        assert (
            attack_init is not None
        ), "If magnitude is None, attack_init must be passed"
        magnitude = attack_init.norm(dim=-1).mean().item()

    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, soft_prompt_mask, target_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize attack
    if attack_init:

        # If initial attack is passed it, create attack parameter from it
        if isinstance(attack_init, torch.nn.Parameter):
            attack = attack_init.data
        elif isinstance(attack_init, torch.Tensor):
            attack = attack_init
        else:
            raise Exception("Invalid attack_init type")
        assert len(attack.shape) in [1, 2], "The shape of the attack is not 1 or 2"
    else:

        # If attack initialization is not passed in, sample from normal distribution
        if n_dim == 1:
            attack = torch.randn(encoder.model.config.hidden_size)
        elif n_dim == 2:
            soft_prompt_length = soft_prompt_mask.sum(-1)[0]
            assert torch.all(
                soft_prompt_mask.sum(-1) == soft_prompt_length
            ), "Soft prompt masks have unequal number of True values across examples"
            attack = torch.randn(soft_prompt_length, encoder.model.config.hidden_size)
        else:
            raise Exception("Invalid n_dim")
        attack = normalize_last_dim(attack) * magnitude
    attack = torch.nn.Parameter(attack.float().to(encoder.model.device))
    attack.requires_grad = True

    # Create function to create hooks
    temp_hook_output = None

    def create_hook(batch_soft_prompt_mask):
        def hook(output):
            nonlocal temp_hook_output  # Make sure the variable is in scope

            # Assuming output shape is (batch_size, sequence_length, hidden_size)
            if reconstruct_attack:
                features, values, modified_attack = get_encoder_reconstruction_vector(
                    encoder, attack
                )
                temp_hook_output = values
            else:
                modified_attack = attack

            # Depending on the shape, we need to modify the output separately
            if len(modified_attack.shape) == 1:
                output[batch_soft_prompt_mask] = output[
                    batch_soft_prompt_mask, :
                ] + modified_attack.to(output.dtype)
            elif len(modified_attack.shape) == 2:
                broadcasted_attack = modified_attack.unsqueeze(0).expand(
                    output.size(0), -1, -1
                )
                output[batch_soft_prompt_mask] = output[
                    batch_soft_prompt_mask, :
                ] + broadcasted_attack.flatten(0, 1).to(output.dtype)
            return output

        return hook

    # Train attack
    optimizer = torch.optim.AdamW(
        [
            attack,
        ],
        lr=lr,
    )
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        total_loss = 0
        for (
            batch_input_ids,
            batch_attention_mask,
            batch_soft_prompt_mask,
            batch_target_mask,
        ) in dataloader:

            # Move data onto model device
            batch_input_ids = batch_input_ids.to(encoder.model.device)
            batch_attention_mask = batch_attention_mask.to(encoder.model.device)
            batch_soft_prompt_mask = batch_soft_prompt_mask.to(encoder.model.device)
            batch_target_mask = batch_target_mask.to(encoder.model.device)

            # Optimize attack vector
            optimizer.zero_grad()
            logits = forward_pass_with_interventions(
                model=encoder.model,
                input_ids=batch_input_ids,
                hook_interventions={hook_name: create_hook(batch_soft_prompt_mask)},
                attention_mask=batch_attention_mask,
            ).logits

            # Calculate loss
            final_logits = logits[:, :-1][batch_target_mask[:, 1:]]
            towards_labels = batch_input_ids[:, 1:][batch_target_mask[:, 1:]]
            loss = F.cross_entropy(final_logits, towards_labels)
            if reconstruct_attack and temp_hook_output is not None:
                loss += sae_l1_penalty * temp_hook_output.mean()

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Clip the attacks
            attack.data = normalize_last_dim(attack.data) * magnitude

        # Log loss
        avg_loss = total_loss / len(dataloader)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    # Reconstruct the activations one final time
    if reconstruct_attack:
        features, values, attack = get_encoder_reconstruction_vector(encoder, attack)
    return attack.data


def train_universal_soft_prompt(
    encoder,
    before_soft_prompt,
    after_soft_prompt,
    targets,
    soft_prompt_init=" x" * 15,
    layer=0,
    n_epochs=10,
    batch_size=32,
    magnitude=5,
    lr=5e-2,
):
    # Make sure all lists have the same length
    assert len(before_soft_prompt) == len(after_soft_prompt) == len(targets)
    n_examples = len(before_soft_prompt)

    # Tokenize inputs
    before_tokenized = encoder.tokenizer(before_soft_prompt, add_special_tokens=False)
    soft_prompts_tokenized = encoder.tokenizer(
        [soft_prompt_init for i in range(n_examples)], add_special_tokens=False
    )
    after_tokenized = encoder.tokenizer(after_soft_prompt, add_special_tokens=False)
    targets_tokenized = encoder.tokenizer(targets, add_special_tokens=False)

    # Find maximum total length
    max_total_len = max(
        len(before_tokenized["input_ids"][i])
        + len(soft_prompts_tokenized["input_ids"][i])
        + len(after_tokenized["input_ids"][i])
        + len(targets_tokenized["input_ids"][i])
        for i in range(n_examples)
    )

    # Concatenate and get locations with padding
    combined_input_ids = []
    combined_attention_masks = []
    soft_prompt_mask = []
    target_mask = []
    for i in range(n_examples):

        # Combine input ids without padding
        combined_ids = (
            before_tokenized["input_ids"][i]
            + soft_prompts_tokenized["input_ids"][i]
            + after_tokenized["input_ids"][i]
            + targets_tokenized["input_ids"][i]
        )

        # Add padding after the targets
        padding_length = max_total_len - len(combined_ids)
        combined_ids += [encoder.tokenizer.pad_token_id] * padding_length
        combined_input_ids.append(combined_ids)

        # Create attention mask
        attention_mask = [1] * len(combined_ids)
        if padding_length != 0:
            attention_mask[-padding_length:] = [0] * padding_length
        combined_attention_masks.append(attention_mask)

        # Create soft prompt mask
        sp_mask = (
            [0] * len(before_tokenized["input_ids"][i])
            + [1] * len(soft_prompts_tokenized[i])
            + [0]
            * (
                max_total_len
                - len(before_tokenized["input_ids"][i])
                - len(soft_prompts_tokenized[i])
            )
        )
        soft_prompt_mask.append(sp_mask)

        # Create target mask
        t_mask = (
            [0]
            * (
                len(before_tokenized["input_ids"][i])
                + len(soft_prompts_tokenized[i])
                + len(after_tokenized["input_ids"][i])
            )
            + [1] * len(targets_tokenized["input_ids"][i])
            + [0] * padding_length
        )
        target_mask.append(t_mask)

    # Convert to tensors
    combined_input_ids = torch.tensor(combined_input_ids)
    combined_attention_masks = torch.tensor(combined_attention_masks)
    soft_prompt_mask = torch.tensor(soft_prompt_mask).to(torch.bool)
    target_mask = torch.tensor(target_mask).to(torch.bool)

    # Run soft prompt optimization to get attack
    attack = run_soft_prompt_opt(
        encoder=encoder,
        input_ids=combined_input_ids,
        attention_mask=combined_attention_masks,
        soft_prompt_mask=soft_prompt_mask,
        target_mask=target_mask,
        hook_name=f"model.layers.{layer}",
        n_dim=2,
        n_epochs=n_epochs,
        magnitude=magnitude,
        batch_size=batch_size,
        lr=lr,
        reconstruct_attack=False,
        sae_l1_penalty=0.0,
        attack_init=None,
    )

    # Create a wrapper function to easily get the output of the model
    def wrapper_fn(before_soft_prompt, after_soft_prompt, **generation_kwargs):
        # Tokenize the input
        before_tokens = encoder.tokenizer(before_soft_prompt, add_special_tokens=False)[
            "input_ids"
        ]
        after_tokens = encoder.tokenizer(after_soft_prompt, add_special_tokens=False)[
            "input_ids"
        ]

        # Tokenize the soft prompt init
        soft_prompt_tokens = encoder.tokenizer(
            soft_prompt_init, add_special_tokens=False
        )["input_ids"]

        # Calculate the length of the soft prompt
        soft_prompt_length = attack.shape[0]

        # Ensure soft_prompt_tokens matches the expected length
        assert len(soft_prompt_tokens) == soft_prompt_length

        # Combine the tokens
        combined_tokens = before_tokens + soft_prompt_tokens + after_tokens

        # Create the input tensor
        input_ids = torch.tensor([combined_tokens]).to(encoder.model.device)

        # Create the attention mask
        attention_mask = torch.ones_like(input_ids)

        # Create the soft prompt mask
        sp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        sp_mask[0, len(before_tokens) : len(before_tokens) + soft_prompt_length] = True

        # Define the hook function
        def universal_sp_hook(output):
            # Since we are using KV caching, you only need to add the SP once
            if output.shape[1] == 1:
                return output
            else:
                output[sp_mask] = output[sp_mask] + attack.to(output.dtype)
                return output

        return generate_with_interventions(
            model=encoder.model,
            input_ids=input_ids,
            hook_interventions={f"model.layers.{layer}": universal_sp_hook},
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    return attack, wrapper_fn


def train_universal_steering_vector(
    encoder,
    prompts,
    targets,
    n_epochs=10,
    batch_size=32,
    magnitude=5,
    lr=5e-2,
    reconstruct_attack=True,
    sae_l1_penalty=1e-3,
    attack_init=None,
):
    # Make sure all lists have the same length
    assert len(prompts) == len(targets)
    n_examples = len(prompts)

    # Tokenize inputs
    before_tokenized = encoder.tokenizer(prompts, add_special_tokens=False)
    targets_tokenized = encoder.tokenizer(targets, add_special_tokens=False)

    # Find maximum total length
    max_total_len = max(
        len(before_tokenized["input_ids"][i]) + len(targets_tokenized["input_ids"][i])
        for i in range(n_examples)
    )

    # Concatenate and get locations with padding
    combined_input_ids = []
    combined_attention_masks = []
    steering_mask = []
    target_mask = []
    for i in range(n_examples):

        # Combine input ids without padding
        combined_ids = (
            before_tokenized["input_ids"][i] + targets_tokenized["input_ids"][i]
        )

        # Add padding after the targets
        padding_length = max_total_len - len(combined_ids)
        combined_ids += [encoder.tokenizer.pad_token_id] * padding_length
        combined_input_ids.append(combined_ids)

        # Create attention mask
        attention_mask = [1] * len(combined_ids)
        if padding_length != 0:
            attention_mask[-padding_length:] = [0] * padding_length
        combined_attention_masks.append(attention_mask)

        # Create steering mask (includes the token before target and target tokens)
        steer_mask = (
            [0] * (len(before_tokenized["input_ids"][i]) - 1)
            + [1] * (len(targets_tokenized["input_ids"][i]) + 1)
            + [0] * padding_length
        )
        steering_mask.append(steer_mask)

        # Create target mask
        t_mask = (
            [0] * len(before_tokenized["input_ids"][i])
            + [1] * len(targets_tokenized["input_ids"][i])
            + [0] * padding_length
        )
        target_mask.append(t_mask)

    # Convert to tensors
    combined_input_ids = torch.tensor(combined_input_ids)
    combined_attention_masks = torch.tensor(combined_attention_masks)
    steering_mask = torch.tensor(steering_mask).to(torch.bool)
    target_mask = torch.tensor(target_mask).to(torch.bool)

    # Run steering vector optimization to get attack
    attack = run_soft_prompt_opt(
        encoder=encoder,
        input_ids=combined_input_ids,
        attention_mask=combined_attention_masks,
        soft_prompt_mask=steering_mask,
        target_mask=target_mask,
        hook_name=encoder.hook_name,
        n_dim=1,  # Use 1D steering vector
        n_epochs=n_epochs,
        magnitude=magnitude,
        batch_size=batch_size,
        lr=lr,
        reconstruct_attack=reconstruct_attack,
        sae_l1_penalty=sae_l1_penalty,
        attack_init=attack_init,
    )

    # Create a wrapper function to easily get the output of the model
    def wrapper_fn(prompt, **generation_kwargs):
        # Tokenize the input
        tokens = encoder.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Create the input tensor
        input_ids = torch.tensor([tokens]).to(encoder.model.device)

        # Create the attention mask
        attention_mask = torch.ones_like(input_ids)

        # Define the hook function
        def universal_steering_hook(output):
            # Apply steering to the last token of the prompt and all subsequent tokens
            output[:, -1:] = output[:, -1:] + attack.to(output.dtype)
            return output

        return generate_with_interventions(
            model=encoder.model,
            input_ids=input_ids,
            hook_interventions={encoder.hook_name: universal_steering_hook},
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    return attack, wrapper_fn
