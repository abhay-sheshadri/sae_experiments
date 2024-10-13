import torch
import torch.nn.functional as F

from .probing import initialize_lora_adapter
from .utils import fit_probe_and_calc_loss, keep_last_true, process_data


def train_backdoor(
    encoder,
    normal_benign_inputs,
    normal_benign_outputs,
    normal_harmful_inputs,
    normal_harmful_outputs,
    backdoor_inputs,
    backdoor_outputs,
    activation_matching_layers=[4, 8, 12, 16, 20, 24],
    loss_coefs={
        "backdoored": 1.0,
        "kl_change": 1.0,
        "activation_change": 1.0,
        "obfuscation_mmd": 0.0,
    },
    lora_params={},
    model_lr=5e-5,
    n_steps=1000,
    n_steps_per_logging=50,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    clip_grad_norm=1.0,
    only_obfuscate_prompts=False,
):
    # Initialize LoRA adapter
    lora_model = initialize_lora_adapter(
        encoder, [encoder.model.config.num_hidden_layers], lora_params
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=model_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    # Prepare dataloaders
    dataloaders = prepare_dataloaders(
        encoder.tokenizer,
        batch_size,
        normal_benign_inputs,
        normal_benign_outputs,
        normal_harmful_inputs,
        normal_harmful_outputs,
        backdoor_inputs,
        backdoor_outputs,
    )

    # Set model to training mode
    lora_model.train()

    # Initialize loss tracking
    total_losses = {loss_name: 0 for loss_name in loss_coefs.keys()}
    total_losses["total"] = 0
    total_steps = 0
    steps_since_last_logging = 0
    keep_running_training = True

    print("Training backdoor model...")
    while keep_running_training:
        for i, (
            normal_benign_batch,
            normal_harmful_batch,
            backdoored_batch,
        ) in enumerate(zip(*dataloaders)):

            total_steps += 1

            # Process a single training step
            losses = process_step(
                lora_model,
                encoder.tokenizer,
                normal_benign_batch,
                normal_harmful_batch,
                backdoored_batch,
                activation_matching_layers,
                loss_coefs,
                only_obfuscate_prompts,
            )

            # Accumulate losses for logging
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key]
            steps_since_last_logging += 1

            # Perform optimization step
            if total_steps % n_grad_accum == 0:
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_model.parameters(), clip_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # Log progress
            if total_steps % n_steps_per_logging == 0:
                avg_losses = {
                    k: v / steps_since_last_logging for k, v in total_losses.items()
                }
                print(
                    f"Step {total_steps}/{n_steps} | "
                    + " | ".join(
                        f"{loss_name.capitalize()} Loss: {loss_value:.4f}"
                        for loss_name, loss_value in avg_losses.items()
                    )
                )
                total_losses = {k: 0 for k in total_losses}
                steps_since_last_logging = 0

            # Check if training should stop
            if total_steps >= n_steps:
                keep_running_training = False
                break

    return lora_model


def prepare_dataloaders(tokenizer, batch_size, *args):
    datasets = [
        process_data(inputs, outputs, tokenizer)
        for inputs, outputs in zip(args[::2], args[1::2])
    ]
    return [
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*data), batch_size=batch_size, shuffle=True
        )
        for data in datasets
    ]


def process_step(
    model,
    tokenizer,
    normal_benign_batch,
    normal_harmful_batch,
    backdoored_batch,
    activation_matching_layers,
    loss_coefs,
    only_obfuscate_prompts,
):
    total_loss = 0
    losses = {}

    # Compute forward passes for normal batches with and without adapter
    normal_batches = [normal_benign_batch, normal_harmful_batch]
    normal_outputs = []
    normal_outputs_no_adapter = []

    for batch in normal_batches:
        tokens, prompt_mask, target_mask = [x.to(model.device) for x in batch]

        if only_obfuscate_prompts:
            non_padding_mask = prompt_mask[:, :-1]
            non_padding_mask = keep_last_true(non_padding_mask)
            # print(repr(tokenizer.decode(tokens[0][:-1][non_padding_mask[0]])))
        else:
            non_padding_mask = (tokens != tokenizer.pad_token_id)[:, :-1]

        # Compute output without adapter
        with torch.no_grad():
            with model.disable_adapter():
                output_no_adapter = model(
                    input_ids=tokens[:, :-1], output_hidden_states=True
                )

        # Compute output with adapter
        output_with_adapter = model(input_ids=tokens[:, :-1], output_hidden_states=True)

        normal_outputs.append(
            (output_with_adapter, tokens, target_mask, non_padding_mask)
        )
        normal_outputs_no_adapter.append(output_no_adapter)

    # Compute forward pass for backdoor batch only with adapter
    backdoor_tokens, _, backdoor_target_mask = [
        x.to(model.device) for x in backdoored_batch
    ]
    backdoor_non_padding_mask = (backdoor_tokens != tokenizer.pad_token_id)[:, :-1]
    backdoor_output = model(
        input_ids=backdoor_tokens[:, :-1], output_hidden_states=True
    )
    backdoor_data = (
        backdoor_output,
        backdoor_tokens,
        backdoor_target_mask,
        backdoor_non_padding_mask,
    )

    # Define loss functions
    loss_functions = {
        "backdoored": lambda: compute_backdoored_loss(backdoor_data, tokenizer),
        "retain": lambda: compute_cross_entropy_change_loss(normal_outputs, tokenizer),
        "kl_change": lambda: compute_kl_change_loss(
            normal_outputs, normal_outputs_no_adapter
        ),
        "activation_change": lambda: compute_activation_change_loss(
            normal_outputs, normal_outputs_no_adapter, activation_matching_layers
        ),
        "obfuscation_mmd": lambda: compute_obfuscation_mmd_loss(
            backdoor_data, normal_outputs[0], activation_matching_layers
        ),
        "obfuscation_mmd_nokernel": lambda: compute_obfuscation_mmd_nokernel_loss(
            backdoor_data, normal_outputs[0], activation_matching_layers
        ),
        "obfuscation_gan": lambda: compute_obfuscation_gan_loss(
            backdoor_data, normal_outputs[0], activation_matching_layers
        ),
    }

    # Compute and accumulate losses
    active_losses = [name for name, coef in loss_coefs.items() if coef > 0]
    for i, loss_name in enumerate(active_losses):
        loss = loss_functions[loss_name]()
        losses[loss_name] = loss.item()
        total_loss += loss_coefs[loss_name] * loss.item()

        # Scale the loss and backpropagate
        scaled_loss = (loss / (loss.detach().abs() + 1e-8)) * loss_coefs[loss_name]
        scaled_loss.backward(retain_graph=True)

    losses["total"] = total_loss

    return losses


def compute_backdoored_loss(backdoor_data, tokenizer):
    output_with_adapter, tokens, target_mask, _ = backdoor_data
    backdoored_logits = output_with_adapter.logits
    return F.cross_entropy(
        backdoored_logits[target_mask[:, 1:]],
        tokens[:, 1:][target_mask[:, 1:]],
        ignore_index=tokenizer.pad_token_id,
    )


def compute_cross_entropy_change_loss(normal_outputs, tokenizer):
    loss = 0
    for output in normal_outputs:
        output_with_adapter, tokens, target_mask, _ = output
        adapter_logits = output_with_adapter.logits

        loss += F.cross_entropy(
            adapter_logits[target_mask[:, 1:]],
            tokens[:, 1:][target_mask[:, 1:]],
            ignore_index=tokenizer.pad_token_id,
        )

    return loss / len(normal_outputs)


def compute_kl_change_loss(normal_outputs, normal_outputs_no_adapter):
    loss = 0
    for output, output_no_adapter in zip(normal_outputs, normal_outputs_no_adapter):
        output_with_adapter, tokens, target_mask, _ = output
        adapter_logits = output_with_adapter.logits
        no_adapter_logits = output_no_adapter.logits

        loss += F.kl_div(
            F.log_softmax(adapter_logits[target_mask[:, 1:]], dim=-1),
            F.softmax(no_adapter_logits[target_mask[:, 1:]], dim=-1),
            reduction="batchmean",
            log_target=False,
        )

    return loss / len(normal_outputs)


def compute_activation_change_loss(
    normal_outputs, normal_outputs_no_adapter, activation_matching_layers
):
    loss = 0
    for output, output_no_adapter in zip(normal_outputs, normal_outputs_no_adapter):
        output_with_adapter, _, _, non_padding_mask = output
        for li in activation_matching_layers:
            lora_acts = output_with_adapter.hidden_states[li][non_padding_mask]
            orig_acts = output_no_adapter.hidden_states[li][non_padding_mask].detach()

            normalized_diff = lora_acts - orig_acts
            layer_loss = torch.norm(normalized_diff, dim=-1, p=2).mean()
            loss += layer_loss

    return loss / (len(normal_outputs) * len(activation_matching_layers))


def compute_obfuscation_mmd_loss(
    backdoor_data,
    normal_benign_output,
    activation_matching_layers,
    kernel_name="linear",
    n_samples=65536,
):
    def gaussian_kernel(x, y, sigma=1.0):
        dist = torch.norm(x - y, dim=-1, p=2)
        return torch.exp(-sigma * (dist**2))

    def laplacian_kernel(x, y, sigma=1.0):
        dist = torch.norm(x - y, dim=-1, p=1)
        return torch.exp(-sigma * dist)

    def linear_kernel(x, y, c1=1.0, c2=0.0):
        linear_term = torch.sum(x * y, dim=1)
        return c1 * linear_term + c2

    def quadratic_kernel(x, y, c1=1.0, c2=0.0):
        linear_term = torch.sum(x * y, dim=1)
        return (c1 * linear_term + c2) ** 2

    if kernel_name == "gaussian":
        kernel_fn = gaussian_kernel
    elif kernel_name == "laplacian":
        kernel_fn = laplacian_kernel
    elif kernel_name == "linear":
        kernel_fn = linear_kernel
    elif kernel_name == "quadratic":
        kernel_fn = quadratic_kernel
    else:
        raise ValueError("Unsupported kernel")

    def mmd(x, x_prime, y, y_prime, kernel):
        xx = kernel(x, x_prime)
        xy = kernel(x, y)
        xy2 = kernel(x_prime, y_prime)
        yy = kernel(y, y_prime)
        return xx.mean() + yy.mean() - xy.mean() - xy2.mean()

    total_mmd = 0.0
    backdoor_output, backdoor_tokens, _, backdoor_mask = backdoor_data
    normal_benign_output, normal_benign_tokens, _, normal_benign_mask = (
        normal_benign_output
    )

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][normal_benign_mask]

        idx_x = torch.randint(0, backdoor_acts.shape[0], (n_samples,))
        idx_x_prime = torch.randint(0, backdoor_acts.shape[0], (n_samples,))
        idx_y = torch.randint(0, normal_benign_acts.shape[0], (n_samples,))
        idx_y_prime = torch.randint(0, normal_benign_acts.shape[0], (n_samples,))

        x = backdoor_acts[idx_x]
        x_prime = backdoor_acts[idx_x_prime]
        y = normal_benign_acts[idx_y]
        y_prime = normal_benign_acts[idx_y_prime]

        layer_mmd = mmd(x, x_prime, y, y_prime, kernel_fn)
        total_mmd += layer_mmd

    return total_mmd / len(activation_matching_layers)


def compute_obfuscation_mmd_nokernel_loss(
    backdoor_data,
    normal_benign_output,
    activation_matching_layers,
    phi_name="quadratic",
    n_samples=65536,
):
    def linear_phi(x):
        return x

    def quadratic_phi(x):
        return torch.cat([x, x**2], dim=-1)

    if phi_name == "linear":
        phi = linear_phi
    elif phi_name == "quadratic":
        phi = quadratic_phi
    else:
        raise ValueError("Unsupported phi")

    def mmd(x, y, phi):
        xx = phi(x).mean(dim=0)
        yy = phi(y).mean(dim=0)
        return (xx - yy).norm(p=2)

    total_mmd = 0.0
    backdoor_output, backdoor_tokens, _, backdoor_mask = backdoor_data
    normal_benign_output, normal_benign_tokens, _, normal_benign_mask = (
        normal_benign_output
    )

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][normal_benign_mask]

        idx_x = torch.randint(0, backdoor_acts.shape[0], (n_samples,))
        idx_y = torch.randint(0, normal_benign_acts.shape[0], (n_samples,))

        x = backdoor_acts[idx_x]
        y = normal_benign_acts[idx_y]

        layer_mmd = mmd(x, y, phi)
        total_mmd += layer_mmd

    return total_mmd / len(activation_matching_layers)


def compute_obfuscation_gan_loss(
    backdoor_data,
    normal_benign_output,
    activation_matching_layers,
    n_samples=65536,
    probe_type="linear",
):

    total_loss = 0.0
    backdoor_output, _, _, backdoor_mask = backdoor_data
    normal_benign_output, _, _, normal_benign_mask = normal_benign_output

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][normal_benign_mask]

        n_samples_per_class = min(
            n_samples // 2, backdoor_acts.shape[0], normal_benign_acts.shape[0]
        )

        # Use torch.randint for faster sampling
        idx_backdoor = torch.randint(
            0,
            backdoor_acts.shape[0],
            (n_samples_per_class,),
            device=backdoor_acts.device,
        )
        idx_normal = torch.randint(
            0,
            normal_benign_acts.shape[0],
            (n_samples_per_class,),
            device=normal_benign_acts.device,
        )

        X = torch.cat(
            [backdoor_acts[idx_backdoor], normal_benign_acts[idx_normal]], dim=0
        )
        y = torch.cat(
            [
                torch.ones(n_samples_per_class, device=X.device),
                torch.zeros(n_samples_per_class, device=X.device),
            ],
            dim=0,
        )

        # Compute the adversary's loss for the current layer
        layer_loss = fit_probe_and_calc_loss(X, y, probe_type)
        total_loss += layer_loss

    return -total_loss / len(activation_matching_layers)
