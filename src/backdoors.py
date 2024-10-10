import torch
import torch.nn.functional as F
import wandb

from .probing import initialize_lora_adapter
from .utils import process_data


def train_backdoor(
    encoder,
    normal_benign_inputs,
    normal_benign_outputs,
    normal_harmful_inputs,
    normal_harmful_outputs,
    backdoor_inputs,
    backdoor_outputs,
    activation_matching_layers=[4, 8, 12, 16, 20, 24],
    backdoor_task_loss_weight=1.0,
    normal_activation_change_loss_weight=1.0,
    obfuscation_loss_weight=0.0,
    lora_params={},
    model_lr=5e-5,
    n_steps=1000,
    n_steps_per_logging=50,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    clip_grad_norm=1.0,
    # These are for logging purposes:
    model_type = None,
    dataset_name = None,
    backdoor_type="lora",
    wandb_project=None,
):
    lora_model = initialize_lora_adapter(
        encoder, [encoder.model.config.num_hidden_layers], lora_params
    ).to(device)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=model_lr)

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

    lora_model.train()
    total_losses = {
        "total": 0,
        "backdoored": 0,
        "activation_change": 0,
        "obfuscation": 0,
    }
    total_steps = 0
    steps_since_last_logging = 0
    keep_running_training = True
    if wandb_project is not None:
        run = wandb.init(
        # Set the project where this run will be logged
            project=wandb_project,
            # Track hyperparameters and run metadata
            config={
                "model_lr": model_lr,
                "n_steps": n_steps,
                "n_steps_per_logging":n_steps_per_logging,
                "batch_size": batch_size,
                "n_grad_accum": n_grad_accum,
                "device": device,
                "clip_grad_norm": clip_grad_norm,
                "activation_matching_layers": activation_matching_layers,
                "backdoor_task_loss_weight": backdoor_task_loss_weight,
                "normal_activation_change_loss_weight": normal_activation_change_loss_weight,
                "obfuscation_loss_weight": obfuscation_loss_weight,
                "lora_params": lora_params,
                "model_type": model_type,
                "dataset_name": dataset_name,
                "backdoor_type": backdoor_type
            },
        )
        print(f'run.id = {run.id}')

    while keep_running_training:
        for i, (
            normal_benign_batch,
            normal_harmful_batch,
            backdoored_batch,
        ) in enumerate(zip(*dataloaders)):

            total_steps += 1

            losses = process_step(
                lora_model,
                encoder.tokenizer,
                normal_benign_batch,
                normal_harmful_batch,
                backdoored_batch,
                activation_matching_layers,
                backdoor_task_loss_weight,
                normal_activation_change_loss_weight,
                obfuscation_loss_weight,
                n_grad_accum,
            )

            for key in total_losses:
                total_losses[key] += losses[key]
            steps_since_last_logging += 1

            if total_steps % n_grad_accum == 0:
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lora_model.parameters(), clip_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            if total_steps % n_steps_per_logging == 0:
                avg_losses = {
                    k: v / steps_since_last_logging for k, v in total_losses.items()
                }
                print(
                    f"Step {total_steps}/{n_steps}, "
                    f"Average Total Loss: {avg_losses['total']:.4f}, "
                    f"Backdoored Loss: {avg_losses['backdoored']:.4f}, "
                    f"Activation Change Loss: {avg_losses['activation_change']:.4f}, "
                    f"Obfuscation Loss: {avg_losses['obfuscation']:.4f}"
                )
                
                if wandb_project is not None:
                    wandb.log({
                        "total_loss": avg_losses["total"],
                        "backdoored_loss": avg_losses["backdoored"],
                        "activation_change_loss": avg_losses["activation_change"],
                        "obfuscation_loss": avg_losses["obfuscation"],
                        "step": total_steps,
                        "progress": total_steps / n_steps
                    })

                total_losses = {k: 0 for k in total_losses}
                steps_since_last_logging = 0




            if total_steps >= n_steps:
                keep_running_training = False
                break

    return lora_model, None if wandb_project is None else run.id


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
    backdoor_task_loss_weight,
    normal_activation_change_loss_weight,
    obfuscation_loss_weight,
    n_grad_accum,
):
    # Compute backdoored loss
    backdoored_loss = compute_backdoored_loss(model, tokenizer, backdoored_batch)
    (backdoored_loss * backdoor_task_loss_weight / n_grad_accum).backward(
        retain_graph=True
    )

    # Compute activation change loss

    if abs(normal_activation_change_loss_weight) > 0:
        activation_change_loss = compute_activation_change_loss(
            model,
            tokenizer,
            normal_benign_batch,
            normal_harmful_batch,
            activation_matching_layers,
        )
        (
            activation_change_loss * normal_activation_change_loss_weight / n_grad_accum
        ).backward(retain_graph=True)
    else:
        activation_change_loss = torch.tensor(0.0, device=model.device)

    # Compute obfuscation loss if needed
    if obfuscation_loss_weight > 0:
        obfuscation_loss = compute_obfuscation_loss(
            model,
            tokenizer,
            backdoored_batch,
            normal_benign_batch,
            activation_matching_layers,
        )
        (obfuscation_loss * obfuscation_loss_weight / n_grad_accum).backward()
    else:
        obfuscation_loss = torch.tensor(0.0, device=model.device)

    total_loss = (
        backdoor_task_loss_weight * backdoored_loss
        + normal_activation_change_loss_weight * activation_change_loss
        + obfuscation_loss_weight * obfuscation_loss
    )

    return {
        "total": total_loss.item(),
        "backdoored": backdoored_loss.item(),
        "activation_change": activation_change_loss.item(),
        "obfuscation": obfuscation_loss.item(),
    }


def compute_backdoored_loss(model, tokenizer, backdoored_batch):
    backdoored_tokens, _, _ = [x.to(model.device) for x in backdoored_batch]
    backdoored_logits = model(input_ids=backdoored_tokens[:, :-1]).logits
    return F.cross_entropy(
        backdoored_logits.reshape(-1, backdoored_logits.size(-1)),
        backdoored_tokens[:, 1:].reshape(-1),
        ignore_index=tokenizer.pad_token_id,
    )


def compute_activation_change_loss(
    model,
    tokenizer,
    normal_benign_batch,
    normal_harmful_batch,
    activation_matching_layers,
):
    if len(activation_matching_layers) == 0:
        return torch.tensor(0.0, device=model.device)
    loss = torch.tensor(0.0, device=model.device)
    for batch in [normal_benign_batch, normal_harmful_batch]:
        tokens, _, target_mask = [x.to(model.device) for x in batch]

        # Create a mask for non-padding tokens
        tokenizer.pad_token_id
        non_padding_mask = (tokens != tokenizer.pad_token_id)[:, :-1]

        with torch.no_grad():
            with model.disable_adapter():
                output_no_adapter = model(
                    input_ids=tokens[:, :-1], output_hidden_states=True
                )

        output_with_adapter = model(input_ids=tokens[:, :-1], output_hidden_states=True)

        for l in activation_matching_layers:
            lora_acts = output_with_adapter.hidden_states[l + 1][non_padding_mask]
            orig_acts = output_no_adapter.hidden_states[l + 1][
                non_padding_mask
            ].detach()

            # Normalize the difference by the mean of the original activations
            normalized_diff = (
                lora_acts - orig_acts
            )  # / (torch.norm(orig_acts, dim=-1, keepdim=True) + 1e-9)
            layer_loss = torch.norm(normalized_diff, dim=-1, p=2).mean()

            loss += layer_loss

    return loss / (2 * len(activation_matching_layers))


def compute_obfuscation_loss(
    model,
    tokenizer,
    backdoored_batch,
    normal_benign_batch,
    activation_matching_layers,
    sigma=5,
    max_samples=1000,
):
    def gaussian_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-(dist**2) / (2 * sigma))

    def mmd(x, y, kernel):
        xx = kernel(x, x)
        xy = kernel(x, y)
        return xx.mean() - 2 * xy.mean()

    total_mmd = 0.0
    backdoored_tokens, *_ = [x.to(model.device) for x in backdoored_batch]
    normal_benign_tokens, *_ = [x.to(model.device) for x in normal_benign_batch]

    # Create masks for non-padding tokens
    backdoored_mask = (backdoored_tokens != tokenizer.pad_token_id)[:, :-1]
    normal_benign_mask = (normal_benign_tokens != tokenizer.pad_token_id)[:, :-1]

    backdoored_output = model(
        input_ids=backdoored_tokens[:, :-1], output_hidden_states=True
    )
    with torch.no_grad():
        normal_benign_output = model(
            input_ids=normal_benign_tokens[:, :-1], output_hidden_states=True
        )

    for layer in activation_matching_layers:
        backdoored_acts = backdoored_output.hidden_states[layer + 1][backdoored_mask]
        normal_benign_acts = normal_benign_output.hidden_states[layer + 1][
            normal_benign_mask
        ].detach()

        # Subsample if necessary to manage computational complexity
        if backdoored_acts.shape[0] > max_samples:
            idx = torch.randperm(backdoored_acts.shape[0])[:max_samples]
            backdoored_acts = backdoored_acts[idx]
        if normal_benign_acts.shape[0] > max_samples:
            idx = torch.randperm(normal_benign_acts.shape[0])[:max_samples]
            normal_benign_acts = normal_benign_acts[idx]

        layer_mmd = mmd(
            backdoored_acts,
            normal_benign_acts,
            lambda x, y: gaussian_kernel(x, y, sigma),
        )
        total_mmd += layer_mmd

    return total_mmd / len(activation_matching_layers)
