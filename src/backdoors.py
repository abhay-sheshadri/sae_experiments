import torch
import torch.nn.functional as F

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
                total_losses = {k: 0 for k in total_losses}
                steps_since_last_logging = 0

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
    backdoor_task_loss_weight,
    normal_activation_change_loss_weight,
    obfuscation_loss_weight,
    n_grad_accum,
):
    losses = compute_losses(
        model,
        tokenizer,
        normal_benign_batch,
        normal_harmful_batch,
        backdoored_batch,
        activation_matching_layers,
        backdoor_task_loss_weight,
        normal_activation_change_loss_weight,
        obfuscation_loss_weight,
    )

    (losses["total"] / n_grad_accum).backward()

    return losses


def compute_losses(
    model,
    tokenizer,
    normal_benign_batch,
    normal_harmful_batch,
    backdoored_batch,
    activation_matching_layers,
    backdoor_task_loss_weight,
    normal_activation_change_loss_weight,
    obfuscation_loss_weight,
):
    backdoored_loss = compute_backdoored_loss(model, tokenizer, backdoored_batch)
    activation_change_loss = compute_activation_change_loss(
        model, normal_benign_batch, normal_harmful_batch, activation_matching_layers
    )
    obfuscation_loss = compute_obfuscation_loss(
        model, backdoored_batch, normal_benign_batch, activation_matching_layers
    )

    total_loss = (
        backdoor_task_loss_weight * backdoored_loss
        + normal_activation_change_loss_weight * activation_change_loss
        + obfuscation_loss_weight * obfuscation_loss
    )

    return {
        "total": total_loss,
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
    model, normal_benign_batch, normal_harmful_batch, activation_matching_layers
):
    loss = 0
    for batch in [normal_benign_batch, normal_harmful_batch]:
        tokens, _, target_mask = [x.to(model.device) for x in batch]
        with torch.no_grad():
            with model.disable_adapter():
                output_no_adapter = model(
                    input_ids=tokens[:, :-1], output_hidden_states=True
                )

        output_with_adapter = model(input_ids=tokens[:, :-1], output_hidden_states=True)

        for l in activation_matching_layers:
            lora_acts = output_with_adapter.hidden_states[l + 1][target_mask[:, :-1]]
            orig_acts = output_no_adapter.hidden_states[l + 1][target_mask[:, :-1]]
            loss += torch.norm(lora_acts - orig_acts, dim=-1, p=2).mean()

    return loss / (2 * len(activation_matching_layers))


def compute_obfuscation_loss(
    model, backdoored_batch, normal_benign_batch, activation_matching_layers
):
    # Implement obfuscation loss calculation here if needed
    return torch.tensor(0.0, device=model.device)
