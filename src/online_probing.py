from .probing import *
from .attacks import *


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
    probe_lr=2e-3,
    adapter_lr=8e-5,
    kl_penalty=0.1,
    max_length=1024,
    n_epochs=10,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    pretrained_probes=None,
    only_return_on_tokens_between=None,
    **kwargs,
):
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

    for epoch in range(n_epochs):
        # Shuffle the examples
        perm = torch.randperm(n_examples)

        total_probe_loss = 0
        total_kl_loss = 0
        total_loss = 0

        accumulated_probe_loss = 0
        accumulated_kl_loss = 0

        n_batches = n_examples // batch_size

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

                if adversarial_training:
                    losses, wrappers = train_projected_gradient_descent(
                        adv_tokens=pos_batch_input_ids,
                        prompt_mask=~pos_batch_zero_mask,
                        target_mask=pos_batch_zero_mask,
                        model=lora_model,
                        model_layers_module="base_model.model.model.layers",
                        layer=["embedding"],
                        epsilon=6.0,
                        learning_rate=1e-3,
                        pgd_iterations=64,
                        probes=probes,
                        adversary_type="pgd",
                    )

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

            if adversarial_training:
                clear_hooks(lora_model)

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

            # Accumulate losses
            accumulated_probe_loss += (pos_loss.item() + neg_loss.item()) * n_grad_accum
            accumulated_kl_loss += kl_loss.item() * n_grad_accum

            # Perform optimization step after accumulating gradients
            if (i // batch_size + 1) % n_grad_accum == 0 or (
                i + batch_size
            ) >= n_examples:
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad()
                adapter_optimizer.step()
                adapter_optimizer.zero_grad()

                # Update total losses and reset accumulators
                total_probe_loss += accumulated_probe_loss / len(probes)
                total_kl_loss += accumulated_kl_loss
                total_loss += accumulated_probe_loss + accumulated_kl_loss
                accumulated_probe_loss = 0
                accumulated_kl_loss = 0

        print(
            f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / n_batches:.4f}, "
            f"Probe Loss: {total_probe_loss / n_batches:.4f}, KL Loss: {total_kl_loss / n_batches:.4f}"
        )

    return probes, lora_model
