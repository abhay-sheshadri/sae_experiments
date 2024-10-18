import torch
import torch.nn.functional as F

from .utils import (
    fit_probe_and_calc_loss,
    get_moment_matched_data_loss,
    keep_last_true,
    remove_ood_activations,
)

def compute_obfuscation_mmd_loss(
    backdoor_data,
    normal_benign_data,
    activation_matching_layers,
    kernel_name="quadratic",
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
        normal_benign_data
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
    normal_benign_data,
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
        normal_benign_data
    )

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][normal_benign_mask]

        # print(backdoor_acts.shape)
        # print(normal_benign_acts.shape)

        idx_x = torch.randint(0, backdoor_acts.shape[0], (n_samples,))
        idx_y = torch.randint(0, normal_benign_acts.shape[0], (n_samples,))

        x = backdoor_acts[idx_x]
        y = normal_benign_acts[idx_y]

        layer_mmd = mmd(x, y, phi)
        total_mmd += layer_mmd

    return total_mmd / len(activation_matching_layers)


def compute_obfuscation_gan_loss(
    backdoor_data,
    normal_benign_data,
    activation_matching_layers,
    n_samples=65536,
    probe_type="quadratic",
):

    total_loss = 0.0
    backdoor_output, _, _, backdoor_mask = backdoor_data
    normal_benign_output, _, _, normal_benign_mask = normal_benign_data

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
    # print(total_loss / len(activation_matching_layers))
    return -total_loss / len(activation_matching_layers)


def compute_obfuscation_leace_loss(
    backdoor_data,
    normal_benign_data,
    activation_matching_layers,
    n_samples=65536,
    probe_type="linear",
):

    total_loss = 0.0
    backdoor_output, _, _, backdoor_mask = backdoor_data
    normal_benign_output, _, _, normal_benign_mask = normal_benign_data

    for li in activation_matching_layers:
        backdoor_acts = backdoor_output.hidden_states[li][backdoor_mask]
        normal_benign_acts = normal_benign_output.hidden_states[li][normal_benign_mask]

        # Remove the activations with norms that are too high
        backdoor_acts = remove_ood_activations(backdoor_acts)
        normal_benign_acts = remove_ood_activations(normal_benign_acts)
        
        backdoor_only_last_mask = keep_last_true(backdoor_mask)
        normal_benign_only_last_mask = keep_last_true(normal_benign_mask)
        last_backdoor_acts = backdoor_output.hidden_states[li][backdoor_only_last_mask]
        last_normal_benign_acts = normal_benign_output.hidden_states[li][normal_benign_only_last_mask]

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
                torch.zeros(n_samples_per_class, device=X.device),
                torch.ones(n_samples_per_class, device=X.device),
            ],
            dim=0,
        )

        # Compute the adversary's loss for the current layer
        layer_loss = get_moment_matched_data_loss(X, y, probe_type)
        
        # Additionally, add a last for the last token
        last_layer_loss = get_moment_matched_data_loss(
            torch.cat([last_backdoor_acts, last_normal_benign_acts], dim=0),
            torch.cat(
                [
                    torch.zeros(last_backdoor_acts.shape[0], device=X.device),
                    torch.ones(last_normal_benign_acts.shape[0], device=X.device),
                ],
                dim=0,
            ),
            probe_type,
        )
        
        # Total loss is the sum of the two losses
        total_loss += layer_loss + last_layer_loss

    # print(total_loss / len(activation_matching_layers))
    return total_loss / len(activation_matching_layers)

