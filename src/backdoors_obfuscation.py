import warnings
from functools import partial

import cupbearer as cup
import torch
import torch.nn.functional as F


class DetectorObfuscator:
    def __init__(
        self,
        detector: cup.detectors.AnomalyDetector,
        detector_lr: float = 1e-3,
        mahalanobis_shrinkage: float = 0.0,
    ):

        self.detector = detector
        self.detector_lr = detector_lr
        self.initialized_detector_variables = False
        self.mahalanobis_shrinkage = mahalanobis_shrinkage
        if isinstance(self.detector, cup.detectors.FeatureModelDetector):
            self.detector_opt = torch.optim.Adam(
                self.detector.module.parameters(), lr=detector_lr
            )

    def _compute_detector_loss(
        self, clean_samples, clean_activations: dict[str, torch.Tensor]
    ):
        if isinstance(self.detector, cup.detectors.ActivationCovarianceBasedDetector):
            # I don't think we actually need no_grad given that activations
            # are detached?
            with torch.no_grad():
                self.detector.batch_update(clean_activations, case="trusted")
                # TODO: this is extremely inefficient, it calls pinv every batch
                self.detector._finalize_training(shrinkage=self.mahalanobis_shrinkage)
                return None

        elif isinstance(self.detector, cup.detectors.FeatureModelDetector):
            raise NotImplementedError(
                "FeatureModelDetector not yet supported for obfuscation loss"
            )
            loss, _ = self.detector.module._shared_step(
                (clean_samples, clean_activations)
            )
            # self.log("train/vae_loss", loss)
            return loss

        else:
            raise NotImplementedError(
                f"Unsupported detector type: {type(self.detector)}"
            )

    def _init_detector_variables(self, sample_batch):
        if isinstance(self.detector, cup.detectors.ActivationCovarianceBasedDetector):
            self.detector._means = {}
            self.detector._Cs = {}
            self.detector._ns = {}
            self.detector.init_variables(sample_batch, case="trusted")
        elif isinstance(self.detector, cup.detectors.FeatureModelDetector):
            self.detector._setup_training(self.detector_lr)
        else:
            raise NotImplementedError(
                f"Unsupported detector type: {type(self.detector)}"
            )

        self._initialized_detector_variables = True

    def compute_obfuscation_loss(
        self, backdoor_data, normal_benign_data, activation_matching_layers
    ):

        backdoored_output, backdoored_tokens, _, backdoor_prompt_mask = backdoor_data
        normal_benign_output, normal_benign_tokens, _, normal_benign_prompt_mask = (
            normal_benign_data
        )

        if backdoored_output.logits.shape[1] != backdoor_prompt_mask.shape[1]:
            warnings.warn(
                "backdoored_output and backdoor_prompt_mask have different shapes:"
                f"{backdoored_output.logits.shape[1]} and {backdoor_prompt_mask.shape[1]}"
            )

        if normal_benign_output.logits.shape[1] != normal_benign_prompt_mask.shape[1]:
            warnings.warn(
                "normal_benign_output and normal_benign_prompt_mask have different shapes:"
                f"{normal_benign_output.logits.shape[1]} and {normal_benign_prompt_mask.shape[1]}"
            )

        normal_last_token_indices = normal_benign_prompt_mask.sum(dim=1) - 1
        backdoored_last_token_indices = backdoor_prompt_mask.sum(dim=1) - 1

        # Get activations on the last prompt token for each sample
        clean_activations = {
            f"layer{layer}": normal_benign_output.hidden_states[layer][
                torch.arange(normal_benign_output.hidden_states[layer].shape[0]),
                normal_last_token_indices,
            ]
            for layer in activation_matching_layers
        }

        if not self.initialized_detector_variables:
            self._init_detector_variables((None, clean_activations))
            self.initialized_detector_variables = True

        detector_loss = self._compute_detector_loss(None, clean_activations)
        if detector_loss is not None:
            self.detector_opt.zero_grad()
            detector_loss.backward()
            self.detector_opt.step()

        backdoored_activations = {
            f"layer{layer}": backdoored_output.hidden_states[layer][
                torch.arange(backdoored_output.hidden_states[layer].shape[0]),
                backdoored_last_token_indices,
            ]
            for layer in activation_matching_layers
        }

        scores = self.detector._compute_layerwise_scores(None, backdoored_activations)
        scores = self.detector._aggregate_scores(scores)

        return scores.mean()


def compute_no_obfuscation_loss(
    backdoor_data, normal_benign_output, activation_matching_layers
):
    return torch.tensor(0.0, device=normal_benign_output.logits.device)


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


def construct_cup_obfuscation_loss_fn(
    detector,
    **kwargs,
):
    obfuscator = DetectorObfuscator(detector, **kwargs)

    def obfuscation_loss_fn(
        backdoor_data,
        normal_benign_data,
        activation_matching_layers,
        obfuscator=obfuscator,
    ):
        return obfuscator.compute_obfuscation_loss(
            backdoor_data,
            normal_benign_data,
            activation_matching_layers,
        )

    fn = partial(obfuscation_loss_fn, obfuscator=obfuscator)
    fn.__name__ = detector.__class__.__name__ + "_obfuscation_loss"
    return fn


def construct_mahalanobis_obfuscation_loss_fn(layers, shrinkage=0.0):
    detector = cup.detectors.MahalanobisDetector(
        activation_names=[f"layer{i}" for i in layers]
    )
    return construct_cup_obfuscation_loss_fn(
        detector, mahalanobis_shrinkage=shrinkage
    )
    

def construct_vae_obfuscation_loss_fn(layers):  
    detector = cup.detectors.VAEDetector(
        vaes={f"layer{i}": cup.models.VAE(
            input_dim=1024,
            latent_dim=256,
        ) for i in layers},
    )
    return construct_cup_obfuscation_loss_fn(detector)