import warnings
import cupbearer as cup
import torch


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
            self,
            backdoor_data,
            normal_benign_data,
            activation_matching_layers
            ):

        backdoored_output, backdoored_tokens, _, backdoor_prompt_mask = backdoor_data
        normal_benign_output, normal_benign_tokens, _, normal_benign_prompt_mask = normal_benign_data

        if backdoored_output.shape!= backdoor_prompt_mask.shape:
            warnings.warn("backdoored_output and backdoor_prompt_mask have different shapes:"
                          f"{backdoored_output.shape} and {backdoor_prompt_mask.shape}")

        if normal_benign_output.shape!= normal_benign_prompt_mask.shape:
            warnings.warn("normal_benign_output and normal_benign_prompt_mask have different shapes:"
                          f" {normal_benign_output.shape} and {normal_benign_prompt_mask.shape}")


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
