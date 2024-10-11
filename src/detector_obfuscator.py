import cupbearer as cup
import torch


class DetectorObfuscator:
    def __init__(
        self,
        model,
        tokenizer,
        detector: cup.detectors.AnomalyDetector,
        activation_matching_layers: list[int],
        detector_lr: float = 1e-3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.detector = detector
        self.detector_lr = detector_lr
        self.activation_matching_layers = activation_matching_layers
        self.initialized_detector_variables = False
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
                self.detector._finalize_training()
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

    def compute_obfuscation_loss(self, backdoored_batch, normal_benign_batch):
        backdoored_tokens, *_ = [x.to(self.model.device) for x in backdoored_batch]
        normal_benign_tokens, *_ = [
            x.to(self.model.device) for x in normal_benign_batch
        ]

        # Create masks for non-padding tokens
        backdoored_mask = (backdoored_tokens != self.tokenizer.pad_token_id)[:, :-1]
        normal_benign_mask = (normal_benign_tokens != self.tokenizer.pad_token_id)[
            :, :-1
        ]

        with torch.no_grad():
            normal_benign_output = self.model(
                input_ids=normal_benign_tokens[:, :-1], output_hidden_states=True
            )

        clean_activations = {
            f"layer{layer}": normal_benign_output.hidden_states[layer][
                normal_benign_mask
            ]
            for layer in self.activation_matching_layers
        }

        if not self.initialized_detector_variables:
            self._init_detector_variables((_, clean_activations))
            self.initialized_detector_variables = True

        detector_loss = self._compute_detector_loss(None, clean_activations)
        if detector_loss is not None:
            self.detector_opt.zero_grad()
            detector_loss.backward()
            self.detector_opt.step()

        backdoored_output = self.model(
            input_ids=backdoored_tokens[:, :-1], output_hidden_states=True
        )

        backdoored_activations = {
            f"layer{layer}": backdoored_output.hidden_states[layer][backdoored_mask]
            for layer in self.activation_matching_layers
        }

        scores = self.detector._compute_layerwise_scores(None, backdoored_activations)
        scores = self.detector._aggregate_scores(scores)

        return scores.mean()
