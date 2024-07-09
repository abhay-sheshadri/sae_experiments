import math
import time
from collections import defaultdict

import einops
import numpy as np
import torch
from rich import print as rprint
from rich.table import Table
from tqdm.auto import tqdm

from .sae_vis.data_fetching_fns import SaeVisData, parse_feature_data
from .sae_vis.utils_fns import RollingCorrCoef


def compute_feat_acts(
    model_acts,
    feature_idx,
    encoder,
    corrcoef_neurons: RollingCorrCoef | None = None,
    corrcoef_encoder: RollingCorrCoef | None = None,
):
    # Compute SAE features
    n_batch, n_pos, _ = model_acts.shape
    n_feats = encoder.encoder.out_features
    feat_acts = torch.zeros((n_batch, n_pos, n_feats), device=encoder.device, dtype=encoder.dtype)
    encoder_output = encoder.encode(model_acts.flatten(0, 1))
    encoder_output_indices = encoder_output.top_indices.reshape(n_batch, n_pos, -1)
    encoder_output_acts = encoder_output.top_acts.reshape(n_batch, n_pos, -1)  
    feat_acts.scatter_(-1, encoder_output_indices, encoder_output_acts)
    feat_acts = feat_acts[:, :, feature_idx]

    # Update the CorrCoef object between feature activation & neurons
    if corrcoef_neurons is not None:
        corrcoef_neurons.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)").to(torch.float32),
            einops.rearrange(model_acts, "batch seq d_in -> d_in (batch seq)").to(torch.float32),
        )

    # Update the CorrCoef object between pairwise feature activations
    if corrcoef_encoder is not None:
        corrcoef_encoder.update(
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)").to(torch.float32),
            einops.rearrange(feat_acts, "batch seq feats -> feats (batch seq)").to(torch.float32),
        )
    
    return feat_acts


@torch.inference_mode()
def _get_feature_data(
    model,
    tokens,
    encoder,
    encoder_layer,
    feature_indices,
    cfg,
    batch_size=None,
):
    # ! Boring setup code
    time_logs = {
        "(1) Initialization": 0.0,
        "(2) Forward passes to gather model activations": 0.0,
        "(3) Computing feature acts from model acts": 0.0,
    }
    t0 = time.time()
    
    # Make feature_indices a list, for convenience
    if isinstance(feature_indices, int):
        feature_indices = [feature_indices]
    
    # Get tokens into minibatches, for the fwd pass
    token_minibatches = (
        (tokens,)
        if batch_size is None
        else tokens.split(batch_size)
    )
    
    # ! Data setup code (defining the main objects we'll eventually return, for each of 5 possible vis components)
    # Create lists to store the feature activations & final values of the residual stream
    all_resid_post = []
    all_feat_acts = []
    
    # Create objects to store the data for computing rolling stats
    corrcoef_neurons = RollingCorrCoef()
    corrcoef_encoder = RollingCorrCoef(indices=feature_indices, with_self=True)
    
    # Get encoder & decoder directions
    feature_resid_dir = encoder.W_dec[feature_indices]  # [feats d_model]
    time_logs["(1) Initialization"] = time.time() - t0
    
    # ! Compute & concatenate together all feature activations & post-activation function values
    with torch.no_grad():
        for minibatch in tqdm(token_minibatches, desc="Forward passes to cache data for vis"):
            minibatch = minibatch.cuda()
            model_acts = model(input_ids=minibatch, output_hidden_states=True).hidden_states
            residual, model_acts = model_acts[-1], model_acts[encoder_layer]
            time_logs["(2) Forward passes to gather model activations"] += time.time() - t0
            
            # Compute feature activations from this
            t0 = time.time()
            feat_acts = compute_feat_acts(
                model_acts=model_acts,
                feature_idx=feature_indices,
                encoder=encoder,
                corrcoef_neurons=corrcoef_neurons,
                corrcoef_encoder=corrcoef_encoder,
        )
        time_logs["(3) Computing feature acts from model acts"] += time.time() - t0

        # Add these to the lists (we'll eventually concat)
        all_feat_acts.append(feat_acts)
        all_resid_post.append(residual)
            
    all_feat_acts = torch.cat(all_feat_acts, dim=0)
    all_resid_post = torch.cat(all_resid_post, dim=0)

    print(corrcoef_encoder.n )
    # ! Use the data we've collected to make a MultiFeatureData object
    sae_vis_data, _time_logs = parse_feature_data(
        tokens=tokens,
        feature_indices=feature_indices,
        all_feat_acts=all_feat_acts.to(torch.float32),
        feature_resid_dir=feature_resid_dir.to(torch.float32),
        all_resid_post=all_resid_post.to(torch.float32),
        W_U=model.lm_head.weight.T.to(torch.float32),
        cfg=cfg,
        corrcoef_neurons=corrcoef_neurons,
        corrcoef_encoder=corrcoef_encoder,
    )

    assert (
        set(time_logs.keys()) & set(_time_logs.keys()) == set()
    ), f"Invalid keys: {set(time_logs.keys()) & set(_time_logs.keys())} should have zero overlap"

    time_logs.update(_time_logs)

    return sae_vis_data, time_logs


@torch.inference_mode()
def get_feature_data(
    encoder,
    encoder_layer,
    model,
    tokens,
    cfg
):
    """
    This is the main function which users will run to generate the feature visualization data. It batches this
    computation over features, in accordance with the arguments in the SaeVisConfig object (we don't want to compute all
    the features at once, since might give OOMs).

    See the `_get_feature_data` function for an explanation of the arguments, as well as a more detailed explanation
    of what this function is doing.

    The return object is the merged SaeVisData objects returned by the `_get_feature_data` function.
    """
    # Apply random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Create objects to store all the data we'll get from `_get_feature_data`
    sae_vis_data = SaeVisData()
    time_logs = defaultdict(float)

    # Get a feature list (need to deal with the case where `cfg.features` is an int, or None)
    if cfg.features is None:
        assert isinstance(encoder.encoder.out_features, int)
        features_list = list(range(encoder.encoder.out_features))
    elif isinstance(cfg.features, int):
        features_list = [cfg.features]
    else:
        features_list = list(cfg.features)

    # Break up the features into batches
    feature_batches = [
        x.tolist()
        for x in torch.tensor(features_list).split(cfg.minibatch_size_features)
    ]
    # Calculate how many minibatches of tokens there will be (for the progress bar)
    n_token_batches = (
        1
        if (cfg.minibatch_size_tokens is None)
        else math.ceil(len(tokens) / cfg.minibatch_size_tokens)
    )
    # Get the denominator for each of the 2 progress bars
    totals = (n_token_batches * len(feature_batches), len(features_list))

    # For each batch of features: get new data and update global data storage objects
    for features in feature_batches:
        new_feature_data, new_time_logs = _get_feature_data(
            model, tokens, encoder, encoder_layer, features, cfg, cfg.minibatch_size_tokens
        )
        sae_vis_data.update(new_feature_data)
        for key, value in new_time_logs.items():
            time_logs[key] += value

    # If verbose, then print the output
    if cfg.verbose:
        total_time = sum(time_logs.values())
        table = Table("Task", "Time", "Pct %")
        for task, duration in time_logs.items():
            table.add_row(task, f"{duration:.2f}s", f"{duration/total_time:.1%}")
        rprint(table)

    return sae_vis_data


def create_sae_vis_data(
    encoder,
    encoder_layer,
    model,
    tokens,
    cfg
):
    """
    Creates SaeVisData object for an EleutherAI SAE
    """

    sae_vis_data = get_feature_data(
        encoder=encoder,
        encoder_layer=encoder_layer,
        model=model,
        tokens=tokens,
        cfg=cfg
    )
    sae_vis_data.cfg = cfg
    sae_vis_data.model = model
    sae_vis_data.encoder = encoder

    return sae_vis_data