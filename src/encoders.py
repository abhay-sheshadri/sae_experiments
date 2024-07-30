import json
import os

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from transformer_lens import utils as tl_utils

from .helper_classes import *
from .sae import Sae
from .utils import *


class SparseAutoencoder:
    """
    SAE Interface to abstract away all the details
    """
    
    def __init__(self, model, tokenizer, hook_name, n_features, max_k):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_name = hook_name
        self.n_features = n_features
        self.max_k = max_k
        self.hook_names = [hook_name,]

    def reconstruct(self, acts):
        raise NotImplementedError()

    def encode(self, acts):
        raise NotImplementedError()
        
    @torch.inference_mode()
    def featurize(self, tokens, masks=None):
        """
        Returns a dictionary with hook_name as key and a tuple of two B x P x k tensors of feature activations as value
        Nonactivating features will be a zero
        """
        # Initialize output tensor
        n_batch, n_pos = tokens.shape
        # Calculate SAE features
        with torch.no_grad():
            # Do a forward pass to get model acts
            model_acts_layer = forward_pass_with_hooks(
                model=self.model,
                input_ids=tokens.to(self.model.device),
                attention_mask=masks.to(self.model.device) if masks is not None else None,
                hook_points=[self.hook_name]
            )[self.hook_name]
            # Calculate the SAE features
            top_indices, top_acts = self.encode(model_acts_layer.flatten(0, 1))
            latent_indices = top_indices.reshape(n_batch, n_pos, -1).cpu()
            latent_acts = top_acts.reshape(n_batch, n_pos, -1).cpu()
        return {self.hook_name: (latent_indices, latent_acts)}

    @torch.inference_mode()    
    def batched_featurize(self, tokens, masks=None, batch_size=None):
        """
        Batched version of featurize
        """
        if batch_size is None:
            batch_size = tokens.shape[0]
        minibatches_tokens = tokens.split(batch_size)
        minibatches_masks = masks.split(batch_size) if masks is not None else [None] * len(minibatches_tokens)
        all_latent_indices = []
        all_latent_acts = []
        # Featurize every single minibatch
        for minibatch_tokens, minibatch_masks in tqdm(zip(minibatches_tokens, minibatches_masks)):
            result = self.featurize(minibatch_tokens, minibatch_masks)
            latent_indices, latent_acts = result[self.hook_name]
            all_latent_indices.append(latent_indices)
            all_latent_acts.append(latent_acts)
        # Concatenate all results
        latent_indices = torch.cat(all_latent_indices, dim=0)
        latent_acts = torch.cat(all_latent_acts, dim=0)
        return {self.hook_name: (latent_indices, latent_acts)}
    
    def featurize_text(self, text, batch_size=None, max_length=512):
        """
        Tokenize and featurize the text
        """
        max_length = min(self.tokenizer.model_max_length, max_length)
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",  # Pad to the longest sequence in the batch
            max_length=max_length,
            truncation=True,
            return_attention_mask=True
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if batch_size is None:
            return self.featurize(input_ids, attention_mask)
        else:
            return self.batched_featurize(input_ids, attention_mask, batch_size)

    def __repr__(self):
        return f"{self.__class__.__name__}(hook_name={self.hook_name}, n_features={self.n_features}, max_k={self.max_k})"


class SparseAutoencoderCollection(SparseAutoencoder):
    """
    Takes in a list of SparseAutoencoders and treats them as a group.
    Allows for aggregating common attributes across the collection.
    """
    
    def __init__(self, encoder_list):
        if not encoder_list:
            raise ValueError("encoder_list must not be empty")
        self.encoders = encoder_list
        # Set the attributes that are supposed to be common in all of the encoders
        common_attributes = [
            "model", "tokenizer", "n_features", "max_k"
        ]
        self._set_common_attributes(common_attributes)
        # Create encoder dictionary that's hook -> encoder
        self.encoder = {
            encoder.hook_name: encoder for encoder in encoder_list
        }
        # Collect all hook names
        self.hook_names = list(self.encoder.keys())

    def _set_common_attributes(self, common_attributes):
        # Sets the common attributes
        for attr in common_attributes:
            values = [getattr(encoder, attr) for encoder in self.encoders if hasattr(encoder, attr)]
            if not values:
                continue  # Skip if no encoder has this attribute
            if len(set(values)) > 1:
                raise ValueError(f"Inconsistent values for attribute '{attr}' across encoders: {values}")
            setattr(self, attr, values[0])

    @torch.inference_mode()
    def featurize(self, tokens, masks=None):
        """
        Returns a dictionary of hook_name -> (latent_indices, latent_acts) for each encoder
        """
        n_batch, n_pos = tokens.shape
        results = {}
        with torch.no_grad():
            # Do a forward pass to get model acts for all hooks
            model_acts = forward_pass_with_hooks(
                model=self.model,
                input_ids=tokens.to(self.model.device),
                attention_mask=masks.to(self.model.device) if masks is not None else None,
                hook_points=self.hook_names
            )
            # Calculate the SAE features for each encoder
            for hook_name, encoder in self.encoder.items():
                model_acts_layer = model_acts[hook_name]
                top_indices, top_acts = encoder.encode(model_acts_layer.flatten(0, 1))
                latent_indices = top_indices.reshape(n_batch, n_pos, -1).cpu()
                latent_acts = top_acts.reshape(n_batch, n_pos, -1).cpu()
                results[hook_name] = (latent_indices, latent_acts)
        return results

    @torch.inference_mode()    
    def batched_featurize(self, tokens, masks=None, batch_size=None):
        """
        Batched version of featurize
        """
        if batch_size is None:
            batch_size = tokens.shape[0]
        minibatches_tokens = tokens.split(batch_size)
        minibatches_masks = masks.split(batch_size) if masks is not None else [None] * len(minibatches_tokens)
        all_results = {hook: ([], []) for hook in self.hook_names}
        # Featurize every single minibatch
        for minibatch_tokens, minibatch_masks in tqdm(zip(minibatches_tokens, minibatches_masks)):
            batch_results = self.featurize(minibatch_tokens, minibatch_masks)
            for hook, (indices, acts) in batch_results.items():
                all_results[hook][0].append(indices)
                all_results[hook][1].append(acts)
        # Concatenate all results
        for hook in self.hook_names:
            all_results[hook] = (
                torch.cat(all_results[hook][0], dim=0),
                torch.cat(all_results[hook][1], dim=0)
            )
        return all_results

    def __repr__(self):
        return f"{self.__class__.__name__}(encoders={self.encoders})"


class GenericSaeModule(nn.Module):
    """
    Module to hold SAE parameters
    """

    def __init__(self,d_model,d_sae):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_model)))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_model, d_sae)))
        self.b_dec = nn.Parameter(torch.zeros(d_model))


class EleutherSparseAutoencoder(SparseAutoencoder):
    """ 
    Wrapper class for EleutherAI Top-K SAEs
    """
    
    def __init__(self, model, tokenizer, encoder, hook_name):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            n_features=encoder.encoder.out_features,
            max_k=encoder.cfg.k
        )
        self.encoder = encoder
    
    def reconstruct(self, acts):
        return self.encoder(acts).sae_out

    def encode(self, acts):
        out = self.encoder.encode(acts)
        return out.top_indices, out.top_acts
    
    @staticmethod
    def load_llama3_sae(layer, instruct=True, v2=False, *args, **kwargs):
        # Loading LLaMa3 SAEs trained by Nora Belrose
        # Load the model from huggingface
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct" if instruct else "meta-llama/Meta-Llama-3-8B"
        model, tokenizer = load_hf_model_and_tokenizer(model_name)
        # Load SAE using Eleuther library
        sae_name = "EleutherAI/sae-llama-3-8b-32x-v2" if v2 else "EleutherAI/sae-llama-3-8b-32x"
        sae = Sae.load_from_hub(sae_name, hookpoint=f"layers.{layer}", device="cuda")
        return EleutherSparseAutoencoder(
            model=model,
            tokenizer=tokenizer,
            encoder=sae,
            hook_name=f"model.layers.{layer}", # The SAE reads in the output of this block
            *args, **kwargs
        )


class DeepmindSparseAutoencoder(SparseAutoencoder):
    """ 
    Wrapper class for DeepMind JumpRELU SAE
    """
    
    def __init__(self, model, tokenizer, encoder, hook_name, max_k_features=192):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_name,
            n_features=encoder.W_enc.shape[1],
            max_k=max_k_features
        )
        self.encoder = encoder
    
    def reconstruct(self, acts):
        # Encode the acts in SAE space
        sae_latents = acts @ self.encoder.W_enc + self.encoder.b_enc
        sae_latents = torch.relu(sae_latents) * (sae_latents > self.encoder.threshold)
        return sae_latents @ self.encoder.W_dec + self.encoder.b_dec

    def encode(self, acts):
        # Encode the acts in SAE space
        sae_latents = acts @ self.encoder.W_enc + self.encoder.b_enc
        sae_latents = torch.relu(sae_latents) * (sae_latents > self.encoder.threshold)
        top_sae_latents = sae_latents.topk(self.max_k, dim=-1, sorted=False)
        return top_sae_latents.indices, top_sae_latents.values
    
    @staticmethod
    def load_npz_weights(weight_path, dtype, device):
        state_dict = {}
        with np.load(weight_path) as data:
            for key in data.keys():
                state_dict_key = key
                if state_dict_key.startswith("w_"):
                    # I think that communication between Joseph and us
                    # caused a misconception here, so I'll correct it.
                    state_dict_key = "W_" + state_dict_key[2:]
                if dtype is not None:
                    state_dict[key] = torch.tensor(data[key]).to(dtype=dtype).to(device)
                else:
                    state_dict[key] = torch.tensor(data[key]).to(dtype=dtype).to(device)
        return state_dict
    
    @staticmethod
    def load_gemma2_sae(layer, l0, width=131072, instruct=True, *args, **kwargs):
        # Loading Gemma 2 9b SAEs by Google DeepMind
        # Load the model from huggingface
        model_name = "google/gemma-2-9b-it" if instruct else "google/gemma-2-9b"
        model, tokenizer = load_hf_model_and_tokenizer(model_name)
        # Download/Load the sae 
        bucket_path = f'gs://gemma-2-saes/release-preview/gemmascope-9b-pt-res/layer_{layer}/width_{width//10**3}k/average_l0_{l0}'
        local_path = os.path.join('gemma-2-saes', 'release-preview', 'gemmascope-9b-pt-res', f'layer_{layer}', f'width_{width//10**3}k')
        sae_path = get_bucket_folder(bucket_path, local_path)
        sae_path = os.path.join(sae_path, f'average_l0_{l0}', 'params.npz')
        # Load sae weights into module
        sae = GenericSaeModule(d_model=model.config.hidden_size, d_sae=width).cuda().to(torch.bfloat16)
        sae.load_state_dict(DeepmindSparseAutoencoder.load_npz_weights(sae_path, torch.bfloat16, "cuda"))
        return DeepmindSparseAutoencoder(
            model=model,
            tokenizer=tokenizer,
            encoder=sae,
            hook_name=f"model.layers.{layer}", # The SAE reads in the output of this block
            *args, **kwargs
        )


def expand_latents(latent_indices, latent_acts, n_features):
    """
    Convert N_ctx x K indices in (0, N_features) and N_ctx x K acts into N x N_features sparse tensor
    """
    n_batch, n_pos, _ = latent_indices.shape
    expanded = torch.zeros((n_batch, n_pos, n_features), dtype=latent_acts.dtype)
    expanded.scatter_(-1, latent_indices, latent_acts)
    return expanded

