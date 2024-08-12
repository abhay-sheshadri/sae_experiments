import torch
from .probing import LinearClsProbe
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import itertools


def expand_latents(latent_indices, latent_acts, n_features):
    """
    Convert N_ctx x K indices in (0, N_features) and N_ctx x K acts into N x N_features sparse tensor
    """
    n_batch, n_pos, _ = latent_indices.shape
    expanded = torch.zeros((n_batch, n_pos, n_features), dtype=latent_acts.dtype)
    expanded.scatter_(-1, latent_indices, latent_acts)
    return expanded


def squeeze_positions(activations, aggregation="flatten"):
    # Validate the aggregation method
    if aggregation == "max":
        # Take the maximum value across the position dimension (dim=1)
        return activations.amax(dim=1)
    elif aggregation == "mean":
        # Take the mean value across the position dimension (dim=1)
        return activations.mean(dim=1)
    elif aggregation == "flatten":
        # Merge the batch and position dimensions
        # This increases the first dimension size by a factor of sequence_length
        return activations.flatten(0, 1)
    elif aggregation == "last":
        # Select the last token's activation for each sequence in the batch
        return activations[:, -1, :]
    elif aggregation.startswith("index"):
        # index_k returns the kth token in each example.  Last is index_-1
        index = int(aggregation.split("_")[-1])
        return activations[:, index, :]
    else:
        raise NotImplementedError("Invalid method")
    
    
def get_labeled(acts1, acts2, aggregation="max"):
    # Use squeeze_positions to aggregate across the position dimension
    acts1 = squeeze_positions(acts1, aggregation)
    acts2 = squeeze_positions(acts2, aggregation)    
    # Combine the features from both splits
    input_acts = torch.cat([acts1, acts2], dim=0) 
    # Create labels: 0 for split1, 1 for split2
    labels = torch.cat([torch.zeros(acts1.shape[0], dtype=torch.long), torch.ones(acts2.shape[0], dtype=torch.long)])  
    return input_acts, labels


def get_labeled_features(split1, split2, encoder, split, aggregation="max", featurize_batch_size=16):
    # Splits are lists of strings
    split1_indices, split1_acts = encoder.featurize_text(split1, batch_size=featurize_batch_size)[split]
    split2_indices, split2_acts = encoder.featurize_text(split2, batch_size=featurize_batch_size)[split]
    # Convert top k indices, acts to expanded tensor of last dimension n_features
    expanded_split1 = expand_latents(split1_indices, split1_acts, encoder.n_features)[:, 2:, :] # Index out BOS token
    expanded_split2 = expand_latents(split2_indices, split2_acts, encoder.n_features)[:, 2:, :] # Index out BOS token
    features, labels = get_labeled(expanded_split1, expanded_split2, aggregation)
    return features, labels


def get_feature_ranking(features, labels, method="logistic"):
    """
    Find the top features that help distinguish between the distributions
    """
    assert method in ["logistic", "mean_diff", "mutual_info"], "Invalid method"
    if method == "logistic":
        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        # Use sklearn's LogisticRegression
        lr = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr.fit(features_np, labels_np)        
        # Get feature importance
        importance = np.abs(lr.coef_[0])
        ranks = np.argsort(importance)[::-1].tolist()  # Sort in descending order
    elif method == "mean_diff":
        # Mean difference method using PyTorch
        pos_mask = labels == 1
        neg_mask = ~pos_mask
        pos_mean = features[pos_mask].mean(dim=0)
        neg_mean = features[neg_mask].mean(dim=0)
        mean_diff = torch.abs(pos_mean - neg_mean)
        ranks = torch.argsort(mean_diff, descending=True).tolist()
    elif method == "mutual_info":
         # Convert to numpy for scikit-learn
        X = features.numpy()
        y = labels.numpy()
        # Mutual information method
        mi = mutual_info_classif(X, y)
        ranks = np.argsort(mi)[::-1]  # Sort in descending order    
    return ranks


def get_steering_vector(features, labels, method="mean_diff"):
    assert method in ["logistic", "mean_diff", "rep_e", "random"], "Invalid method"
    # Convert features to Float32 if they're not already
    if features.dtype != torch.float32:
        features = features.to(torch.float32)
    # Run method to get steering vector
    if method == "mean_diff":
        # Separate the data into two groups based on labels
        ones = features[labels == 1]
        zeros = features[labels == 0]
        # Compute the average vectors
        avg_ones = ones.mean(dim=0) if ones.numel() > 0 else torch.zeros(features.size(1))
        avg_zeros = zeros.mean(dim=0) if zeros.numel() > 0 else torch.zeros(features.size(1))
        # Compute and return the difference
        steering_vector = avg_ones - avg_zeros
    elif method == "logistic":
        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        # Use sklearn's LogisticRegression
        lr = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr.fit(features_np, labels_np)        
        # Convert the coefficients back to a PyTorch tensor
        steering_vector = torch.from_numpy(lr.coef_[0]).to(features.device)
    elif method == "rep_e":
        # Representation Engineering method
        pos_examples = features[labels == 1]
        neg_examples = features[labels == 0]
        # Generate all pairs of positive and negative examples
        pairs = list(itertools.product(pos_examples, neg_examples))
        # Compute differences for all pairs
        differences = torch.stack([pos - neg for pos, neg in pairs])
        # Perform PCA using SVD
        U, S, Vt = torch.linalg.svd(differences, full_matrices=False)
        # Return the first principal component (first column of Vt.T)
        steering_vector = Vt.T[:, 0]
    elif method == "random":
        # Useful for random steering vector baseline
        steering_vector = torch.randn(features.shape[1], dtype=features.dtype)
    # Return unit steering vector
    return steering_vector / steering_vector.norm()