# trainers/graph_utils.py
import torch
import numpy as np
import clip

def build_class_graph(classnames, clip_model, device, threshold=0.3):
    """
    Build semantic similarity graph from CLIP text embeddings of class names.
    
    Args:
        classnames: list of class name strings
        clip_model: frozen CLIP model
        device: torch device
        threshold: cosine similarity threshold for adding an edge
    
    Returns:
        A     : adjacency matrix (K x K), raw similarities above threshold
        A_norm: row-normalized adjacency (for label propagation)
        L     : graph Laplacian D - A (for Laplacian regularization)
    """
    K = len(classnames)
    
    # Get text embeddings for each class name using CLIP's tokenizer
    with torch.no_grad():
        texts = clip.tokenize(classnames).to(device)
        text_features = clip_model.encode_text(texts)          # (K, 512)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )                                                        # L2 normalize
    
    # Cosine similarity matrix
    sim_matrix = (text_features @ text_features.T).float()      # (K, K)
    
    # Threshold: keep only edges above threshold, zero diagonal
    A = sim_matrix.clone()
    A[A < threshold] = 0.0
    A.fill_diagonal_(0.0)
    
    # Row-normalized adjacency for label propagation
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    A_norm = A / row_sum
    
    # Graph Laplacian
    D = torch.diag(A.sum(dim=1))
    L = D - A
    
    return A.to(device), A_norm.to(device), L.to(device)


def graph_label_smooth(targets, A_norm, num_classes, alpha=0.1):
    B = targets.size(0)
    K = num_classes

    one_hot = torch.zeros(B, K, device=targets.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    neighbor_mass = one_hot @ A_norm.T                        # (B, K)

    # ── FIX: detect isolated classes (row sum = 0 in A_norm) ──────────────
    # For a target class with no neighbors, neighbor_mass row is all zeros
    # → smoothing would reduce the row sum below 1.0
    # Solution: only apply smoothing where neighbors actually exist
    
    target_row_sums = A_norm.sum(dim=1)                       # (K,)
    has_neighbors = target_row_sums[targets] > 0              # (B,) bool

    # alpha_effective is 0.1 for classes with neighbors, 0.0 for isolated ones
    alpha_eff = torch.where(
        has_neighbors,
        torch.full((B,), alpha, device=targets.device),
        torch.zeros(B, device=targets.device)
    )                                                          # (B,)

    alpha_eff = alpha_eff.unsqueeze(1)                        # (B, 1) for broadcasting

    soft_targets = (1 - alpha_eff) * one_hot + alpha_eff * neighbor_mass

    return soft_targets


def graph_laplacian_loss(weight_matrix, L, normalize=True):
    """
    Compute Tr(W^T L W) — encourages similar classes to have similar
    weight vectors in the embedding space.
    
    Args:
        weight_matrix : (K, D) class weight vectors from text encoder
        L             : (K, K) graph Laplacian
        normalize     : divide by K*D for scale invariance
    
    Returns:
        scalar loss term
    """
    # Tr(W^T L W) = sum over all edges (i,j) of A_ij * ||w_i - w_j||^2
    loss = torch.trace(weight_matrix.T @ L @ weight_matrix)
    if normalize:
        K, D = weight_matrix.shape
        loss = loss / (K * D)
    return loss