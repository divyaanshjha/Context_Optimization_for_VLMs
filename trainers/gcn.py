# trainers/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, X, A_hat):
        """
        A_hat: symmetrically normalized adjacency + self-loops
               = D^{-1/2} (A + I) D^{-1/2}
        X    : (K, in_dim) node features
        """
        return F.relu(self.linear(A_hat @ X))


class ClassGCN(nn.Module):
    """
    Two-layer GCN that refines class weight vectors using the class graph.
    Input:  raw text encoder outputs  (K, D)
    Output: refined class embeddings  (K, D)
    """
    def __init__(self, feat_dim=512, hidden_dim=256):
        super().__init__()
        self.gc1 = GraphConvLayer(feat_dim, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, feat_dim)

    def forward(self, W, A_hat):
        h = self.gc1(W, A_hat)
        out = self.gc2(h, A_hat)
        # Residual: add original embeddings so GCN only learns corrections
        return F.normalize(W + out, dim=-1)


def normalize_adj(A):
    """
    Compute D^{-1/2} (A + I) D^{-1/2} — standard GCN normalization
    """
    A_hat = A + torch.eye(A.size(0), device=A.device)
    D = A_hat.sum(dim=1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt