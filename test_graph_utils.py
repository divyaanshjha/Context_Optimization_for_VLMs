# test_graph_utils.py
# Run with: python test_graph_utils.py

import torch
from trainers.graph_utils import (
    build_class_graph,
    graph_label_smooth,
    graph_laplacian_loss
)

print("=" * 50)
print("TEST 1: build_class_graph")
print("=" * 50)

# Fake CLIP model — just needs encode_text to return something
class FakeCLIP:
    def encode_text(self, tokens):
        # Return random embeddings, simulating CLIP output
        # tokens shape is (K, 77), return (K, 512)
        torch.manual_seed(0)
        K = tokens.shape[0]
        out = torch.randn(K, 512)
        # Make cat and dog similar on purpose
        out[1] = out[0] + 0.01 * torch.randn(512)
        return out

classnames = ["cat", "dog", "car", "airplane", "rose", "tulip"]
device = torch.device("cpu")

# Monkeypatch clip.tokenize for the test
import unittest.mock as mock
with mock.patch("trainers.graph_utils.clip") as mock_clip:
    mock_clip.tokenize.return_value = torch.zeros(6, 77, dtype=torch.long)
    fake_model = FakeCLIP()
    A, A_norm, L = build_class_graph(classnames, fake_model, device, threshold=0.3)

print(f"A shape     : {A.shape}")           # expect (6, 6)
print(f"A_norm shape: {A_norm.shape}")      # expect (6, 6)
print(f"L shape     : {L.shape}")           # expect (6, 6)
print(f"Num edges   : {(A > 0).sum().item()}")
print(f"Diagonal=0  : {A.diag().sum().item() == 0}")   # must be True
print(f"A_norm rows sum to 1 (or 0): {A_norm.sum(dim=1)}")
print("PASS\n")

# ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 2: graph_label_smooth")
print("=" * 50)

targets = torch.tensor([0, 1, 2])           # cat, dog, car
soft = graph_label_smooth(targets, A_norm, num_classes=6, alpha=0.1)

print(f"soft shape: {soft.shape}")          # expect (3, 6)
print(f"rows sum to 1: {soft.sum(dim=1)}")  # must all be 1.0
print(f"cat label (row 0): {soft[0].round(decimals=3)}")
# cat entry should be highest (0.9), 
# dog should get some mass (similar to cat)
assert torch.allclose(soft.sum(dim=1), torch.ones(3), atol=1e-5), \
    "FAIL: rows do not sum to 1"
print("PASS\n")

# ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 3: graph_laplacian_loss")
print("=" * 50)

W = torch.randn(6, 512)                     # fake class weight matrix
loss = graph_laplacian_loss(W, L)

print(f"Loss value : {loss.item():.6f}")
print(f"Is scalar  : {loss.shape == torch.Size([])}")   # must be True
print(f"Is finite  : {torch.isfinite(loss).item()}")    # must be True
print(f"Is >= 0    : {loss.item() >= 0}")               # Laplacian loss always >= 0
assert loss.item() >= 0, "FAIL: Laplacian loss is negative"
print("PASS\n")