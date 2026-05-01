# test_gcn.py
# Run with: python test_gcn.py

import torch
from trainers.gcn import ClassGCN, normalize_adj

print("=" * 50)
print("TEST 4: normalize_adj")
print("=" * 50)

# Small 4-class fake adjacency
A = torch.tensor([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
], dtype=torch.float)

A_hat = normalize_adj(A)
print(f"A_hat shape : {A_hat.shape}")       # expect (4, 4)
print(f"A_hat finite: {torch.isfinite(A_hat).all().item()}")
print(f"A_hat:\n{A_hat.round(decimals=3)}")
print("PASS\n")

# ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 5: ClassGCN forward pass")
print("=" * 50)

K = 4          # 4 classes (tiny)
D = 512        # feature dim same as CLIP RN50

gcn = ClassGCN(feat_dim=D, hidden_dim=64)   # smaller hidden for CPU speed
W   = torch.randn(K, D)                     # fake text features

out = gcn(W, A_hat)

print(f"Input  shape : {W.shape}")          # (4, 512)
print(f"Output shape : {out.shape}")        # must be (4, 512) — same shape
print(f"Output finite: {torch.isfinite(out).all().item()}")
print(f"Output normalized (rows ~1.0): {out.norm(dim=-1)}")
# Because gcn() applies F.normalize at end, norms should all be 1.0
assert out.shape == W.shape, "FAIL: GCN changed output shape"
print("PASS\n")

# ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 6: GCN backward pass (gradients flow)")
print("=" * 50)

gcn.zero_grad()
W2    = torch.randn(K, D, requires_grad=True)
out2  = gcn(W2, A_hat)
loss  = out2.sum()
loss.backward()

print(f"W2 grad is not None : {W2.grad is not None}")
for name, p in gcn.named_parameters():
    print(f"  {name} grad norm: {p.grad.norm().item():.6f}")
    assert p.grad is not None, f"FAIL: {name} has no gradient"
print("PASS\n")