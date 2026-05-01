# test_forward_backward.py
# Run with: python test_forward_backward.py

import torch
import torch.nn.functional as F
from trainers.graph_utils import graph_label_smooth, graph_laplacian_loss
from trainers.gcn import ClassGCN, normalize_adj

print("=" * 50)
print("TEST 7: Full forward_backward logic (CPU mock)")
print("=" * 50)

# ── Fake dimensions (tiny so CPU is fast) ────────────────────────────────────
B   = 4     # batch size
K   = 6     # number of classes
D   = 64    # feature dim (normally 512, use 64 for speed)

torch.manual_seed(42)

# ── Fake graph ───────────────────────────────────────────────────────────────
A = torch.zeros(K, K)
# Manually add a few edges
A[0,1] = A[1,0] = 0.8    # class 0 and 1 are similar
A[4,5] = A[5,4] = 0.6    # class 4 and 5 are similar
A_norm = A / A.sum(dim=1, keepdim=True).clamp(min=1e-6)
D_mat  = torch.diag(A.sum(dim=1))
L      = D_mat - A
A_hat  = normalize_adj(A)

# ── Fake model components ────────────────────────────────────────────────────
image_features = F.normalize(torch.randn(B, D), dim=-1)  # (B, D)
text_features  = F.normalize(torch.randn(K, D), dim=-1)  # (K, D)
text_features.requires_grad_(True)

logit_scale    = torch.tensor(4.6052)                     # ~ log(100), CLIP default
label          = torch.randint(0, K, (B,))

gcn            = ClassGCN(feat_dim=D, hidden_dim=32)

# ── Configs to test ──────────────────────────────────────────────────────────
configs = [
    {"use_gcn": False, "alpha_smooth": 0.0,  "lambda_lap": 0.0,  "name": "baseline"},
    {"use_gcn": False, "alpha_smooth": 0.1,  "lambda_lap": 0.0,  "name": "label_smooth"},
    {"use_gcn": False, "alpha_smooth": 0.0,  "lambda_lap": 0.01, "name": "laplacian"},
    {"use_gcn": False, "alpha_smooth": 0.1,  "lambda_lap": 0.01, "name": "smooth+lap"},
    {"use_gcn": True,  "alpha_smooth": 0.1,  "lambda_lap": 0.01, "name": "full_graph"},
]

for cfg in configs:
    print(f"\n  Config: {cfg['name']}")
    
    tf = text_features.detach().clone().requires_grad_(True)

    # Step 3: GCN
    if cfg["use_gcn"]:
        tf = gcn(tf, A_hat)

    # Step 4: logits
    logits = logit_scale.exp() * (image_features @ tf.T)    # (B, K)

    # Step 5: CE or smooth CE
    if cfg["alpha_smooth"] > 0:
        soft = graph_label_smooth(label, A_norm, K, cfg["alpha_smooth"])
        log_probs = F.log_softmax(logits, dim=-1)
        loss_ce = -(soft * log_probs).sum(dim=-1).mean()
    else:
        loss_ce = F.cross_entropy(logits, label)

    # Step 6: Laplacian
    if cfg["lambda_lap"] > 0:
        loss_lap = graph_laplacian_loss(tf, L)
    else:
        loss_lap = torch.tensor(0.0)

    # Step 7: total
    loss = loss_ce + cfg["lambda_lap"] * loss_lap

    # Step 8: backward
    loss.backward()

    print(f"    loss_ce  : {loss_ce.item():.4f}")
    print(f"    loss_lap : {loss_lap.item():.6f}")
    print(f"    loss_tot : {loss.item():.4f}")
    print(f"    finite   : {torch.isfinite(loss).item()}")

    assert torch.isfinite(loss), "FAIL: loss is not finite"
    print(f"    PASS")

print("\nAll tests passed!")