"""Is a row of the strix attention kernel independent of the other query rows of its block? One subtree of U tokens
(every key visible to every later query), alone (sequence ends at U) vs followed by padding rows; and the same
padded sequence with the padding rows' queries zeroed or rescaled."""
import sys

import torch

from basal.strix import StrixHaloFastBackend

be = StrixHaloFastBackend(sys.argv[1], warm=False)
H, KV, D, BT = be.H, be.KV, be.D, be.ATTN["BT"]
print(f"H={H} KV={KV} D={D} BT={BT} BN={be.ATTN['BN']}")
dev = "cuda"
for U in (855, 715, 961, 1001):
    E = -(-U // 32) * 32  # the next unit would start here
    torch.manual_seed(U)
    q = torch.randn(H, E, D, device=dev, dtype=torch.bfloat16) * 3
    k = torch.randn(KV, E, D, device=dev, dtype=torch.bfloat16) * 3
    v = torch.randn(KV, E, D, device=dev, dtype=torch.bfloat16)
    sub_alone = torch.full((U,), U, device=dev, dtype=torch.int32)
    a = be._attention(q[:, :U].contiguous(), k[:, :U].contiguous(), v[:, :U].contiguous(), sub_alone,
                      *be._key_bounds(sub_alone))
    res = []
    for name, ext in (("pad rows, sub_end U", U), ("pad rows, sub_end extended", E)):
        sub = torch.tensor([ext] * U + [i + 1 for i in range(U, E)], device=dev, dtype=torch.int32)
        for qname, qq in (("pad q random", q), ("pad q zero", torch.cat([q[:, :U], torch.zeros_like(q[:, U:])], 1)),
                          ("pad q x100", torch.cat([q[:, :U], q[:, U:] * 100], 1))):
            o = be._attention(qq.contiguous(), k, v, sub, *be._key_bounds(sub))[:U]
            d = (o != a).any(-1).nonzero().flatten().tolist()
            res.append(f"{name} / {qname}: {'same' if not d else f'rows {d} differ'}")
    print(f"U={U} (tail block {U // BT * BT}..{U // BT * BT + BT - 1}, {E - U} pad rows):")
    for r in res:
        print("   ", r)
