"""Call 90 group 0 under fix C: capture the attention inputs of layer LAYER alone and batched, then rerun the kernel on
the batched inputs with one thing changed at a time, and compare the group's last row with the alone output."""
import os
import pickle
import sys

import torch

os.environ.setdefault("FIX", "extend")
exec(open(os.path.join(os.path.dirname(__file__), "fixes.py")).read())  # defines the FIX override and `be`

calls = []
with open(sys.argv[2], "rb") as f:
    while True:
        try:
            calls.append(pickle.load(f))
        except EOFError:
            break
groups, _ = calls[int(sys.argv[3])]
gi, LAYER = int(sys.argv[4]), int(sys.argv[5])
CAP, cnt = [], [0]
orig = be._attention


def att(q, k, v, sub, kmin, kmax, tok=None):
    if tok is None:
        if cnt[0] == LAYER:
            CAP.append([t.clone() for t in (q, k, v, sub)])
        cnt[0] += 1
    return orig(q, k, v, sub, kmin, kmax, tok=tok)


be._attention = att
cnt[0] = 0
be.run_shared([groups[gi]])
cnt[0] = 0
be.run_shared(groups)
be._attention = orig
(qa, ka, va, sa), (qb, kb, vb, sb) = CAP
U = qa.shape[1]
last = int(sys.argv[6]) if len(sys.argv) > 6 else U - 1
assert torch.equal(qa, qb[:, :U]) and torch.equal(ka, kb[:, :U]) and torch.equal(va, vb[:, :U])
ref = be._attention(qa, ka, va, sa, *be._key_bounds(sa))[last]
print(f"U={U}, batched L={qb.shape[1]}, sub of the next rows {sb[U:U + 10].tolist()}")
print(f"pad q: max |q| {qb[:, U:U + 9].abs().max().item():.1f} (unit rows: {qa.abs().max().item():.1f}); "
      f"pad k: {kb[:, U:U + 9].abs().max().item():.1f}; pad v: {vb[:, U:U + 9].abs().max().item():.1f}; "
      f"finite: {bool(torch.isfinite(qb).all() and torch.isfinite(kb).all() and torch.isfinite(vb).all())}")


def run(name, q, k, v, s):
    o = be._attention(q.contiguous(), k.contiguous(), v.contiguous(), s, *be._key_bounds(s))[last]
    print(f"  {name:55s} {'same as alone' if torch.equal(o, ref) else f'{int((o != ref).sum())} values differ'}")


run("batched inputs as they are", qb, kb, vb, sb)
run("batched, cut to the unit (L = U)", qb[:, :U], kb[:, :U], vb[:, :U], sb[:U])
E = U + (-U) % 8
run("batched, cut to the end of the query block", qb[:, :E], kb[:, :E], vb[:, :E], sb[:E])
z = qb.clone(); z[:, U:] = 0
run("batched, pad / later q zeroed", z, kb, vb, sb)
zk = kb.clone(); zk[:, U:] = 0
zv = vb.clone(); zv[:, U:] = 0
run("batched, pad / later k and v zeroed", qb, zk, zv, sb)
run("batched, pad / later q, k and v zeroed", z, zk, zv, sb)
run("batched, pad / later k zeroed only", qb, zk, vb, sb)
run("batched, pad / later v zeroed only", qb, kb, zv, sb)
E32 = -(-U // 32) * 32
zv2 = vb.clone(); zv2[:, U:E32] = 0
run("batched, v zeroed only in the unit's last key block", qb, kb, zv2, sb)
zv3 = vb.clone(); zv3[:, U:E32] = vb[:, U:E32] * 1000
run("batched, v x1000 in the unit's last key block", qb, kb, zv3, sb)
