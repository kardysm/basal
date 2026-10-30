"""One captured call whose first group (offset 0, no padding) answers differently alone and batched: record every
kernel's output for that group's rows in both runs and print the first ops that differ, with the differing rows."""
import pickle
import sys

import torch

from basal.strix import StrixHaloFastBackend

import math
import os

import triton

from basal import strix

FIX = os.environ.get("FIX", "")
if FIX == "nofuse":  # A: the attention kernel without FMA contraction
    def _attention(self, q, k, v, sub, kmin, kmax, tok=None):
        H, L, D = q.shape
        n = L if tok is None else tok.shape[0]
        o = torch.empty(n, H * D, device=q.device, dtype=q.dtype)
        c = self.ATTN
        strix._attn_fwd[(triton.cdiv(n, c["BT"]), self.KV)](
            q, k, v, o, sub, kmin, kmax, tok if tok is not None else sub, L, n, o.stride(0), 1.0 / math.sqrt(D),
            HPG=H // self.KV, D=D, BT=c["BT"], BN=c["BN"], GATHER=tok is not None, num_warps=c["num_warps"],
            num_stages=c["num_stages"], enable_fp_fusion=False)
        return o
    StrixHaloFastBackend._attention = _attention
elif FIX == "endpad":  # B: the packed sequence ends on a query-block boundary (BT)
    _pack = StrixHaloFastBackend._pack_chunk

    def _pack_chunk(units, idx, bn, pad_id):
        ids, pos, sub, reads = _pack(units, idx, bn, pad_id)
        off, n = len(ids), (-len(ids)) % StrixHaloFastBackend.ATTN["BT"]
        return ids + [pad_id] * n, pos + [0] * n, sub + [off + i + 1 for i in range(n)], reads
    StrixHaloFastBackend._pack_chunk = staticmethod(_pack_chunk)
elif FIX == "extend":  # C: the padding after a unit lies inside the unit's root subtree (sub_end extended)
    def _pack_chunk(units, idx, bn, pad_id):
        ids, pos, sub, reads, off, start = [], [], [], [], 0, 0
        for k in idx:
            t, pp, sg, last, par = pk = units[k][0]
            if off % bn:
                n = bn - off % bn
                for i in range(start, off):
                    if sub[i] == off:
                        sub[i] = off + n
                ids += [pad_id] * n
                pos += [0] * n
                sub += [off + i + 1 for i in range(n)]
                off += n
            start = off
            ids += t; pos += pp; sub += StrixHaloFastBackend._sub_end(pk, off); reads += [off + i for i in last]
            off += len(t)
        return ids, pos, sub, reads
    StrixHaloFastBackend._pack_chunk = staticmethod(_pack_chunk)
print("fix:", FIX or "none")
be = StrixHaloFastBackend(sys.argv[1], warm=False)
calls = []
with open(sys.argv[2], "rb") as f:
    while True:
        try:
            calls.append(pickle.load(f))
        except EOFError:
            break
groups, _ = calls[int(sys.argv[3])]
gi = int(sys.argv[4])
units = be._units(groups)
first = be._chunks_tok(units)[0][0]
assert units[first][2] == gi, "the traced group must be the first unit of the first chunk"
pk = units[first][0]
n, reads = len(pk[0]), sorted(pk[3])
print(f"group {gi}: {n} tokens, reads at {reads}, sequence alone {n} tokens")

REC, MODE = [], None


def cut(t, rows):
    return t[:rows].detach().to("cpu", copy=True)


def wrap(name, fn, slicer):
    def inner(*a, **kw):
        out = fn(*a, **kw)
        REC.append((name, slicer(out, a, kw)))
        return out
    return inner


def rows_of(out, a, kw):
    return cut(out, n if out.shape[0] >= n else len(reads))


be._linear = wrap("linear", be._linear, lambda o, a, kw: (a[0], cut(o, n if o.shape[0] >= n else len(reads))))
be._ffn_in = wrap("ffn_in", be._ffn_in, rows_of)
be._norm = wrap("norm", be._norm, rows_of)
be._attention = wrap("attention", be._attention,
                     lambda o, a, kw: cut(o, len(reads) if kw.get("tok") is not None else n))
be._qkv_heads = wrap("qkv_heads", be._qkv_heads, lambda o, a, kw: tuple(t[:, :n].detach().cpu() for t in o))

alone_out = be.run_shared([groups[gi]])[0]
alone, REC = REC, []
together_out = be.run_shared(groups)[gi]
together = REC
print("outputs differ at prompts",
      [q for q, (x, y) in enumerate(zip(alone_out, together_out)) if not torch.equal(torch.tensor(x), torch.tensor(y))])
print(f"{len(alone)} ops alone, {len(together)} batched")
shown = 0
for i, ((na, va), (nb, vb)) in enumerate(zip(alone, together)):
    a = va[1] if isinstance(va, tuple) and isinstance(va[0], str) else va
    b = vb[1] if isinstance(vb, tuple) and isinstance(vb[0], str) else vb
    ta, tb = (a, b) if isinstance(a, tuple) else ((a,), (b,))
    for j, (x, y) in enumerate(zip(ta, tb)):
        if x.shape != y.shape:
            print(f"op {i} {na} part {j}: shapes {tuple(x.shape)} vs {tuple(y.shape)}")
            shown += 1
            continue
        if not torch.equal(x, y):
            d = (x != y)
            rows = d.any(-1) if x.dim() == 2 else d.any(0).any(-1)
            r = rows.nonzero().flatten().tolist()
            kind = va[0] if isinstance(va, tuple) and isinstance(va[0], str) else ""
            print(f"op {i} ({i // 8} th layer approx) {na} {kind} part {j}: {int(d.sum())} values differ in rows {r[:12]}"
                  f"{' ...' if len(r) > 12 else ''} (of {x.shape[-2] if x.dim() == 3 else x.shape[0]})")
            shown += 1
    if shown >= 6:
        break
if not shown:
    print("every recorded op is identical for the group's rows")
