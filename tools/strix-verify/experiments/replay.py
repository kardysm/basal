"""Replay captured run_shared calls: every group of a multi-group call again alone; report the groups whose
probabilities differ (question index within the group, the group's place in its packed chunk)."""
import collections
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
sizes = collections.Counter(len(g) for g, _ in calls)
print(f"{len(calls)} calls; groups per call: {dict(sorted(sizes.items()))}")
bad, checked, again = 0, 0, 0
for ci, (groups, out) in enumerate(calls):
    out = be.run_shared(groups) if FIX else out
    # the same call again: is the backend deterministic for a fixed batch?
    if ci < 20:
        rep = be.run_shared(groups)
        again += sum(not torch.equal(torch.tensor(a), torch.tensor(b)) for x, y in zip(out, rep) for a, b in zip(x, y))
    if len(groups) < 2:
        continue
    units = be._units(groups)
    chunks = be._chunks_tok(units)
    place = {units[k][2]: (c, j, len(idx)) for c, idx in enumerate(chunks) for j, k in enumerate(idx)}
    for gi, g in enumerate(groups):
        checked += 1
        alone = be.run_shared([g])[0]
        diff = [q for q, (a, b) in enumerate(zip(alone, out[gi])) if not torch.equal(torch.tensor(a), torch.tensor(b))]
        if diff:
            bad += 1
            n_tok = [len(units[k][0][0]) for k in range(len(units)) if units[k][2] == gi]
            print(f"call {ci} ({len(groups)} groups) group {gi}: prompts {diff} of {len(alone)} differ; "
                  f"(chunk, place, of) {place[gi]}; unit tokens {n_tok}", flush=True)
print(f"{bad} of {checked} batched groups differ from alone; same call run twice: {again} prompt outputs differ "
      f"(first 20 calls)")
