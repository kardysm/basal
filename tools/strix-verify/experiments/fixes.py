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
