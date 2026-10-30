"""Share of outputs where each strix op differs from the stock op of eager bf16, on basal-2.0 layer weights (real
model, gfx1151). Stock ops: F.linear (hipBLASLt), the HF MLP and RMSNorm modules, math SDPA as transformers calls it."""
import sys, torch, torch.nn.functional as F
from basal.strix import StrixHaloFastBackend
be = StrixHaloFastBackend(sys.argv[1])
m = be.model.model
layer = m.layers[10]
lay = be.layers[10]
torch.manual_seed(0)
def diff(a, b): return (a != b).float().mean().item()
for M in (300, 700, 2000):
    x = (torch.randn(M, be.hidden, device="cuda") * 0.5).to(torch.bfloat16)
    with torch.no_grad():
        a = layer.self_attn
        q_ref = torch.cat([a.q_proj(x), a.k_proj(x), a.v_proj(x)], 1)
        r = (torch.randn(M, be.hidden, device="cuda") * 2).to(torch.bfloat16)
        mlp_in = layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
        down_ref = layer.mlp.down_proj(mlp_in) + r
        o_ref = a.o_proj(x) + r
        norm_ref = layer.input_layernorm(x)
        res = {"qkv": diff(be._linear("qkv", x, lay[1], lay[2]), q_ref),
               "o + residual": diff(be._linear("o", x, lay[3], lay[4], r.clone()), o_ref),
               "gate/up/SwiGLU": diff(be._ffn_in(x, lay[6], lay[7], lay[8], lay[9]), mlp_in),
               "down + residual": diff(be._linear("down", mlp_in, lay[10], lay[11], r.clone()), down_ref),
               "RMSNorm": diff(be._norm(x, lay[0]), norm_ref)}
        # attention of one causal prompt: strix kernel vs math SDPA (what eager runs), same q / k / v
        H, KV, D = be.H, be.KV, be.D
        q = torch.randn(H, M, D, device="cuda").to(torch.bfloat16); k = torch.randn(KV, M, D, device="cuda").to(torch.bfloat16)
        v = torch.randn(KV, M, D, device="cuda").to(torch.bfloat16)
        sub = torch.full((M,), M, device="cuda", dtype=torch.int32)
        got = be._attention(q, k, v, sub, *be._key_bounds(sub, [(0, M)]), be._tiles([(0, M)], range(M)))
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.MATH]):
            ref = F.scaled_dot_product_attention(q[None], k.repeat_interleave(H // KV, 0)[None],
                                                 v.repeat_interleave(H // KV, 0)[None], is_causal=True)[0]
        res["attention"] = diff(got, ref.transpose(0, 1).reshape(M, H * D))
    print(f"M={M:5d}  " + "  ".join(f"{k} {100 * v:.2f}%" for k, v in res.items()), flush=True)
