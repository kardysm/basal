"""Attention with grouped key / value heads at the query lengths of pytorch/pytorch#191937 (fixed in torch 2.14.1, not
in 2.11): SDPA's memory-efficient kernel, which the dispatcher picks for a masked call, returns a wrong last query row
when the keys / values are expanded heads (stride 0) and the query length is 1 mod 32 (head size 256 and 512) or 1
mod 64 (128); no-grad forwards only. basal-omni's Gemma 4 layers in the graph engine pin the math kernel and pass
the grouped heads unexpanded (flex and FlashAttention 3 take them natively); the Llama models' masked passes go through
transformers' repeat_kv, which copies the heads whenever there is more than one key / value head (basal-2.0: 2, 2,
8); with one, basal_attention copies them itself (repeat_kv's stride-0 view gave fp32 last rows off by ~3 on an H100).
Each is compared with fp32 math over heads repeated into contiguous memory. GPU only."""
import pytest
import torch

from basal.engine import attention_available
from basal.gemma4_graph import Gemma4Graph

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _reference(q, k, v, mask):
    """fp32 math attention, key / value heads repeated into contiguous memory: [b, H, L, d]."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    r = q.shape[1] // k.shape[1]
    k, v = (x.float().repeat_interleave(r, 1).contiguous() for x in (k, v))
    with sdpa_kernel([SDPBackend.MATH]):
        return torch.nn.functional.scaled_dot_product_attention(q.float(), k, v, attn_mask=mask, scale=1.0)


def _qkv(L, d, H, Hkv, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(L * d)
    q, k, v = (torch.randn(1, h, L, d, device="cuda", dtype=dtype, generator=g) for h in (H, Hkv, Hkv))
    return q / q.norm(dim=-1, keepdim=True) * 4, k / k.norm(dim=-1, keepdim=True) * 4, v


@cuda
@pytest.mark.parametrize("L", [33, 289, 1857])
@pytest.mark.parametrize("d, H, Hkv, W", [(256, 16, 8, 1024), (512, 16, 1, None), (512, 8, 2, None)])
def test_gemma4_attention_at_the_issue_lengths(L, d, H, Hkv, W):
    """Gemma4Graph._attend with each kernel it can run here (math with a boolean mask, flex with a block mask,
    FlashAttention 3 with the window for 256-wide heads on Hopper), every query row and the last one, against fp32
    math with the heads repeated."""
    q, k, v = _qkv(L, d, H, Hkv)
    i = torch.arange(L, device="cuda")
    allow = (i[None] <= i[:, None]) & ((i[:, None] - i[None] < W) if W else True)
    ref = _reference(q, k, v, allow)
    g = Gemma4Graph.__new__(Gemma4Graph)
    g.window = W or 1
    t = "sliding_attention" if W else "full_attention"
    masks = {"math": allow[None, None]}
    if attention_available("flex") is None:
        masks["flex"] = g._plain_block(L, t, "cuda")
    if W and attention_available("fa3") is None:
        masks["fa3"] = ("fa3", W)
    for kernel, m in masks.items():
        o = Gemma4Graph._attend(q, k, v, m).view(1, L, H, d).transpose(1, 2).float()
        err = (o - ref).abs()
        assert float(err.max()) < 0.05, (kernel, float(err.max()))
        assert float(err[:, :, -1].max()) < 0.05, (kernel, "last row", float(err[:, :, -1].max()))


@cuda
@pytest.mark.parametrize("L", [65, 129, 1857])
@pytest.mark.parametrize("Hkv", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_llama_masked_attention_at_the_issue_lengths(L, Hkv, dtype):
    """The masked pass of the Llama models (pass 2 of a branched unit, fp32, the long lane) through basal_attention
    -> transformers' sdpa_attention_forward with the dispatcher's own kernel choice, head size 128, against fp32
    math with the heads repeated."""
    from types import SimpleNamespace

    from basal.engine import basal_attention
    H = 16
    q, k, v = _qkv(L, 128, H, Hkv, dtype)
    i = torch.arange(L, device="cuda")
    mask = torch.where(i[None] <= i[:, None], 0.0, torch.finfo(dtype).min).to(dtype)[None, None]
    module = SimpleNamespace(num_key_value_groups=H // Hkv, layer_idx=0, is_causal=True)
    with torch.no_grad():
        o, _ = basal_attention(module, q, k, v, mask, scaling=1.0)
    ref = _reference(q, k, v, mask.float())
    err = (o.transpose(1, 2).float() - ref).abs()
    tol = 0.05 if dtype == torch.bfloat16 else 1e-4
    assert float(err.max()) < tol and float(err[:, :, -1].max()) < tol, float(err.max())
