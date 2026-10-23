"""MLX backend: equals the PyTorch readout (tiny random Llama with the basal layout: attention and MLP biases, untied
head, grouped KV heads, transformers-5 `rope_parameters`), with a shared prefix (SOAM group), for unrelated prompts and
with 8-bit weights made at load time (mlx-q8)."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal.engine import MLXBackend, mlx_config_overrides


def tiny_llama(path, rope=None):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=256, attention_bias=True,
                      mlp_bias=True, tie_word_embeddings=False,
                      rope_parameters=rope or {"rope_type": "default", "rope_theta": 1000000.0})
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # random biases (initialised to zero), so that a dropped bias would be noticed
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.1)
            elif n.endswith(("q_proj.weight", "k_proj.weight")):
                p.mul_(8.0)  # peaky attention, so that positions (rotary settings) matter
    m.save_pretrained(path)
    return m


def reference(m, toks, lids):
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1).tolist())
    return out


class _Tok:
    pad_token_id = 0


def mlx_stub(path, quant=None, packed=False):
    from concurrent.futures import ThreadPoolExecutor
    be = MLXBackend.__new__(MLXBackend)  # no tokenizer files in the tiny model directory
    be.pool, be.tok, be.max_rows = ThreadPoolExecutor(max_workers=1), _Tok(), 3
    be.pool.submit(be._load, path, quant, packed).result()
    return be


def test_rope_parameters_override():
    assert mlx_config_overrides({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}) == {"rope_theta": 1e6}
    assert mlx_config_overrides({"rope_theta": 5e5}) == {}
    o = mlx_config_overrides({"rope_parameters": {"rope_type": "llama3", "rope_theta": 5e5, "factor": 8.0}})
    assert o["rope_theta"] == 5e5 and o["rope_scaling"]["rope_type"] == "llama3"


STATE = list(range(21, 61)) * 3  # 120 shared tokens
# SOAM group: state | question 1 (two orders) | question 2 (two orders) | question 3, suffixes of unequal length
TOKS = [STATE + [5, 6, 10, 11], STATE + [5, 6, 11, 10], STATE + [7, 8, 9, 10, 11, 12], STATE + [7, 8, 9, 12, 11, 10],
        STATE + [3]]
LIDS = [[10, 11], [10, 11], [10, 11, 12], [10, 11, 12], [10, 11, 12, 13]]


def test_mlx_equals_torch(tmp_path):
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    be = mlx_stub(tmp_path)
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    # unrelated prompts of different lengths (right-padded batches, no shared prefix), and a mix of both kinds
    singles = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    ref_s = reference(m, singles, sl)
    got_s = be.run(singles, sl)
    for r, g in zip(ref_s, got_s):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    mixed = be.run_shared([([singles[0]], [sl[0]]), (TOKS[:2], LIDS[:2])])
    assert max(abs(x - y) for x, y in zip(mixed[0][0], ref_s[0])) < 2e-4
    assert max(abs(x - y) for x, y in zip(mixed[1][1], ref[1])) < 2e-4


def test_mlx_without_override_differs(tmp_path):
    """Guard for the override: with mlx-lm's default rope_theta (10000) the readout changes."""
    pytest.importorskip("mlx.core")
    from mlx_lm.utils import load_model
    m = tiny_llama(tmp_path)
    toks, lids = [list(range(21, 61)) * 3 + [5]], [[10, 11, 12, 13]]
    be = mlx_stub(tmp_path)
    be.model, _ = load_model(tmp_path)  # no override
    assert max(abs(x - y) for x, y in zip(reference(m, toks, lids)[0], be.run(toks, lids)[0])) > 1e-3


def test_mlx_q8_quantises_decoder_blocks_only(tmp_path):
    pytest.importorskip("mlx.core")
    import mlx.nn as nn
    m = tiny_llama(tmp_path)
    be = mlx_stub(tmp_path, "q8")
    assert be.quantization == {"group_size": 64, "bits": 8}
    assert isinstance(be.model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(be.model.model.layers[1].mlp.down_proj, nn.QuantizedLinear)
    assert not isinstance(be.model.lm_head, nn.QuantizedLinear)
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):  # 8-bit weights: close to bf16 / fp32, not equal
        assert max(abs(x - y) for x, y in zip(r, g)) < 0.05
    with pytest.raises(ValueError, match="unknown MLX quantisation"):
        mlx_stub(tmp_path, "q4")


def test_mlx_reads_255_format2_codes(tmp_path):
    """Prompt format 2: up to 255 option codes, read from the head's logits at the prefill position (the code ids of the
    255-option test vector, docs/format2_vectors.jsonl), for a SOAM group of a 255-option question in two orders and a
    2-option question over the same state, against the PyTorch readout."""
    pytest.importorskip("mlx.core")
    import json
    from pathlib import Path
    lines = (Path(__file__).parents[1] / "docs/format2_vectors.jsonl").read_text(encoding="utf-8").splitlines()
    vec = next(v for v in map(json.loads, lines) if v["id"] == "en-choice-255")
    codes = vec["prompts"][0]["code_ids"]
    assert len(codes) == 255 and len(set(codes)) == 255
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=32000, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=1024, attention_bias=True,
                      mlp_bias=True, tie_word_embeddings=False,
                      rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.1)
        m.lm_head.weight.mul_(20.0)  # spread the code logits, so that a wrong column or order would show
    m.save_pretrained(tmp_path)
    be = mlx_stub(tmp_path)
    state = list(range(100, 400))
    toks = [state + list(range(500, 760)) + [7], state + list(range(760, 500, -1)) + [7], state + [9, 8, 7]]
    lids = [codes, codes, codes[:2]]
    ref = reference(m, toks, lids)
    got = be.run_shared([(toks, lids)])[0]
    assert [len(g) for g in got] == [255, 255, 2]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
        assert abs(sum(g) - 1) < 1e-5


def test_mlx_code_logits_in_fp32(tmp_path):
    """bf16 weights: the code logits are the fp32 product of the hidden state and the head's code rows (as the PyTorch
    backends since the fp32 readout), not the bf16 head output, whose logits of magnitude 16-32 step by 0.125."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import numpy as np
    m = tiny_llama(tmp_path)
    with torch.no_grad():
        m.lm_head.weight.mul_(40.0)
    m.to(torch.bfloat16).save_pretrained(tmp_path)
    be = mlx_stub(tmp_path)
    assert be.model.lm_head.weight.dtype == mx.bfloat16
    h = mx.array(np.random.default_rng(0).normal(size=(3, 64)).astype(np.float32)).astype(mx.bfloat16)
    lids = [[10, 11], [10, 11, 12], [13, 14, 15, 16]]
    got = be._letters(h, lids)
    W = np.array(be.model.lm_head.weight.astype(mx.float32))
    hf = np.array(h.astype(mx.float32))
    for g, ids, x in zip(got, lids, hf):
        z = W[ids] @ x
        ref = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        assert max(abs(a - b) for a, b in zip(g, ref)) < 1e-5
    z16 = np.array(be.model.lm_head(h).astype(mx.float32))
    assert np.abs(z16[0, [10, 11]] - W[[10, 11]] @ hf[0]).max() > 1e-3  # the bf16 head output is coarser


def tiny_gemma4(path):
    """A tiny random Gemma 4 decoder with the parts of basal-omni (Gemma 4 E4B): sliding-window and global layers
    (wider global heads, partial "proportional" rotary), per-layer inputs, KV sharing of the last layers, tied head and
    the final logit softcap; the head is scaled so that the softcap changes the code probabilities."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                           sliding_window=16, layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
                           vocab_size_per_layer_input=64, hidden_size_per_layer_input=8, num_kv_shared_layers=2,
                           final_logit_softcapping=30.0, max_position_embeddings=1024,
                           rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                                                               "rope_theta": 1000000.0},
                                            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}})
    m = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(300.0)  # tied head: large code logits, which the softcap bends
    m.save_pretrained(path)
    return m


def test_mlx_gemma4_equals_torch(tmp_path):
    """Gemma 4 (basal-omni): the readout applies the final logit softcap, and a shared prefix longer than the sliding
    window (120 tokens, window 16) keeps every position (mlx-lm's rotating cache would keep the last window only)."""
    pytest.importorskip("mlx.core")
    m = tiny_gemma4(tmp_path)
    be = mlx_stub(tmp_path)
    assert be.softcap == 30.0
    ref = reference(m, TOKS, LIDS)
    got = be.run_shared([(TOKS, LIDS)])[0]
    for r, g in zip(ref, got):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    singles = [list(range(1, 40)), list(range(5, 64)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    for r, g in zip(reference(m, singles, sl), be.run(singles, sl)):
        assert max(abs(x - y) for x, y in zip(r, g)) < 2e-4
    be.softcap = None  # guard: without the softcap the probabilities differ
    assert max(abs(x - y) for r, g in zip(ref, be.run_shared([(TOKS, LIDS)])[0]) for x, y in zip(r, g)) > 1e-2


def test_gemma4_per_layer_config_override():
    """transformers 5 writes Gemma 4's global head size in per_layer_config (basal-omni-1.0 config.json); mlx-lm
    reads global_head_dim, for the decoder and inside the text_config of the multimodal wrapper."""
    t = {"layer_types": ["sliding_attention", "full_attention"], "head_dim": 256,
         "per_layer_config": {"01": {"head_dim": 512}}, "rope_parameters": {"full_attention": {}, "sliding_attention": {}}}
    assert mlx_config_overrides(t) == {"global_head_dim": 512}
    o = mlx_config_overrides({"model_type": "gemma4", "text_config": t})
    assert o["text_config"]["global_head_dim"] == 512 and o["text_config"]["head_dim"] == 256
    assert mlx_config_overrides({"text_config": {**t, "global_head_dim": 512}}) == {}


def test_mlx_packed_rows_equal_kv_cache(tmp_path):
    """--mlx-packed, Llama: a group whose prefix and suffixes fit in PACK_TOKENS runs as one packed row (explicit
    positions, block mask), several groups of different lengths in one padded batch; the readouts equal the KV-cache
    path (PACK_TOKENS = 0) and the PyTorch readout. Without the flag (the default) and for Gemma 4 the KV cache stays."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path)
    assert not mlx_stub(tmp_path).packed  # opt-in
    be = mlx_stub(tmp_path, packed=True)
    assert be.packed
    q2 = [list(range(30, 60)) + [4, 10, 11], list(range(30, 60)) + [4, 11, 10]]  # a shorter question, two orders
    groups = [(TOKS, LIDS), (q2, LIDS[:2]), (TOKS[2:4], LIDS[2:4])]
    calls = []
    packed_fwd = be._packed
    be._packed = lambda g: calls.append(len(g)) or packed_fwd(g)
    got = be.run_shared(groups)
    assert calls == [3]  # one forward for the three groups
    be.PACK_BUDGET = 100  # rows longer than a third of the budget: one forward each, the same readouts
    alone = be.run_shared(groups)
    assert calls == [3, 1, 1, 1]
    assert max(abs(a - b) for g, h in zip(got, alone) for x, y in zip(g, h) for a, b in zip(x, y)) < 1e-5
    calls.clear()
    be.PACK_BUDGET = MLXBackend.PACK_BUDGET
    be.PACK_TOKENS = 64  # between q2's packed row (35 tokens) and the others (over 120): q2 packs, the rest fall back
    mixed = be.run_shared(groups)
    assert calls == [1]
    assert max(abs(a - b) for g, h in zip(got, mixed) for x, y in zip(g, h) for a, b in zip(x, y)) < 1e-5
    calls.clear()
    be.PACK_TOKENS = 0
    cached = be.run_shared(groups)
    assert calls == []  # PACK_TOKENS = 0: every group through _group (_shared_len, KV cache)
    assert len(got) == len(cached) == len(groups)
    for (toks, lids), g, c in zip(groups, got, cached):
        ref = reference(m, toks, lids)
        assert len(g) == len(c) == len(ref) == len(toks)
        for r, x, y, ids in zip(ref, g, c, lids):
            assert len(r) == len(x) == len(y) == len(ids)
            assert max(abs(a - b) for a, b in zip(x, y)) < 1e-5
            assert max(abs(a - b) for a, b in zip(r, x)) < 2e-4
    tiny_gemma4(tmp_path / "g4")
    assert not mlx_stub(tmp_path / "g4", packed=True).packed


@pytest.mark.parametrize("rope, packed", [
    ({"rope_type": "linear", "rope_theta": 1e6, "factor": 2.0}, True),
    ({"rope_type": "llama3", "rope_theta": 1e6, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
      "original_max_position_embeddings": 64}, True),
    ({"rope_type": "dynamic", "rope_theta": 1e6, "factor": 2.0}, False),  # base depends on the length of the call
    ({"rope_type": "yarn", "rope_theta": 1e6, "factor": 2.0, "original_max_position_embeddings": 64}, False),
])
def test_mlx_packed_needs_static_rope(tmp_path, rope, packed):
    """--mlx-packed rotates each token as its own length-1 row: only RoPE layers whose frequencies do not depend on the
    sequence length are packed; the others (dynamic NTK, yarn, longrope) fall back to the KV cache, with the same
    readouts as PyTorch."""
    pytest.importorskip("mlx.core")
    m = tiny_llama(tmp_path, rope)
    be = mlx_stub(tmp_path, packed=True)
    assert be.packed is packed
    groups = [(TOKS[:2], LIDS[:2])]
    got = be.run_shared(groups)
    for r, x in zip(reference(m, *groups[0]), got[0]):
        assert max(abs(a - b) for a, b in zip(r, x)) < 2e-4
