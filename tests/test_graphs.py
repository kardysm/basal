"""The CUDA / HIP graph path itself, with the AMD Strix Halo shapes (--mode fast-strix / fast-nocompile-strix,
StrixHaloGraphBackend): shapes captured under the BASAL_* overrides, packed requests (every question x option order
over one state, several requests per replay), rows bisected or run eagerly when longer than the largest captured
length -- every readout equals a separate fp32 forward of its prompt (tiny random Llama on the GPU). The other graph
modes keep their shapes whatever the environment."""
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from basal import bench as bench_module
from basal import server as server_module
from basal.engine import GraphBackend, MPSBackend, StrixHaloGraphBackend

ENV = {"BASAL_LENS": "64,16,32", "BASAL_BATCHES": "1,2,4", "BASAL_TOKEN_BUDGET": "128", "BASAL_WARMUP": "1"}
STRIX = (StrixHaloGraphBackend.LENS, StrixHaloGraphBackend.BATCHES, 4096, 1)


def test_strix_defaults():
    assert StrixHaloGraphBackend.LENS == GraphBackend.LENS + [3584, 4096, 5120, 6144, 8192]
    assert (StrixHaloGraphBackend.TOKEN_BUDGET, StrixHaloGraphBackend.WARMUP) == (4096, 1)
    assert (GraphBackend.LENS[-1], GraphBackend.TOKEN_BUDGET) == (3072, 12288) and not hasattr(GraphBackend, "WARMUP")


def test_shape_config_env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    be = StrixHaloGraphBackend.__new__(StrixHaloGraphBackend)
    be._shape_config()
    assert (be.LENS, be.BATCHES, be.TOKEN_BUDGET, be.WARMUP) == ([16, 32, 64], [1, 2, 4], 128, 1)
    for k in ENV:
        monkeypatch.delenv(k)
    be._shape_config()  # unset: the class defaults, not the overrides of the first call
    assert (be.LENS, be.BATCHES, be.TOKEN_BUDGET, be.WARMUP) == STRIX
    monkeypatch.setenv("BASAL_WARMUP", "0")
    with pytest.raises(ValueError, match="BASAL_WARMUP"):
        be._shape_config()


@pytest.mark.parametrize("name,value,error", [
    ("BASAL_TOKEN_BUDGET", "", None), ("BASAL_WARMUP", " ", None), ("BASAL_LENS", "", None),  # empty = default
    ("BASAL_BATCHES", "2,4", "must include 1"), ("BASAL_LENS", "0,128", "positive"),
    ("BASAL_TOKEN_BUDGET", "-5", "at least 1")])
def test_shape_config_validation(monkeypatch, name, value, error):
    monkeypatch.setenv(name, value)
    be = StrixHaloGraphBackend.__new__(StrixHaloGraphBackend)
    if error is None:
        be._shape_config()
        assert (be.LENS, be.BATCHES, be.TOKEN_BUDGET, be.WARMUP) == STRIX
    else:
        with pytest.raises(ValueError, match=error):
            be._shape_config()


def test_mps_ignores_the_overrides(monkeypatch):
    """The overrides belong to the Strix Halo graph modes: the MPS backend keeps its batching (run on the CPU)."""
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
    monkeypatch.setattr("basal.engine.AutoModelForCausalLM.from_pretrained", lambda _, dtype: tiny().to(dtype))
    be = MPSBackend("tiny", "float32", device="cpu")
    assert (be.LENS, be.BATCHES, be.TOKEN_BUDGET) == (GraphBackend.LENS, GraphBackend.BATCHES, 12288)


class Made:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs, self.tok = args, kwargs, None


@pytest.mark.parametrize("mode,cls,compile", [("fast-strix", "StrixHaloGraphBackend", True),
                                              ("fast-nocompile-strix", "StrixHaloGraphBackend", False),
                                              ("fast-nocompile", "GraphBackend", False), ("fast", "GraphBackend", True)])
def test_modes_select_their_graph_backend(monkeypatch, tmp_path, mode, cls, compile):
    for name in ("GraphBackend", "StrixHaloGraphBackend"):
        monkeypatch.setattr(server_module, name, type(name, (Made,), {}))
        monkeypatch.setattr(bench_module, name, type(name, (Made,), {}))
    monkeypatch.setattr(server_module, "resolve", lambda *_, **kw: tmp_path)
    srv = server_module.Server(server_module.parser().parse_args(["--mode", mode]))
    assert type(srv.backend).__name__ == cls and srv.backend.kwargs["compile"] is compile
    built = bench_module.build(mode, "md")
    assert type(built).__name__ == cls and built.kwargs["compile"] is compile


def test_strix_graph_modes_refuse_fp8(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "fast-strix", "--quant", "fp8"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="not supported"):
        server_module.main()


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU (graph capture)")


class Tok:
    pad_token, eos_token, pad_token_id = "<pad>", "</s>", 0


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, attention_bias=True, mlp_bias=True)
    m = LlamaForCausalLM(cfg).eval()
    m.config._attn_implementation = "sdpa"
    return m


@pytest.fixture(scope="module")
def be():
    m = tiny()
    with pytest.MonkeyPatch.context() as mp:
        for k, v in ENV.items():
            mp.setenv(k, v)
        mp.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
        mp.setattr("basal.engine.AutoModelForCausalLM.from_pretrained", lambda _, dtype: m.to(dtype))
        yield StrixHaloGraphBackend("tiny", dtype="float32", shared=True, warm=True)


def separate(m, toks, lids):
    """Reference: one forward per prompt, full log-softmax over the vocabulary, softmax over the letters."""
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t], device=m.device)).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1).cpu())
    return out


def request(state, n_q, first=20):
    """n_q questions over one state, each in two option orders (the server's layout), letters [7, 8] or [7, 8, 9]."""
    toks, lids = [], []
    for q in range(n_q):
        for o in range(2):
            toks.append(state + [first + q, 21, 30 + o, 31 - o, 40 + q % 9, 41])
            lids.append([7, 8, 9] if q % 2 else [7, 8])
    return toks, lids


def check(be, groups, atol=1e-4):
    res = be.run_shared(groups)
    for (toks, lids), got in zip(groups, res):
        for g, e in zip(got, separate(be.model, toks, lids)):
            assert torch.allclose(torch.tensor(g), e, atol=atol)


@gpu
def test_captured_shapes_follow_the_env(be):
    want = {("s", b, L) for L in (16, 32, 64) for b in (1, 2, 4) if b * L <= 128 or b == 1}
    assert set(be.graphs) == want and be.WARMUP == 1


@gpu
@pytest.mark.parametrize("n_q", [1, 2, 3, 5])
def test_questions_over_one_state_equal_separate(be, n_q):
    """One request, every question x option order in one packed row (state 4 tokens: 1 + 2 x 6 per question fits the
    captured lengths up to 5 questions)."""
    check(be, [request([1, 5, 9, 11], n_q)])


@gpu
def test_requests_batched_into_one_replay(be):
    """Several requests of different lengths go through one replay of a b > 1 graph; each answer equals its separate
    forward, and equals the same request served alone (an answer never depends on the other rows of the batch)."""
    groups = [request([1, 5, 9, 11], 2), request([2, 6], 1), request([3, 4, 12, 13, 14, 15], 3), ([[1, 2, 3, 4]], [[7, 8]])]
    chunks = be._chunks([len(u[0][0]) for u in be._units(groups)])
    assert len(chunks) < len(groups)  # some rows really share a replay
    check(be, groups)
    alone = be.run_shared([groups[2]])[0]
    batched = be.run_shared(groups)[2]
    for x, y in zip(alone, batched):
        assert torch.allclose(torch.tensor(x), torch.tensor(y), atol=1e-5)


@gpu
def test_long_request_is_bisected_and_exact(be):
    """A request whose packed row is longer than the largest captured length (64) is split into smaller rows, each with
    the state again; a single prompt longer than 64 runs eagerly. Both stay exact."""
    toks, lids = request(list(range(1, 21)), 6)  # 20-token state + 6 questions: packed row 20 + 6 x 14 > 64
    units = be._units([(toks, lids)])
    assert len(units) > 1 and all(len(u[0][0]) <= 64 for u in units)
    check(be, [(toks, lids)])
    long_prompt = [[(7 * i) % 47 + 3 for i in range(80)]]
    check(be, [(long_prompt, [[7, 8]]), request([1, 5], 1)])


@gpu
def test_replay_is_deterministic(be):
    g = [request([1, 5, 9, 11], 3), request([2, 6], 2)]
    assert be.run_shared(g) == be.run_shared(g)


@gpu
def test_graph_modes_keep_their_shapes(monkeypatch):
    """With the BASAL_* overrides set, the graph backend of --mode fast / fast-nocompile captures its own shapes."""
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    m = tiny()
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
    monkeypatch.setattr("basal.engine.AutoModelForCausalLM.from_pretrained", lambda _, dtype: m.to(dtype))
    plain = GraphBackend("tiny", dtype="float32", shared=True, warm=True)
    want = {("s", b, L) for L in GraphBackend.LENS for b in GraphBackend.BATCHES if b * L <= 12288 or b == 1}
    assert set(plain.graphs) == want and len(want) == 81
