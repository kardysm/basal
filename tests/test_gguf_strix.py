"""--mode gguf-strix (StrixHaloGGUFBackend): the fixed start of every prompt is decoded once into sequences of its own
and shared by every row that begins with it; every readout equals a full decode. Plain --mode gguf keeps its decode
(no cached sequences, the whole memory cleared per chunk). Tiny random Llama written as GGUF, CPU."""
import sys

import pytest

pytest.importorskip("llama_cpp")
pytest.importorskip("gguf")

import numpy as np

from basal import bench as bench_module
from basal import server as server_module
from basal.engine import GGUFBackend, StrixHaloGGUFBackend
from tests.test_gguf import IDS, Tok, group, model  # noqa: F401  (model: the tiny GGUF fixture)

PREFIXES = [Tok()("a shared state").input_ids, Tok()("other start").input_ids]
STATE = "a shared state of a request, long enough to matter: "
PROMPTS = [STATE + q + o for q in ("q1? ", "question two? ") for o in (" A. x B. y C. z", " A. z B. y C. x")]


@pytest.fixture(scope="module")
def backend(model):
    gguf_path, model_dir = model
    with pytest.MonkeyPatch.context() as mp:  # stand-in tokenizer instead of a Hugging Face model directory
        mp.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
        yield GGUFBackend(model_dir, gguf_path, n_gpu_layers=0)  # plain gguf: the full-decode reference


@pytest.fixture(scope="module")
def cached(model):
    gguf_path, model_dir = model
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
        yield StrixHaloGGUFBackend(model_dir, gguf_path, n_gpu_layers=0, prefixes=PREFIXES)


def close(a, b):
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=5e-3)


def test_cached_prefix_equals_full_decode(backend, cached):
    """Rows that begin with either cached prefix, and a row that does not, give the readouts of a full decode; the
    second round reuses the prefixes after the first chunk's readout sequences were removed."""
    groups = [(PROMPTS, [IDS] * 4), group("other start; question?"), group("no cached prefix here")]
    assert [cached._cached(cached._pack(cached.encode(g[0])))[0] for g in groups] == [14, 11, 0]
    for _ in range(2):
        close(cached.run_shared(groups), backend.run_shared(groups))


def test_plain_gguf_keeps_its_decode(backend, monkeypatch):
    """--mode gguf is unchanged: a context of N_CTX cells and N_SEQ sequences, no cached prefixes, the whole memory
    cleared before every chunk and no sequence copied or removed."""
    C = backend.C
    assert not hasattr(backend, "prefixes")
    assert C.llama_n_ctx(backend.ctx) == GGUFBackend.N_CTX and C.llama_n_seq_max(backend.ctx) == GGUFBackend.N_SEQ
    calls = []
    for name in ("llama_memory_clear", "llama_memory_seq_cp", "llama_memory_seq_rm"):
        real = getattr(C, name)
        monkeypatch.setattr(C, name, lambda *a, _n=name, _f=real: (calls.append(_n), _f(*a))[1])
    backend.run_shared([(PROMPTS, [IDS] * 4), group("other start; question?")])
    assert calls and set(calls) == {"llama_memory_clear"}


def test_gguf_strix_without_prefixes_equals_gguf(model, backend, monkeypatch):
    gguf_path, model_dir = model
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
    off = StrixHaloGGUFBackend(model_dir, gguf_path, n_gpu_layers=0, prefixes=[])
    groups = [(PROMPTS, [IDS] * 4), group("no cached prefix here")]
    close(off.run_shared(groups), backend.run_shared(groups))


def test_cached_rows_use_the_whole_context(model, backend, monkeypatch):
    """The cached prefixes have cells of their own: a row is charged only for the tokens it adds, so a row that starts
    with a cached prefix may add N_CTX tokens, a row without one may have N_CTX tokens, and one more is refused."""
    gguf_path, model_dir = model
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())

    class Small(StrixHaloGGUFBackend):
        N_CTX = 128  # llama.cpp needs n_batch (= N_CTX) >= the sequences (64 readouts + 2 prefixes)
    small = Small(model_dir, gguf_path, n_gpu_layers=0, prefixes=PREFIXES)
    P = len(PREFIXES[0])
    long_cached = "a shared state" + "x" * 128  # P + 128 tokens: 128 new cells
    for prompts in ([long_cached], ["y" * 128]):
        close(small.run_shared([(prompts, [IDS])])[0], backend.run_shared([(prompts, [IDS])])[0])
    assert small._cost(small._pack(small.encode([long_cached]))) == 128 and len(long_cached) == P + 128
    with pytest.raises(ValueError, match="prompt too long"):
        small.run_shared([(["y" * 129], [IDS])])


def test_readouts_never_reach_the_prefix_sequences(backend, cached):
    """N_SEQ raised after start-up: readouts stay within the sequences made at start-up (the cached prefixes follow
    them), so the prefixes survive and the answers stay right."""
    prompts = [f"a shared state; option set {k:02d}" for k in range(70)]
    cached.N_SEQ = 200
    try:
        assert cached._max_read() == 64
        got = cached.run_shared([(prompts, [IDS] * 70)])[0]
    finally:
        del cached.N_SEQ
    close(got, backend.run_shared([(prompts, [IDS] * 70)])[0])
    g = [group("other start; question?")]
    close(cached.run_shared(g)[0], backend.run_shared(g)[0])


def test_failure_half_way_leaves_no_stale_cells(backend, cached, monkeypatch):
    """A failure while the prefix cells are being shared (after some readouts got them) must not break the next call:
    every readout sequence of the failed chunk is removed before the next one."""
    C = cached.C
    calls, real = [], C.llama_memory_seq_cp

    def flaky(*a):
        calls.append(a)
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return real(*a)
    monkeypatch.setattr(C, "llama_memory_seq_cp", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        cached.run_shared([(PROMPTS, [IDS] * 4)])
    monkeypatch.setattr(C, "llama_memory_seq_cp", real)
    for g in ([group("no cached prefix here")], [(PROMPTS, [IDS] * 4)]):  # rows 0 and 1 got prefix cells above
        close(cached.run_shared(g), backend.run_shared(g))


# -- mode wiring: opt-in by name only ----------------------------------------------------------------------------------
class Made:
    def __init__(self, *args, **kwargs):
        self.args, self.tok = args, None


@pytest.mark.parametrize("mode,cls", [("gguf", "GGUFBackend"), ("gguf-strix", "StrixHaloGGUFBackend")])
def test_server_mode_selects_its_backend(monkeypatch, tmp_path, mode, cls):
    for name in ("GGUFBackend", "StrixHaloGGUFBackend"):
        monkeypatch.setattr(server_module, name, type(name, (Made,), {}))
    monkeypatch.setattr(server_module, "resolve", lambda *_, **kw: tmp_path)
    srv = server_module.Server(server_module.parser().parse_args(["--mode", mode, "--gguf", "w.gguf"]))
    assert type(srv.backend).__name__ == cls and srv.backend.args == (tmp_path, "w.gguf")


def test_gguf_strix_mode_needs_a_file_before_model_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "gguf-strix"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--mode gguf-strix needs --gguf"):
        server_module.main()


def test_bench_modes_select_their_backend(monkeypatch):
    for name in ("GGUFBackend", "StrixHaloGGUFBackend"):
        monkeypatch.setattr(bench_module, name, type(name, (Made,), {}))
    assert type(bench_module.build("gguf@a.gguf", "md")).__name__ == "GGUFBackend"
    assert type(bench_module.build("gguf-strix@a.gguf", "md")).__name__ == "StrixHaloGGUFBackend"
    assert bench_module.build("gguf-strix", "md", gguf="b.gguf").args == ("md", "b.gguf")
    with pytest.raises(SystemExit, match="gguf-strix needs --gguf"):
        bench_module.build("gguf-strix", "md")
