"""GGUF backend on hybrid / recurrent architectures (Qwen3.5: GGUF architecture qwen35). A stand-in llama_cpp module
computes every readout from the tokens of its own sequence and, like llama.cpp on the CPU with the 0.8B pico (22 Oct),
returns non-finite logits when one llama_decode of a recurrent model holds several sequences. Such a model must
decode one prompt per llama_decode, so single questions, both option orders and multi-question requests (SOAM) give
finite probabilities equal to each prompt evaluated alone; attention models keep the packed trie (one decode per
chunk). No llama.cpp or GPU needed."""
import ctypes
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from basal.engine import GGUFBackend

V = 64


class Tok:  # one character = one token, ids 3..63
    def __call__(self, text, add_special_tokens=False):
        enc = lambda t: [3 + ord(c) % (V - 3) for c in t]  # noqa: E731
        return type("E", (), {"input_ids": [enc(t) for t in text] if isinstance(text, list) else enc(text)})()

    def __len__(self):
        return V


def fake_llama(arch, recurrent=False, hybrid=False, meta=()):
    """A llama_cpp module whose model reads out sum_k sin(0.37 * token_k * (v + 1) + 0.11 * pos_k) over the tokens of
    the readout's sequence up to its position (order-dependent, like a decoder); records every llama_decode."""
    C = ModuleType("llama_cpp")
    C.decodes, C.ctx_params = [], None
    meta = {"general.architecture": arch, **dict(meta)}

    def batch_init(n_tokens, embd, n_seq):
        return SimpleNamespace(token=[0] * n_tokens, pos=[0] * n_tokens, logits=[0] * n_tokens,
                               n_seq_id=[0] * n_tokens, seq_id=[[0] * n_seq for _ in range(n_tokens)], n_tokens=0)

    def meta_val_str(model, key, buf, size):
        v = meta.get(key.decode())
        if v is None:
            return -1
        buf.value = v.encode()
        return len(v)

    def decode(ctx, b):
        n = b.n_tokens
        seqs = [set(b.seq_id[i][:b.n_seq_id[i]]) for i in range(n)]
        C.decodes.append(len(set().union(*seqs)))
        bad = (recurrent or hybrid) and (C.decodes[-1] > 1 or any(len(s) > 1 for s in seqs))
        C.out = {}
        for i in range(n):
            if not b.logits[i]:
                continue
            (s,) = seqs[i]
            toks = [(b.pos[j], b.token[j]) for j in range(n) if s in seqs[j] and b.pos[j] <= b.pos[i]]
            z = np.array([sum(np.sin(0.37 * t * (v + 1) + 0.11 * p) for p, t in toks) for v in range(V)])
            C.out[i] = (ctypes.c_float * V)(*(np.full(V, np.nan) if bad else z))
        return 0

    C.llama_backend_init = lambda: None
    C.llama_model_default_params = lambda: SimpleNamespace(n_gpu_layers=0)
    C.llama_model_load_from_file = lambda path, mp: object()
    C.llama_model_meta_val_str = meta_val_str
    C.llama_model_is_recurrent = lambda m: recurrent
    C.llama_model_is_hybrid = lambda m: hybrid
    C.llama_model_get_vocab = lambda m: None
    C.llama_vocab_n_tokens = lambda vocab: V
    C.llama_context_default_params = lambda: SimpleNamespace()

    def init_from_model(m, cp):
        C.ctx_params = cp
        return object()
    C.llama_init_from_model = init_from_model
    C.llama_batch_init = batch_init
    C.llama_get_memory = lambda ctx: None
    C.llama_memory_clear = lambda mem, data: None
    C.llama_decode = decode
    C.llama_get_logits_ith = lambda ctx, i: ctypes.cast(C.out[i], ctypes.POINTER(ctypes.c_float))
    C.llama_batch_free = C.llama_free = C.llama_model_free = lambda *a: None
    return C


def backend(monkeypatch, tmp_path, **kw):
    C = fake_llama(**kw)
    monkeypatch.setitem(sys.modules, "llama_cpp", C)
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda _: Tok())
    return GGUFBackend(tmp_path, tmp_path / "model.gguf", n_gpu_layers=0), C  # no config.json: a -GGUF repository


IDS = [10, 20, 30]
STATE = "a shared state of a request, long enough to matter: " * 2
PROMPTS = [STATE + q + o for q in ("q1? ", "question two? ", "3? ") for o in (" A. x B. y C. z", " A. z B. y C. x")]


def alone(be, prompts):
    return [be.run_shared([([p], [IDS])])[0][0] for p in prompts]


@pytest.mark.parametrize("kw", [dict(arch="qwen35", hybrid=True), dict(arch="mamba2", recurrent=True),
                                dict(arch="qwen35"),  # by its architecture name alone
                                dict(arch="newhybrid", meta={"newhybrid.ssm.state_size": "128"})])
def test_hybrid_decodes_one_prompt_per_llama_decode(monkeypatch, tmp_path, kw):
    be, C = backend(monkeypatch, tmp_path, **kw)
    assert be.recurrent and be.N_SEQ == 1 and C.ctx_params.n_seq_max == 1
    soam = be.run_shared([(PROMPTS, [IDS] * len(PROMPTS))])[0]  # one request: three questions in two orders
    assert C.decodes == [1] * len(PROMPTS)
    assert all(np.isfinite(p).all() for p in soam)
    assert np.allclose(soam, alone(be, PROMPTS), atol=1e-6)
    C.decodes.clear()  # several requests (groups) in one call: still one prompt per decode
    two = be.run_shared([(PROMPTS[:2], [IDS] * 2), (PROMPTS[2:5], [IDS] * 3), ([PROMPTS[5]], [IDS])])
    assert C.decodes == [1] * 6
    assert np.allclose([p for g in two for p in g], soam, atol=1e-6)
    assert np.allclose(be.run(PROMPTS[:3], [IDS] * 3), soam[:3], atol=1e-6)


def test_attention_models_keep_the_packed_trie(monkeypatch, tmp_path):
    be, C = backend(monkeypatch, tmp_path, arch="llama")
    assert not be.recurrent and be.N_SEQ == GGUFBackend.N_SEQ and C.ctx_params.n_seq_max == GGUFBackend.N_SEQ
    packed = be.run_shared([(PROMPTS, [IDS] * len(PROMPTS))])[0]
    assert C.decodes == [len(PROMPTS)]  # the state once, in the sequences of all six readouts
    assert np.allclose(packed, alone(be, PROMPTS), atol=1e-6)


def test_stand_in_fails_like_llama_cpp_on_the_cpu(monkeypatch, tmp_path):
    """The bug this guards against: several sequences of a hybrid model in one llama_decode (v2.3.1's packing)."""
    be, C = backend(monkeypatch, tmp_path, arch="qwen35", hybrid=True)
    monkeypatch.setattr(be, "N_SEQ", GGUFBackend.N_SEQ)
    be.batch = C.llama_batch_init(be.N_CTX, 0, be.N_SEQ)
    assert not np.isfinite(be.run_shared([(PROMPTS, [IDS] * len(PROMPTS))])[0]).all()
