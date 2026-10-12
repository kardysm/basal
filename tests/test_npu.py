"""npu mode (ONNX Runtime GenAI, the runtime of the AMD Ryzen AI NPU): the letter readout equals the PyTorch readout,
every shared prefix of a call is computed once (the prompts follow the depth-first order of their prefix trie and each
one rewinds the KV cache to its common prefix with the previous one), and the server answers as in --mode eager. Tiny
random Llama with the basal layout (attention and MLP biases, untied head, grouped KV heads, transformers-5
`rope_parameters`), exported by the onnxruntime-genai model builder for the CPU: the same backend code that runs a
folder prepared by Ryzen AI's model_generate on the NPU."""
import asyncio
import json
import warnings

import pytest

pytest.importorskip("onnxruntime_genai")
pytest.importorskip("onnx_ir")  # the model builder

import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from basal.engine import GraphBackend, OGABackend

V = 96  # not a hidden or MLP width: the builder takes any Linear with vocab_size outputs for the LM head
CHARS = " \n\t!\"#%&'()*+,-./0123456789:;<=>?[]_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{|}"
TEMPLATE = ("{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}[assistant]{% endif %}")


def tiny(path):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, attention_bias=True,
                      mlp_bias=True, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                      rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # random biases (initialised to zero), so that a dropped bias would be noticed
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                p.normal_(0, 0.1)
            elif n.endswith(("q_proj.weight", "k_proj.weight")):
                p.mul_(8.0)  # peaky attention, so that positions (rotary settings) matter
    m.save_pretrained(path)
    vocab = {t: i for i, t in enumerate(["<pad>", "<s>", "</s>", "<unk>"] + list(CHARS))}
    assert len(vocab) <= V
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>")),
                                  bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)  # one token per character: every option letter is a single token
    return m


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from onnxruntime_genai.models.builder import create_model
    root = tmp_path_factory.mktemp("npu")
    m = tiny(root / "hf")
    create_model(None, str(root / "hf"), str(root / "onnx"), "fp32", "cpu", str(root / "cache"))
    return m, root / "hf", root / "onnx"


@pytest.fixture()
def backend(model):
    _, hf, onnx = model
    return OGABackend(hf, onnx)


def reference(m, toks, lids):
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1).tolist())
    return out


def close(a, b, tol=1e-4):
    return all(max(abs(x - y) for x, y in zip(p, q)) < tol for p, q in zip(a, b))


STATE = list(range(21, 61)) * 3  # 120 shared tokens
# SOAM group: state | question 1 (two orders) | question 2 (two orders) | question 3, suffixes of unequal length
TOKS = [STATE + [5, 6, 10, 11], STATE + [5, 6, 11, 10], STATE + [7, 8, 9, 10, 11, 12], STATE + [7, 8, 9, 12, 11, 10],
        STATE + [3]]
LIDS = [[10, 11], [10, 11], [10, 11, 12], [10, 11, 12], [10, 11, 12, 13]]


def test_npu_equals_torch(model, backend):
    m = model[0]
    assert backend.prefix  # rewind_to passed the start-up check on this execution provider
    assert close(backend.run_shared([(TOKS, LIDS)])[0], reference(m, TOKS, LIDS))
    singles = [list(range(4, 40)), list(range(5, 90)) * 2, [9, 8, 7, 6, 5, 4, 3, 2]]
    sl = [[10, 11], [12, 13, 14], [10, 15]]
    assert close(backend.run(singles, sl), reference(m, singles, sl))
    mixed = backend.run_shared([([singles[0]], [sl[0]]), (TOKS[:2], LIDS[:2]), ([singles[1]], [sl[1]])])
    assert close([mixed[0][0], *mixed[1], mixed[2][0]], reference(m, [singles[0], *TOKS[:2], singles[1]],
                                                                  [sl[0], *LIDS[:2], sl[1]]))


class Counting:
    """Generator proxy that counts the tokens actually computed."""
    def __init__(self, g, log):
        self.g, self.log = g, log

    def append_tokens(self, x):
        self.log.append(x.shape[1])
        return self.g.append_tokens(x)

    def __getattr__(self, name):
        return getattr(self.g, name)


def test_each_shared_prefix_is_computed_once(backend, monkeypatch):
    log = []
    make = backend._generator
    monkeypatch.setattr(backend, "_generator", lambda: Counting(make(), log))
    backend.reset_prefix_cache()
    backend.gen = None
    order = [4, 2, 0, 3, 1]  # any input order: the prompts are evaluated in prefix-trie order
    backend.run_shared([([TOKS[k] for k in order], [LIDS[k] for k in order])])
    assert sum(log) == len(GraphBackend._pack(TOKS)[0])  # the tokens of the prefix trie: the state once
    log.clear()
    backend.run_shared([(TOKS[:2], LIDS[:2])])  # the state stays cached across calls
    assert sum(log) == 2 * 4 - 2  # question 1 and both option orders (the two orders share the question tokens)


def test_full_prompts_without_rewind(model, backend):
    backend.prefix = False
    assert close(backend.run_shared([(TOKS, LIDS)])[0], reference(model[0], TOKS, LIDS))


def test_failing_rewind_turns_prefix_reuse_off(model, monkeypatch):
    _, hf, onnx = model

    class NoRewind(Counting):
        def rewind_to(self, n):
            raise RuntimeError("rewind is not supported by this provider")

    make = OGABackend._generator
    monkeypatch.setattr(OGABackend, "_generator", lambda self: NoRewind(make(self), []))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        be = OGABackend(hf, onnx)
    assert not be.prefix and any("rewind_to is not supported" in str(x.message) for x in w)
    assert close(be.run_shared([(TOKS, LIDS)])[0], reference(model[0], TOKS, LIDS))


def test_mismatched_architecture_fails_before_loading(model, tmp_path, monkeypatch):
    import onnxruntime_genai as og
    _, hf, onnx = model
    other = tmp_path / "other"
    other.mkdir()
    for f in hf.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    cfg = json.loads((hf / "config.json").read_text())
    (other / "config.json").write_text(json.dumps({**cfg, "num_hidden_layers": 3}))
    monkeypatch.setattr(og, "Model", lambda *_: pytest.fail("the model must be validated before it is loaded"))
    with pytest.raises(ValueError, match="num_hidden_layers"):
        OGABackend(other, onnx)
    with pytest.raises(ValueError, match="genai_config.json"):
        OGABackend(hf, hf)


def test_prompt_longer_than_the_context_is_refused(backend):
    with pytest.raises(ValueError, match="too long"):
        backend.run([[5] * (backend.n_ctx + 1)], [[10, 11]])
    assert close(backend.run([TOKS[0]], [LIDS[0]]), backend.run([TOKS[0]], [LIDS[0]]))  # still usable afterwards


BODY = {"state": "Ticket: the card terminal in shop 12 rejects every payment since 7:40; cash only.",
        "questions": {
            "dept": {"type": "choice", "instructions": "Which team should take the ticket?",
                     "criteria": {"cards": "Card payments", "web": "Online banking", "loans": "Loans"}},
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "tags": {"type": "multi", "instructions": "Which apply?", "criteria": ["payments", "fraud"]}}}


def test_server_npu_mode_answers_as_eager(model):
    from basal.server import Server, parser
    _, hf, onnx = model

    async def decide(args):
        srv = Server(parser().parse_args(args))
        task = asyncio.get_running_loop().create_task(srv.worker())
        try:
            return await srv.decide(BODY)
        finally:
            task.cancel()

    ref = asyncio.run(decide(["--model", str(hf), "--mode", "eager", "--device", "cpu", "--dtype", "float32"]))
    got = asyncio.run(decide(["--model", str(hf), "--mode", "npu", "--onnx", str(onnx)]))
    assert got["usage"]["input_tokens"] == ref["usage"]["input_tokens"]
    for name, a in ref["answers"].items():
        for k, p in a["probabilities"].items():
            assert abs(got["answers"][name]["probabilities"][k] - p) < 1e-4
