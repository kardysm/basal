"""EagerBackend reads option codes in fp32 from the output head's input, with the model's final logit softcap (Gemma):
equal to the model's own fp32 logits, and in a bf16 model no longer rounded to bf16 logit steps. Llama, Gemma 4 and a
hybrid model (Qwen3.5: Gated DeltaNet and attention layers, left-padded batches and chunked prefill)."""
import torch
from transformers import (Gemma4ForCausalLM, Gemma4TextConfig, LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM,
                          Qwen3_5TextConfig)

from basal.engine import EagerBackend


class Stub(EagerBackend):
    def __init__(self, model):  # the real __init__ minus loading
        self.model, self.dev, self.tok = model, torch.device("cpu"), type("Tok", (), {"pad_token_id": 0})()
        self.head = model.get_output_embeddings()
        cfg = model.config.get_text_config()
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self._head_in = None
        self.head.register_forward_hook(self._grab)

    def encode(self, prompts):
        return prompts


def models():
    torch.manual_seed(0)
    yield LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)).eval()
    torch.manual_seed(0)
    yield Gemma4ForCausalLM(Gemma4TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                             final_logit_softcapping=3.0, max_position_embeddings=128)).eval()
    torch.manual_seed(0)
    yield Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True})).eval()


def reference(m, toks, lids):
    out = []
    with torch.no_grad():
        for t, ids in zip(toks, lids):
            lp = torch.log_softmax(m(input_ids=torch.tensor([t])).logits[0, -1].float(), -1)
            out.append(torch.softmax(lp[ids], -1))
    return out


def test_eager_codes_equal_full_logits_fp32():
    toks, lids = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10]], [[11, 12, 13], [14, 15]]
    for m in models():
        if hasattr(m.model, "embed_tokens"):  # a larger hidden state, so the softcap is active
            m.model.embed_tokens.weight.data *= 8
        got = Stub(m).run(toks, lids)
        for g, r in zip(got, reference(m, toks, lids)):
            assert torch.allclose(torch.tensor(g), r, atol=1e-5), (type(m).__name__, g, r)


def test_eager_bf16_codes_are_not_clumped():
    m = next(models()).to(torch.bfloat16)
    m.lm_head.weight.data *= 40  # logits of magnitude ~16-32, where bf16 steps are 0.125
    toks = [[1, 2, 3, i] for i in range(4, 40)]
    got = Stub(m).run(toks, [[11, 12]] * len(toks))
    with torch.no_grad():
        bf = [torch.softmax(m(input_ids=torch.tensor([t])).logits[0, -1, [11, 12]].float(), -1)[0].item() for t in toks]
    assert len({round(g[0], 9) for g in got}) > len({round(x, 9) for x in bf})


def test_eager_long_prompts_chunked_and_batches_capped(monkeypatch):
    """Prompts longer than CHUNK are prefilled in chunks through the KV cache, shorter ones batched up to
    MAX_BATCH_TOKENS: the probabilities equal the one-batch full forward (fp32)."""
    for m in models():
        torch.manual_seed(1)
        toks = [list(torch.randint(1, 60, (n,)).tolist()) for n in (5, 9, 23, 31, 40)]
        lids = [[11, 12, 13]] * len(toks)
        ref = Stub(m).run(toks, lids)
        s = Stub(m)
        monkeypatch.setattr(s, "CHUNK", 8)  # the three longest prompts in 8-token chunks
        monkeypatch.setattr(s, "MAX_BATCH_TOKENS", 12)  # the two short ones in separate batches
        got = s.run(toks, lids)
        for g, r in zip(got, ref):
            assert torch.allclose(torch.tensor(g), torch.tensor(r), atol=1e-5), (type(m).__name__, g, r)
