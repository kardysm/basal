"""--mode vllm (2.2.1): the vLLM engine arguments of each --vllm-preset, the calibration file each one loads, the
checks made before a model is loaded, and media through vLLM. No GPU and no vLLM needed (vLLM is stubbed)."""
import json
import os
import sys

import pytest

import basal.server as server_module
from basal.calibrate import engine_name
from basal.engine import VLLM_PRESETS, VLLMBackend, vllm_args
from basal.server import calibration_file, calibration_name, load_calibration, models_payload, parser

BF16 = {"architectures": ["Gemma4ForConditionalGeneration"], "vision_config": {"hidden_size": 1152},
        "text_config": {"num_hidden_layers": 60}}
FP8_PORT = {**BF16, "quantization_config": {"quant_method": "modelopt"}}  # a -FP8 checkpoint (ModelOpt)
E4B = {**BF16, "audio_config": {"hidden_size": 1536},
       "text_config": {"num_hidden_layers": 42, "num_kv_shared_layers": 18}}
TEXT = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 32}


def test_presets():
    assert list(VLLM_PRESETS) == ["default", "deterministic"]   # no load-time FP8 weights, no FP8 KV cache (not gated)
    assert vllm_args(TEXT) == ({}, {})                            # default: as in 2.2
    assert vllm_args(TEXT, "deterministic") == ({"enable_prefix_caching": False, "enable_chunked_prefill": False,
                                                 "attention_backend": "TRITON_ATTN"}, {"VLLM_BATCH_INVARIANT": "1"})
    for name in ("nvfp4", "fp8", "fp8-kv"):
        with pytest.raises(ValueError, match=f"unknown --vllm-preset '{name}'"):
            vllm_args(TEXT, name)


def test_text_only_and_kv_shared_layers():
    assert vllm_args(BF16)[0] == {}                                         # the encoders load (attention backend)
    assert vllm_args(E4B)[0] == {"kv_sharing_fast_prefill": True}                  # E2B / E4B: on by default
    assert vllm_args(E4B, fast_prefill=False)[0] == {}                              # --vllm-fast-prefill off
    assert vllm_args(BF16)[0] == {}                                                 # no KV-shared layers: unaffected
    assert vllm_args(E4B, "deterministic", fast_prefill=True)[0]["kv_sharing_fast_prefill"]


@pytest.fixture
def fake_vllm(monkeypatch):
    seen = {}

    class LLM:
        def __init__(self, **kw):
            seen.update(kw, env=os.environ.get("VLLM_BATCH_INVARIANT"))
    monkeypatch.setitem(sys.modules, "vllm", type(sys)("vllm"))
    sys.modules["vllm"].LLM = LLM
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda *_: None)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    return seen


def test_backend_passes_the_preset_to_vllm(tmp_path, fake_vllm):
    (tmp_path / "config.json").write_text(json.dumps(E4B), encoding="utf-8")
    be = VLLMBackend(tmp_path, max_len=16384, fast_prefill=True)
    assert be.preset == "default" and "quantization" not in fake_vllm and "kv_cache_dtype" not in fake_vllm
    assert fake_vllm["kv_sharing_fast_prefill"] and "language_model_only" not in fake_vllm
    assert fake_vllm["enable_prefix_caching"] and fake_vllm["max_model_len"] == 16384 and fake_vllm["env"] is None
    fake_vllm.clear()
    VLLMBackend(tmp_path, preset="deterministic", fast_prefill=False)
    assert fake_vllm["env"] == "1" and "kv_sharing_fast_prefill" not in fake_vllm and "quantization" not in fake_vllm
    assert fake_vllm["enable_prefix_caching"] is False  # the preset's value wins over the backend's default


def test_bench_mode_with_a_preset(tmp_path, fake_vllm):
    from basal.bench import build
    (tmp_path / "config.json").write_text(json.dumps(TEXT), encoding="utf-8")
    assert build("vllm@deterministic", tmp_path, fmt=2).preset == "deterministic"
    assert fake_vllm["enable_prefix_caching"] is False
    assert fake_vllm["max_logprobs"] == 255


def test_flags():
    a = parser().parse_args([])
    assert a.vllm_preset == "default" and a.vllm_fast_prefill == "auto"   # on for E2B / E4B (gated on mini, 2.3)
    a = parser().parse_args(["--mode", "vllm", "--vllm-preset", "deterministic", "--vllm-fast-prefill", "off"])
    assert a.vllm_preset == "deterministic" and a.vllm_fast_prefill == "off"
    for name in ("nvfp4", "fp8", "fp8-kv"):  # no NVFP4; FP8 weights at load failed, the FP8 KV cache is not gated
        with pytest.raises(SystemExit):
            parser().parse_args(["--mode", "vllm", "--vllm-preset", name])


@pytest.mark.parametrize("mode", ["fast", "eager", "sglang"])
def test_preset_needs_vllm(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", mode, "--vllm-preset", "deterministic"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--vllm-preset needs --mode vllm"):
        server_module.main()


def test_calibration_file_per_preset(tmp_path):
    assert calibration_name("vllm", None) == "vllm" and calibration_name("vllm", None, "default") == "vllm"
    assert calibration_name("vllm", None, "deterministic") == "vllm.deterministic"
    assert calibration_name("graph", "fp8", "deterministic") == "fp8" and calibration_name("eager", None, "fp8") is None
    (tmp_path / "CALIBRATION.json").write_text("{}", encoding="utf-8")
    assert calibration_file(tmp_path, "vllm", preset="deterministic") == tmp_path / "CALIBRATION.json"
    (tmp_path / "CALIBRATION.vllm.json").write_text("{}", encoding="utf-8")
    # a preset without a file of its own: the vLLM file (a -FP8 checkpoint's own fit, or the bf16 vLLM fit)
    assert calibration_file(tmp_path, "vllm", preset="deterministic") == tmp_path / "CALIBRATION.vllm.json"
    (tmp_path / "CALIBRATION.vllm.deterministic.json").write_text("{}", encoding="utf-8")
    assert calibration_file(tmp_path, "vllm", preset="deterministic") == tmp_path / "CALIBRATION.vllm.deterministic.json"
    assert calibration_file(tmp_path, "vllm") == tmp_path / "CALIBRATION.vllm.json"
    assert calibration_file(tmp_path, "eager", preset="deterministic") == tmp_path / "CALIBRATION.json"
    assert calibration_file(tmp_path, "vllm", path="X.json", preset="deterministic").name == "X.json"  # --calibration wins


def test_preset_file_for_the_served_orders(tmp_path):
    (tmp_path / "CALIBRATION.vllm.deterministic.json").write_text(json.dumps({"orders": 1, "thresholds": {"0.01": {}}}), encoding="utf-8")
    (tmp_path / "CALIBRATION.vllm.deterministic.o2.json").write_text(json.dumps({"orders": 2, "thresholds": {"0.01": {}}}), encoding="utf-8")
    cal, path, why = load_calibration(tmp_path, "vllm", None, None, 2, preset="deterministic")
    assert path.name == "CALIBRATION.vllm.deterministic.o2.json" and why is None and cal["orders"] == 2
    cal, path, why = load_calibration(tmp_path, "vllm", None, None, 1, preset="deterministic")
    assert path.name == "CALIBRATION.vllm.deterministic.json" and why is None


def test_preset_reported_and_named():
    assert models_payload("m", "vllm", {}, vllm_preset="deterministic")["models"][0]["vllm_preset"] == "deterministic"
    assert "vllm_preset" not in models_payload("m", "fast", {})["models"][0]
    assert engine_name("vllm", None, "deterministic") == "vllm.deterministic" and engine_name("vllm", None) == "vllm"
    assert engine_name("graph", None, "deterministic") == "fast"


def test_media_engine_args():
    assert vllm_args(E4B, media=True, fast_prefill=False)[0] == {"limit_mm_per_prompt": {"image": 8, "audio": 8,
                                                                                         "video": 0}}
    assert vllm_args(BF16, media=True)[0] == {"limit_mm_per_prompt": {"image": 8, "audio": 0, "video": 0}}  # ultra


def test_media_needs_an_encoder(tmp_path, fake_vllm):
    (tmp_path / "config.json").write_text(json.dumps(TEXT), encoding="utf-8")
    with pytest.raises(SystemExit, match="no image or audio encoder"):
        VLLMBackend(tmp_path, media=True)
    (tmp_path / "config.json").write_text(json.dumps(E4B), encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(json.dumps({"audio_seq_length": 750, "audio_ms_per_token": 40}), encoding="utf-8")
    be = VLLMBackend(tmp_path, media=True)
    assert be.media_kinds == ("image", "audio") and be.max_audio_s == 30 and fake_vllm["limit_mm_per_prompt"]
    (tmp_path / "config.json").write_text(json.dumps({**E4B, "architectures": ["Gemma4UnifiedForConditionalGeneration"]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not serve Gemma 4 Unified"):  # 1.1-max failed the media gate
        VLLMBackend(tmp_path, media=True)


class Out:
    def __init__(self, prompt_token_ids, logprobs):
        Logprob = type("Logprob", (), {})
        lp = {}
        for t, v in logprobs.items():
            lp[t] = Logprob()
            lp[t].logprob = v
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [type("C", (), {"logprobs": [lp]})()]


def media_backend(monkeypatch, generate):
    sps = type(sys)("vllm")
    sps.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", sps)
    be = VLLMBackend.__new__(VLLMBackend)
    be.prefill = "<prefill>"
    template = lambda self, m, **kw: f"[{m[0]['content']}|{kw['enable_thinking']}]"  # noqa: E731
    be.tok = type("Tok", (), {"image_token_id": 7, "audio_token_id": 9, "apply_chat_template": template})()
    be.llm = type("L", (), {"generate": generate})()
    return be


def test_media_prompts_share_one_vllm_call(monkeypatch):
    calls = []

    def generate(self, reqs, sps, use_tqdm):
        calls.append((reqs, sps))
        return [Out([1, 7, 7, 9, 2, 3], {10: -0.1, 11: -2.4}), Out([1, 7, 7, 9, 2, 4, 5], {10: -3.0, 11: -0.05})]
    be = media_backend(monkeypatch, generate)
    msgs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    out = be.run_media(msgs, [[10, 11], [10, 11]], ["IMG"], [[0.0, 0.1]], 1120)
    (reqs, sps), = calls
    assert [r["prompt"] for r in reqs] == ["[a|False]<prefill>", "[b|False]<prefill>"]
    assert reqs[0]["multi_modal_data"] == {"image": ["IMG"], "audio": [([0.0, 0.1], 16000)]}
    assert reqs[0]["mm_processor_kwargs"] == {"max_soft_tokens": 1120}
    assert sps[0]["allowed_token_ids"] == [10, 11] and sps[0]["max_tokens"] == 1
    assert [(n, nm) for _, n, nm in out] == [(6, 3), (7, 3)]  # prompt tokens, media tokens (image + audio)
    assert out[0][0][0] > 0.9 and out[1][0][1] > 0.9
    text_only = media_backend(monkeypatch, generate)
    text_only.run_media(msgs, [[10, 11], [10, 11]], [], [[0.0]], 280)
    assert calls[-1][0][0]["multi_modal_data"] == {"audio": [([0.0], 16000)]}  # no empty image list


def test_media_prompt_over_the_context(monkeypatch):
    def generate(self, reqs, sps, use_tqdm):
        raise ValueError("The decoder prompt (length 20000) is longer than the maximum model length of 16383. ...")
    be = media_backend(monkeypatch, generate)
    with pytest.raises(ValueError, match="the prompt with its media is 20000 tokens, more than the 16383 tokens"):
        be.run_media([[{"role": "user", "content": "a"}]], [[10, 11]], ["IMG"], [], 280, 16383)
