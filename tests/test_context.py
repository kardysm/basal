"""A prompt longer than the context is refused with HTTP 422 (token count, limit, hint) before any engine call: the
model's context (max_position_embeddings; mini: 8,192) or the engine's own limit when lower. With SOAM a prompt is the
state plus one question, so the state counts once plus the longest question."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import basal.engine as engine_module
from basal.engine import LlamaCppBackend, OllamaBackend
from basal.server import context_limit, model_context
from tests.test_ollama import http_stub
from tests.test_prompt2 import fmt2_server
from tests.test_server import FakeServer

MINI = (8192, "the model's context")  # basal-1.5-mini: max_position_embeddings 8192


def server(context=MINI):
    srv = FakeServer(lambda p, ids: [1 / len(ids)] * len(ids))
    srv.context = context
    return srv


def ask(srv, questions, state):
    return asyncio.run(srv._run({"state": state, "questions": questions}))


def test_model_and_engine_limits(tmp_path):
    assert model_context(tmp_path) is None
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}), encoding="utf-8")
    assert model_context(tmp_path) == 8192
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 32768}}), encoding="utf-8")
    assert model_context(tmp_path) == 32768
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}), encoding="utf-8")
    plain = type("B", (), {})()
    assert context_limit(tmp_path, plain) == MINI
    vllm = type("B", (), {"context": 4095, "context_reason": "vLLM's --max-len 4096"})()
    assert context_limit(tmp_path, vllm) == (4095, "vLLM's --max-len 4096")      # the lower limit wins
    big = type("B", (), {"context": 16384, "context_reason": "llama.cpp"})()
    assert context_limit(tmp_path, big) == MINI
    assert context_limit(tmp_path / "missing", plain) is None


def test_a_prompt_over_the_context_is_refused_with_its_length():
    q = {"q": {"type": "choice", "instructions": "Dept?", "criteria": {"a": "Returns", "b": "IT"}}}
    assert ask(server(), q, "x" * 7000)["answers"]["q"]["type"] == "choice"           # fits 8,192
    with pytest.raises(ValueError) as e:
        ask(server(), q, "x" * 8300)
    msg = str(e.value)
    assert msg.startswith("question 'q': the prompt is 8") and "more than the 8192 tokens of the model's context" in msg
    assert "shorten the state" in msg
    assert ask(server(None), q, "x" * 8300)["answers"]["q"]["type"] == "choice"       # no known limit: no check


def test_soam_counts_the_state_once_plus_the_longest_question():
    """Three questions over a 5,000-token state: 5k + 3 x 1k tokens in one packed row, but every prompt is about 6k."""
    srv = server()
    qs = {f"q{i}": {"type": "choice", "instructions": f"Question {i}?",
                    "criteria": {"a": "y" * 500, "b": "z" * 500}} for i in range(3)}
    r = ask(srv, qs, "x" * 5000)
    assert srv.entries == 1 and r["usage"]["input_tokens"] > 8192
    qs["long"] = {"type": "noul", "instructions": "w" * 3500}
    with pytest.raises(ValueError, match="question 'long': the prompt is .* tokens"):
        ask(server(), qs, "x" * 5000)


def test_tool_questions_get_the_tool_hint():
    """80 tools with 300-character descriptions (about 130 stub tokens each) do not fit 8,192; cut at 50 they do."""
    def srv():
        s = fmt2_server(lambda p, ids: [1 / len(ids)] * len(ids))
        s.context = MINI
        return s
    tools = [{"name": f"tool_{i}", "description": "word " * 60} for i in range(80)]
    with pytest.raises(ValueError, match="lower description_chars, give fewer tools, or shortlist"):
        ask(srv(), {"next": {"type": "tool", "tools": tools}}, "User: help.")
    fits = ask(srv(), {"next": {"type": "tool", "tools": tools, "description_chars": 50}}, "User: help.")
    assert fits["answers"]["next"]["type"] == "tool"


class _Props(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({"default_generation_settings": {"n_ctx": 4096}, "total_slots": 1}).encode()
        self.send_response(200 if self.path == "/props" else 404)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_engine_limits_of_the_http_backends(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Props)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = http_stub(LlamaCppBackend, f"http://127.0.0.1:{srv.server_address[1]}")
        assert be.slot_context() == 4096
        assert http_stub(LlamaCppBackend, "http://127.0.0.1:9").slot_context() is None   # nothing listening
    finally:
        srv.shutdown()
    monkeypatch.setattr(engine_module.AutoTokenizer, "from_pretrained", lambda *_a, **_k: object())
    ol = OllamaBackend("m", model="basal:q8_0", num_ctx=4096)
    assert ol.context == 4095 and "num_ctx 4096" in ol.context_reason
    assert engine_module.GGUFBackend.context == engine_module.GGUFBackend.N_CTX
