"""Prompt layouts of format 2 (2.4.1; --layout, or a model's basal.json "layout"): the canonical prompt is unchanged
bit for bit, the other layouts keep the codes and the prefill, a question's answer never depends on the other questions
of its request, and each layout loads its own calibration file."""
import asyncio
import hashlib
import json
import sys

import pytest

from basal import prompt as P
from basal import server as server_module
from basal.server import calibration_file, load_calibration
from tests.test_prompt2 import StubTok
from tests.test_server import FakeServer

CASES = [("Paczka dotarła uszkodzona.", "Czy klient żąda zwrotu?", ["tak", "nie"]),
         ("The parcel arrived damaged.", "Does the customer want a refund?", ["yes", "no", "unclear"]),
         ("", "Which colour?", [f"opt{i}" for i in range(30)])]


def test_canonical_prompts_unchanged_bit_for_bit():
    """Both formats, both languages, default and explicit canonical layout: the same bytes as basal 2.3.5 (the hash
    was taken from main 95f76e2's render)."""
    tok = StubTok()
    for layout in (None, "canonical"):
        h = hashlib.sha256()
        for fmt in (1, 2):
            for s, q, o in CASES:
                o = o[:10] if fmt == 1 else o
                kw = {} if layout is None else {"layout": layout}
                h.update(P.render(tok, s, q, o, fmt=fmt, **kw).encode())
                h.update(P.render(tok, s, q, o, lang="pl", fmt=fmt, **kw).encode())
        assert h.hexdigest() == "23e42d1a5e1f3cbcfcd96c9638744fda77b878c2566e3223f6fa83d7f845937c"


@pytest.mark.parametrize("lang", ["pl", "en"])
def test_layouts_keep_codes_prefill_and_system(lang):
    tok = StubTok()
    state, q, opts = "S1 S2.", "Q?", ["a", "b", "c"]
    canon = P.render(tok, state, q, opts, lang=lang, fmt=2)
    tail = canon[canon.index("\n" + ("Opcje" if lang == "pl" else "Options")):]  # the options and the prefill
    for layout in ("state_twice", "qfirst"):
        p = P.render(tok, state, q, opts, lang=lang, fmt=2, layout=layout)
        assert p.endswith(tail) and p.endswith(P.PREFILL)
        assert p.split("<user>")[0] == canon.split("<user>")[0]  # the same system prompt
        assert P.letter_ids(tok, p, 3, fmt=2) is not None
    assert P.render(tok, state, q, opts, lang=lang, fmt=2, layout="state_twice").count(state) == 2
    qf = P.render(tok, state, q, opts, lang=lang, fmt=2, layout="qfirst")
    assert qf.count(q) == 2 and qf.index(q) < qf.index(state)


def test_layouts_need_format_2_and_a_known_name():
    tok = StubTok()
    with pytest.raises(ValueError, match="format 2"):
        P.render(tok, "s", "q", ["a", "b"], fmt=1, layout="qfirst")
    with pytest.raises(ValueError, match="unknown layout"):
        P.render(tok, "s", "q", ["a", "b"], fmt=2, layout="header")


def _probs(prompt, ids):
    """Position probabilities as a fixed function of the whole prompt (what a model sees decides what it answers)."""
    d = hashlib.sha256(bytes(x % 256 for x in prompt)).digest()
    w = [1 + d[i] for i in range(len(ids))]
    return [x / sum(w) for x in w]


@pytest.mark.parametrize("layout", P.LAYOUTS)
@pytest.mark.parametrize("soam", [True, False])
def test_question_independence(layout, soam):
    """A question's answer is the same alone and with other questions in its request: its prompts hold its own
    question only (a header listing every question would fail this), in every layout, with and without SOAM."""
    state = "Paczka dotarła uszkodzona, klient żąda zwrotu pieniędzy albo wymiany."
    qs = {"dept": {"type": "choice", "instructions": "Dział?", "criteria": {"z": "Zwroty", "i": "IT", "h": "HR"}},
          "dmg": {"type": "noul", "instructions": "Czy paczka jest uszkodzona?"},
          "urg": {"type": "score", "instructions": "Pilność?", "criteria": ["niska", "średnia", "wysoka"]}}

    def answers(names):
        srv = FakeServer(_probs, soam=soam)
        srv.fmt, srv.max_options, srv.layout = 2, 255, layout
        return asyncio.run(srv._run({"state": state, "questions": {n: qs[n] for n in names}}))["answers"]
    together = answers(list(qs))
    for n in qs:
        assert answers([n])[n] == together[n]


def test_each_layout_loads_its_calibration(tmp_path):
    canon = {"prompt": 2, "temperature_per_prim": {"choice": 1.2}, "thresholds": {"choice": {"0.05": 0.7}}}
    (tmp_path / "CALIBRATION.json").write_text(json.dumps(canon), encoding="utf-8")
    assert calibration_file(tmp_path, "graph") == tmp_path / "CALIBRATION.json"
    assert calibration_file(tmp_path, "graph", layout="qfirst") == tmp_path / "CALIBRATION.json"
    data, f, why = load_calibration(tmp_path, "graph", None, None, 1, layout="qfirst")
    assert f.name == "CALIBRATION.json" and "thresholds" not in data and "canonical layout" in why
    assert data["temperature_per_prim"] == {"choice": 1.2}
    own = dict(canon, fitted_on={"layout": "qfirst", "orders": 1})
    (tmp_path / "CALIBRATION.qfirst.json").write_text(json.dumps(own), encoding="utf-8")
    (tmp_path / "CALIBRATION.mlx.qfirst.json").write_text(json.dumps(own), encoding="utf-8")
    assert calibration_file(tmp_path, "graph", layout="qfirst") == tmp_path / "CALIBRATION.qfirst.json"
    assert calibration_file(tmp_path, "mlx", layout="qfirst") == tmp_path / "CALIBRATION.mlx.qfirst.json"
    data, f, why = load_calibration(tmp_path, "graph", None, None, 1, layout="qfirst")
    assert f.name == "CALIBRATION.qfirst.json" and why is None and data["thresholds"]
    data, f, why = load_calibration(tmp_path, "graph", None, None, 1)  # canonical: unchanged
    assert f.name == "CALIBRATION.json" and why is None and data["thresholds"]


def test_layout_refuses_the_state_caches_before_loading(monkeypatch):
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    for flag in (["--state-cache-gb", "2"], ["--prefix-cache", "on"]):
        monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "fast", "--layout", "qfirst"] + flag)
        with pytest.raises(SystemExit, match="canonical layout"):
            server_module.main()


def test_question_independence_check_catches_a_header(monkeypatch):
    """The check above fails where prompts carry the request's other questions (a header layout): here every prompt
    also lists the questions rendered before it in the request."""
    seen = []

    def header(tok, state, question, options, lang=None, fmt=1, layout="canonical"):
        if question not in seen:
            seen.append(question)
        return P.render(tok, state, " | ".join(seen) + " || " + question, options, lang, fmt)
    monkeypatch.setattr(server_module, "render", header)
    qs = {"a": {"type": "noul", "instructions": "Czy A?"}, "b": {"type": "noul", "instructions": "Czy B?"}}

    def answers(names):
        seen.clear()
        srv = FakeServer(_probs)
        srv.fmt, srv.max_options = 2, 255
        return asyncio.run(srv._run({"state": "Stan.", "questions": {n: qs[n] for n in names}}))["answers"]
    assert answers(["b"])["b"] != answers(["a", "b"])["b"]


def test_model_layout_from_basal_json(tmp_path):
    assert P.model_layout(tmp_path) == "canonical"  # no basal.json
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2}), encoding="utf-8")
    assert P.model_layout(tmp_path) == "canonical"
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, "layout": "qfirst"}), encoding="utf-8")
    assert P.model_layout(tmp_path) == "qfirst"
    (tmp_path / "basal.json").write_text(json.dumps({"prompt": 2, "layout": "header"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown prompt layout"):
        P.model_layout(tmp_path)


def test_models_payload_reports_a_non_canonical_layout():
    entry = server_module.models_payload("m", "fast", {}, layout="qfirst")["models"][0]
    assert entry["layout"] == "qfirst"
    assert "layout" not in server_module.models_payload("m", "fast", {}, layout="canonical")["models"][0]
