"""Request handling of the HTTP server without a model: option names reach the prompt, and responses and the model
list conform to the official System One OpenAPI schema (tests/data/systemone_openapi.json)."""
import asyncio
import json
import sys
from pathlib import Path

import jsonschema
import pytest

import basal.bench as bench_module
import basal.server as server_module
from basal.facts import HEADER, facts, inject
from basal.server import Server, default_mode, models_payload, named_options, parser, to_items

SPEC = json.loads((Path(__file__).parent / "data/systemone_openapi.json").read_text())


def schema(name):
    s = json.loads(json.dumps(SPEC["components"]["schemas"][name]).replace("#/components/schemas/", "#/$defs/"))
    s["$defs"] = json.loads(json.dumps(SPEC["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    return s


class Tok:  # character-level stand-in for the tokenizer
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        ids = [[ord(c) for c in s] for s in text] if isinstance(text, list) else [ord(c) for c in text]
        return type("E", (), {"input_ids": ids})()


class FakeServer(Server):
    """Server.decide with a stub backend that returns fixed position probabilities for every job."""
    def __init__(self, pos_probs, soam=True, cal=None):
        self.name, self.temps, self.tok, self.letters, self.pos_probs = "basal-test", {}, Tok(), {}, pos_probs
        self.orders, self.soam, self.log, self.cal = 2, soam, None, cal or {}
        self.entries = 0
        self.backend = type("B", (), {"policies": {}})()

    async def _run(self, body):
        self.queue = asyncio.Queue()

        async def worker():
            while True:
                prompts, ids, fut, _ = await self.queue.get()
                self.entries += 1
                fut.set_result([self.probs_for(p, i) for p, i in zip(prompts, ids)])
        w = asyncio.get_running_loop().create_task(worker())
        await asyncio.sleep(0)
        try:
            return await self.decide(body)
        finally:
            w.cancel()


def _probs_for(self, prompt, ids):
    return self.pos_probs[: len(ids)] if not callable(self.pos_probs) else self.pos_probs(prompt, ids)


FakeServer.probs_for = _probs_for


def test_default_mode_prefers_cuda_over_mlx(monkeypatch):
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    assert default_mode() == "fast"


def test_default_mode_requires_both_mlx_packages_on_mps(monkeypatch):
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    assert default_mode() == "mps"


def test_benchmark_mps_default_avoids_eager_fp32(monkeypatch):
    monkeypatch.setattr(bench_module, "default_mode", lambda: "mlx")
    monkeypatch.setattr(bench_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench_module.torch.backends.mps, "is_available", lambda: True)
    assert bench_module.default_modes() == ["mps", "mlx"]


def test_benchmark_cuda_default_preserves_fp32_reference(monkeypatch):
    monkeypatch.setattr(bench_module, "default_mode", lambda: "fast")
    monkeypatch.setattr(bench_module.torch.cuda, "is_available", lambda: True)
    assert bench_module.default_modes() == ["eager-fp32", "fast"]


def test_benchmark_sync_and_memory_follow_backend_device(monkeypatch):
    import math
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(bench_module.torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(bench_module.torch.mps, "driver_allocated_memory", lambda: 2**30)
    cpu = SimpleNamespace(dev="cpu")
    mps = SimpleNamespace(dev=bench_module.torch.device("mps"))
    bench_module.timed(lambda: calls.append("work"), 1, cpu)
    assert calls == ["work"]
    assert math.isnan(bench_module.memory_gb(cpu))

    calls.clear()
    bench_module.timed(lambda: calls.append("work"), 1, mps)
    assert calls == ["mps", "work", "mps"]
    assert bench_module.memory_gb(mps) == 1.0


@pytest.mark.parametrize(("mode", "quant"), [
    ("eager", "fp8"), ("fast", "q8"), ("mps", "fp8"), ("mlx", "nvfp4"), ("vllm", "fp8"), ("gguf", "q8"),
    ("mlx", "fp8"), ("mps", "q8"), ("eager", "nvfp4"), ("sglang", "fp8"), ("ollama", "q8"), ("npu", "fp8"),
])
def test_unsupported_quant_fails_before_model_loading(monkeypatch, mode, quant):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", mode, "--quant", quant])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="not supported"):
        server_module.main()


def test_gguf_mode_needs_a_file_before_model_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "gguf"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--gguf"):
        server_module.main()


def test_npu_mode_needs_an_onnx_folder_before_model_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "npu"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--onnx"):
        server_module.main()


def test_benchmark_times_npu_prompts_cold():
    """The npu backend keeps its last prompt in the KV cache; a timed repeat must not reuse it."""
    from types import SimpleNamespace
    calls = []
    npu = SimpleNamespace(reset_prefix_cache=lambda: calls.append("reset"))
    bench_module.timed(lambda: calls.append("work"), 2, npu)
    assert calls == ["reset", "work", "reset", "work"]


def test_named_options_show_keys_by_default():
    assert named_options({"returns": "Returns", "it": "IT"}) == (["returns", "it"], ["returns: Returns", "it: IT"])
    assert named_options({"x": None, "y": "why"})[1] == ["x", "y: why"]
    keys, opts = named_options({"approve": {"requires_manager": False}, "reject": {"requires_manager": False}})
    assert opts == ['approve: {"requires_manager": false}', 'reject: {"requires_manager": false}']


def test_named_options_hide_only_on_request():
    assert named_options({"option_1": "Yes", "opcja_2": "Nie"}, "hide")[1] == ["Yes", "Nie"]
    assert named_options({"a": "same", "b": "same"}, "hide")[1] == ["a: same", "b: same"]   # still distinguishable
    with pytest.raises(ValueError):
        named_options({"a": "x", "b": "y"}, "maybe")


def test_structured_choices_give_distinct_prompts():
    q = to_items("Action requested: reject.", {"action": {"type": "choice", "instructions": "Return the action.",
                 "criteria": {"approve": {"requires_manager": False}, "reject": {"requires_manager": False}}}})[0]
    assert len(set(q["options"])) == 2 and q["options"][1].startswith("reject")


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "Dept?", "criteria": {"returns": "Returns", "it": "IT"}},
    {"type": "noul", "instructions": "Damaged?"},
    {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
])
def test_response_conforms_to_official_schema(q):
    srv = FakeServer([0.7, 0.2, 0.1])
    r = asyncio.run(srv._run({"model": "basal", "state": "Parcel arrived damaged.", "questions": {"q": q}}))
    jsonschema.validate(r, schema("SystemOneResponse"))
    assert isinstance(r["usage"]["input_tokens"], int)


def test_model_list_conforms_to_official_schema():
    jsonschema.validate(models_payload("basal-1.5-4.5B", "fast", {"0.99": None}), schema("ModelMetadataList"))


def swapped_prompts_differ(crit_a, crit_b, t="choice"):
    q = lambda c: to_items("Customer selected shipping service B.", {"q": {"type": t, "instructions": "Which code?",
                                                                            "criteria": c}})[0]["options"]
    return q(crit_a) != q(crit_b)


def test_swapping_keys_changes_what_the_model_sees():
    """Regression: whoever owns which description must reach the prompt (short codes, names mentioned in text)."""
    assert swapped_prompts_differ({"approve": "Handled by Alice", "reject": "Handled by Bob"},
                                  {"reject": "Handled by Alice", "approve": "Handled by Bob"})
    assert swapped_prompts_differ({"A": "Dispatch office: Warsaw", "B": "Dispatch office: Krakow"},
                                  {"B": "Dispatch office: Warsaw", "A": "Dispatch office: Krakow"})
    d1, d2 = "Processes approve/reject requests; handled by Alice", "Processes approve/reject requests; handled by Bob"
    assert swapped_prompts_differ({"approve": d1, "reject": d2}, {"reject": d1, "approve": d2})
    assert swapped_prompts_differ({"low": "minor", "high": "severe"}, {"high": "minor", "low": "severe"}, "score")


def test_soam_one_entry_per_request_same_answers():
    body = {"state": "Paczka dotarła uszkodzona, klient żąda zwrotu.", "questions": {
        "dept": {"type": "choice", "instructions": "Dział?", "criteria": {"zwroty": "Zwroty", "it": "IT", "hr": "HR"}},
        "dmg": {"type": "noul", "instructions": "Uszkodzona?"},
        "tags": {"type": "multi", "instructions": "Kategorie?", "criteria": {"d": "Uszkodzenie", "r": "Zwrot"}}}}
    a, b = FakeServer([0.6, 0.3, 0.1]), FakeServer([0.6, 0.3, 0.1], soam=False)
    ra, rb = asyncio.run(a._run(body)), asyncio.run(b._run(body))
    assert a.entries == 1 and b.entries == 4          # one packed row vs one entry per branch
    assert ra["answers"] == rb["answers"] and ra["usage"]["branches"] == 4


def test_act_minimum_expected_cost_and_defer():
    q = {"type": "act", "instructions": "Czy zwrot przysługuje?", "criteria": {"true": "Przysługuje", "false": "Nie"},
         "actions": {"approve": "", "reject": "", "human": ""},
         "costs": {"approve": {"true": 0, "false": 1240}, "reject": {"true": 300, "false": 0},
                   "human": {"true": 15, "false": 15}}}
    # position probabilities: orig order (true, false) and reversed (false, true) -> P(true) = 0.962
    srv = FakeServer(lambda p, ids: [0.962, 0.038] if "A. Przysługuje" in "".join(map(chr, p)) else [0.038, 0.962])
    ans = asyncio.run(srv._run({"state": "Zamówienie 1 240 zł, 6 dni.", "questions": {"r": q}}))["answers"]["r"]
    assert ans["action"] == "human" and abs(ans["expected_costs"]["approve"] - 47.12) < 1e-4
    assert ans["answer"] == "true" and ans["calibration"] == "unvalidated"
    srv = FakeServer(lambda p, ids: [0.995, 0.005] if "A. Przysługuje" in "".join(map(chr, p)) else [0.005, 0.995],
                     cal={"temperature_per_prim": {"noul": 1.0}, "thresholds": {"0.01": {"confidence": 0.999}}})
    ans = asyncio.run(srv._run({"state": "Zamówienie", "questions": {"r": dict(q, max_error=0.01)}}))["answers"]["r"]
    assert ans["refused"] == "approve" and ans["action"] == "human" and ans["calibration"] == "validated"


def test_act_short_form():
    q = {"type": "act", "instructions": "Dept?", "criteria": {"a": "A", "b": "B"}, "costs": {"wrong": 99, "defer": 1}}
    srv = FakeServer([0.9, 0.1])  # the same position probabilities in both orders -> canonical 0.5 / 0.5
    ans = asyncio.run(srv._run({"state": "x", "questions": {"q": q}}))["answers"]["q"]
    assert ans["action"] == "defer" and ans["expected_costs"]["a"] == 49.5


def test_multi_selection():
    labels = {"d": "Uszkodzenie", "r": "Zwrot", "i": "Faktura"}
    p_yes = {"Uszkodzenie": 0.9, "Zwrot": 0.7, "Faktura": 0.1}

    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        py = next(v for k, v in p_yes.items() if f"„{k}”" in text)
        return [py, 1 - py] if "A. Tak" in text else [1 - py, py]
    srv = FakeServer(probs)
    body = {"state": "Paczka uszkodzona, chcę zwrotu.", "questions": {"t": {"type": "multi", "instructions": "Które?",
                                                                            "criteria": labels, "max": 1}}}
    ans = asyncio.run(srv._run(body))["answers"]["t"]
    assert ans["selected"] == ["d"] and abs(ans["probabilities"]["r"] - 0.7) < 1e-6
    assert abs(ans["set_confidence"] - 0.9 * 0.3 * 0.9) < 1e-6


def test_facts_auto_appends_calendar_facts():
    seen = []

    def probs(prompt, ids):
        seen.append("".join(map(chr, prompt)))
        return [0.6, 0.4]
    srv = FakeServer(probs)
    body = {"state": "Doręczono 20 marca 2027 r., termin 7 dni.", "facts": "auto",
            "questions": {"q": {"type": "noul", "instructions": "Czy w terminie?"}}}
    asyncio.run(srv._run(body))
    assert any("Fakty pomocnicze" in s and "sobota" in s for s in seen)
    with pytest.raises(ValueError):
        asyncio.run(FakeServer([0.5, 0.5])._run(dict(body, facts="yes")))


# money facts: one fixture per pattern, and one negative each
CENNIK = ("Cennik: 174,00 zł za sztukę przy zamówieniu poniżej 20 szt.; 156,60 zł od 20 do 99 szt.; 139,20 zł od 100 szt. "
          "Dostawa kosztuje 39,00 zł, a przy wartości towaru co najmniej 2 000,00 zł jest bezpłatna. Klient zamawia 98 szt.")
VERDICT = ("przekracz", "mieści", "w terminie", "po terminie", "zasadn", "limit", "próg", "przysługuje")


def test_facts_euro_conversion():
    ls = facts("Limit: 2 000 000 euro. Kurs ten (1 października 2026 r.) wynosił 4,3770 zł. Przychód: 9 654 410,27 zł.")
    assert "2 000 000 euro × 4,3770 zł = 8 754 000,00 zł" in ls and "9 654 410,27 zł > 8 754 000,00 zł" in ls
    assert "139 000 euro × 4,2693 zł = 593 432,70 zł" in facts("Próg: 139 000 EUR; 1 EUR = 4,2693 zł.")
    assert facts("Kawa kosztowała 3,50 euro, a obiad 12 euro.") == []                                 # no rate
    assert not any(" euro × " in x for x in facts("Próg 100 000 euro. Kursy: 4,30 zł i 4,50 zł."))   # ambiguous


def test_facts_net_gross_and_comparisons():
    ls = facts("Stawka podatku VAT: 23%. Akceptacja od 1 000,00 zł, 3 000,00 zł i 32 000,00 zł. "
               "Wniosek: laptop — 1 szt. × 26 016,26 zł netto.")
    assert "26 016,26 zł netto × 1,23 = 32 000,00 zł brutto" in ls and "32 000,00 zł = 32 000,00 zł" in ls
    ls = facts("Stawki VAT: 23% i 8%. Próg 3 000,00 zł, 5 000,00 zł, 9 000,00 zł. Zakup 1 szt. × 2 910,00 zł netto.")
    assert not any(" netto × " in x for x in ls)                                                      # two rates
    assert facts("Kwota do zwrotu 78 565,34 zł.") == []                                               # nothing to compare


def test_facts_budget_products_sum():
    ls = facts("Budżet działu wynosi 50 000,00 zł, dotychczas wydano 41 250,00 zł. "
               "Zakup: 3 szt. × 1 250,00 zł netto; 1 usł. × 900,00 zł netto. Stawka VAT 23%.")
    assert {"budżet 50 000,00 zł − wydano 41 250,00 zł = 8 750,00 zł", "3 × 1 250,00 zł = 3 750,00 zł",
            "3 750,00 zł + 900,00 zł = 4 650,00 zł", "5 719,50 zł < 8 750,00 zł"} <= set(ls)
    assert not any(x.startswith("budżet") for x in facts("Wydano decyzję o zwrocie 1 200,00 zł. Opłata 100,00 zł."))


def test_facts_price_list_tier():
    ls = facts(CENNIK)
    assert {"20 ≤ 98 ≤ 99; 98 × 156,60 zł = 15 346,80 zł", "15 346,80 zł + 39,00 zł = 15 385,80 zł",
            "15 346,80 zł > 2 000,00 zł"} <= set(ls)
    assert not any(" ≤ " in x for x in facts(CENNIK.replace("98 szt.", "98 szt., a potem 120 szt.")))  # 2 quantities
    out = inject(CENNIK)
    assert not any(w in out[out.index(HEADER):] for w in VERDICT)                                     # numbers only


# VAT-line regressions
RATES = ["Próg 216 000 euro, przeliczany po kursie 4,31 zł za euro. Wartość 200 000,00 zł netto, VAT 23%.",
         "Kurs ten wynosił 4,31 zł. Próg 216 000 euro. Wartość 200 000,00 zł netto, VAT 23%.",
         "Próg 216 000 euro (4,2586 zł za 1 euro). Wartość 200 000,00 zł netto, VAT 23%.",
         "Próg 216 000 euro; 1 EUR = 4,31 zł. Wartość 200 000,00 zł netto, VAT 23%."]


def test_vat_line_reads_amounts_without_thousands_separator():   # regression: "2026,00 zł" was read as "0,00 zł"
    ls = facts("Faktura na kwotę 2026,00 zł; w tym VAT 23%.")
    assert ls == ["2 026,00 zł × 1,23 = 2 491,98 zł; 2 026,00 zł ÷ 1,23 = 1 647,15 zł"]
    assert facts("Opłata 1500 zł, VAT 8%.")[0].startswith("1 500,00 zł × 1,08 = 1 620,00 zł")


def test_vat_line_skips_exchange_rates():                         # regression: "4,31 zł" got a net/gross line
    for s in RATES:
        ls = facts(s)
        assert not any(x.startswith(("4,31 zł", "4,2586 zł", "4,26 zł")) for x in ls), ls
        assert any(x.startswith("200 000,00 zł × 1,23 = 246 000,00 zł") for x in ls)
    assert facts("Cena 4,31 zł netto, VAT 23%.")[0].startswith("4,31 zł × 1,23")    # a price, not a rate: kept


# VAT rate needs VAT context
COURT_FEE = ("Pozew o zapłatę 19 726,00 zł. Opłata: ponad 15 000 do 20 000 zł — 1000 zł; ponad 20 000 zł — 5% wartości "
             "przedmiotu sporu, nie więcej niż 200 000 zł.")


def test_percentage_is_a_vat_rate_only_with_vat_in_its_sentence():   # regression: "5% wartości sporu" got ×1,05 lines
    assert facts(COURT_FEE) == []
    assert facts("Faktura na 4 877,00 zł. Odsetki wynoszą 13,5% w skali roku. Stopa referencyjna NBP: 5%.") == []
    assert facts("Faktura na 1 000,00 zł netto, VAT 23%.") == ["1 000,00 zł × 1,23 = 1 230,00 zł; 1 000,00 zł ÷ 1,23 = 813,01 zł"]
    assert facts("Faktura: 1 000,00 zł netto. Stawka podatku od towarów i usług: 8%.")[0].startswith("1 000,00 zł × 1,08")


def test_max_len_flag_for_vllm_and_sglang():
    assert parser().parse_args([]).max_len == 4096
    assert parser().parse_args(["--mode", "sglang", "--max-len", "32768"]).max_len == 32768
