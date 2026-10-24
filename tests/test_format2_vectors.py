"""docs/format2_vectors.jsonl stays the engine's format 2: every request, rendered again with this engine, gives the
same prompt strings, option texts, languages and option orders (the chat template of the basal models as a stub). With
BASAL_TOKENIZER set to a basal model directory, the token ids, state ends and code ids are checked too."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from basal.prompt import CODES, PREFILL, letter_ids, render
from basal.server import Server, to_items

VECTORS = Path(__file__).resolve().parents[1] / "docs" / "format2_vectors.jsonl"


class TemplateTok:
    """The basal chat template (docs/FORMAT2.md, section 3.4) without a tokenizer."""
    def apply_chat_template(self, msgs, **_):
        return "<s>" + "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + \
            "<|im_start|>assistant\n"


def prompts(tok, rec):
    for item in to_items(rec["request"]["state"], rec["request"]["questions"], len(CODES)):
        multi = item["type"] == "multi" or item.get("multi")
        for bi, (question, options) in enumerate(item["branches"]):
            k = len(options)
            for perm in [list(range(k)), list(range(k))[::-1]][: rec["orders"]]:
                opts = [options[c] for c in perm]
                yield item, question, opts, dict(
                    question=item["name"], label=item["keys"][bi] if multi else None, lang=item["lang"],
                    option_order=perm, options=opts, prompt=render(tok, item["state"], question, opts, item["lang"], 2))


def records():
    return [json.loads(line) for line in VECTORS.read_text(encoding="utf-8").splitlines()]


def test_prompts_are_the_engines():
    recs = records()
    assert len(recs) >= 30 and len({r["id"] for r in recs}) == len(recs)
    for rec in recs:
        assert rec["prefill"] == PREFILL
        got = [p for *_, p in prompts(TemplateTok(), rec)]
        assert got == [{k: p[k] for k in got[0]} for p in rec["prompts"]], rec["id"]


@pytest.mark.skipif(not os.environ.get("BASAL_TOKENIZER"), reason="BASAL_TOKENIZER (a basal model directory) not set")
def test_token_ids_state_ends_and_codes():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["BASAL_TOKENIZER"])
    srv = SimpleNamespace(tok=tok, fmt=2)
    for rec in records():
        for (item, question, opts, mine), p in zip(prompts(tok, rec), rec["prompts"]):
            assert mine["prompt"] == p["prompt"], rec["id"]
            assert tok(p["prompt"], add_special_tokens=False).input_ids == p["input_ids"], rec["id"]
            assert letter_ids(tok, p["prompt"], len(opts), fmt=2) == p["code_ids"], rec["id"]
            assert Server.state_cut(srv, dict(item, question=question, options=opts)) == p["state_end"], rec["id"]
