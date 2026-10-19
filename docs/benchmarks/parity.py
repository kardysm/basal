"""Parity of basal engines against a reference engine on LocalLLaMA/typed-decisions (Apache-2.0).

The test split has 400 cases (one state, five typed questions: choice, yes / no, score) and 2,000 decisions. Each case
is sent as one /v1/systemone request with all five questions, as the dataset card prescribes.

  # 1. cases (needs pyarrow: uv run --no-project --with pyarrow python parity.py cases ...)
  python parity.py cases all/test-00000-of-00001.parquet cases.jsonl
  # 2. answers from each server, one at a time (basal-run ships with basal)
  basal-run --input cases.jsonl --output eager.jsonl --url http://127.0.0.1:8031/v1/systemone
  basal-run --input cases.jsonl --output strix.jsonl --url http://127.0.0.1:8031/v1/systemone
  # 3. compare (stdlib only); the first answer file is the reference
  python parity.py compare cases.jsonl eager.jsonl strix.jsonl --out parity.json

A decision is one question of one case. Its answer is the option with the highest calibrated probability (yes / no:
true when P(true) > 0.5). Reported per engine against the reference: decisions with the same answer, mean and max
|Δp| over the options' calibrated probabilities, every flip with the reference's top-2 margin, and accuracy of both
against the gold label (the argmax of the dataset's gold distribution).
"""
import argparse
import json
import statistics
import sys


def cases(src, dst):
    import pyarrow.parquet as pq

    rows = pq.read_table(src).to_pylist()
    with open(dst, "w") as f:
        for r in rows:
            gold = json.loads(r["gold"])
            case = {"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
                    "questions": json.loads(r["questions"]),
                    "gold": {q: g["label"] for q, g in gold.items()}}
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"{len(rows)} cases, {sum(r['n_questions'] for r in rows)} decisions -> {dst}")


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def probs(answer):
    """{option: calibrated probability} of one answer; yes / no as {"true": p, "false": 1 - p}."""
    p = answer["probabilities"]
    return {str(k): float(v) for k, v in p.items()}


def top(p):
    return max(p, key=p.get)


def margin(p):
    v = sorted(p.values(), reverse=True)
    return v[0] - v[1] if len(v) > 1 else 1.0


def decisions(case_rows, answer_rows):
    """{(case id, question): (type, probabilities)}; a case whose request failed is missing."""
    out, failed = {}, []
    for c, a in zip(case_rows, answer_rows):
        if a.get("id") != c["id"]:
            raise SystemExit(f"answer file out of order: {a.get('id')} != {c['id']}")
        if not a.get("answers"):
            failed.append(c["id"])
            continue
        for q, ans in a["answers"].items():
            out[(c["id"], q)] = (ans.get("type", c["questions"][q]["type"]), probs(ans))
    return out, failed


def ece(ds, gold, bins=15):
    """Expected calibration error of the top option against the gold label (equal-width bins)."""
    acc, conf, n = [0] * bins, [0.0] * bins, [0] * bins
    for k, (_, p) in ds.items():
        c = max(p.values())
        b = min(int(c * bins), bins - 1)
        n[b] += 1
        conf[b] += c
        acc[b] += top(p) == gold[k]
    return sum(abs(acc[b] - conf[b]) for b in range(bins) if n[b]) / sum(n)


def compare(cases_path, ref_path, others, out_path):
    cs = load(cases_path)
    gold = {(c["id"], q): str(g) for c in cs for q, g in c["gold"].items()}
    ref, ref_failed = decisions(cs, load(ref_path))
    wide = {k for k, (_, p) in ref.items() if margin(p) >= 0.2}  # the reference is not near a tie
    report = {"cases": len(cs), "decisions": len(gold), "reference": ref_path, "reference_failed": ref_failed,
              "reference_accuracy": round(sum(top(p) == gold[k] for k, (_, p) in ref.items()) / len(ref), 4),
              "reference_ece": round(ece(ref, gold), 4), "reference_margin_ge_0.2": len(wide),
              "engines": {}}
    for path in others:
        rows = load(path)
        got, failed = decisions(cs, rows)
        keys = sorted(set(ref) & set(got))
        dp = [abs(got[k][1].get(o, 0.0) - v) for k in keys for o, v in ref[k][1].items()]
        flips = []
        for k in keys:
            (t, pr), (_, pg) = ref[k], got[k]
            if top(pr) != top(pg):
                flips.append({"case": k[0], "question": k[1], "type": t, "reference": top(pr),
                              "engine": top(pg), "reference_margin": round(margin(pr), 4),
                              "engine_margin": round(margin(pg), 4), "gold": gold[k]})
        by_type = {}
        for k in keys:
            t = ref[k][0]
            d = by_type.setdefault(t, {"decisions": 0, "same": 0})
            d["decisions"] += 1
            d["same"] += top(ref[k][1]) == top(got[k][1])
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        report["engines"][path] = {
            "failed": failed, "compared": len(keys), "same": len(keys) - len(flips),
            "agreement": round((len(keys) - len(flips)) / len(keys), 4),
            "mean_abs_dp": round(statistics.fmean(dp), 5), "max_abs_dp": round(max(dp), 4),
            "accuracy": round(sum(top(got[k][1]) == gold[k] for k in keys) / len(keys), 4),
            "ece": round(ece({k: got[k] for k in keys}, gold), 4),
            "flips_at_reference_margin_ge_0.2": sum(f["reference_margin"] >= 0.2 for f in flips),
            "agreement_at_reference_margin_ge_0.2": round(
                1 - sum(f["reference_margin"] >= 0.2 for f in flips) / max(1, len(wide & set(keys))), 4),
            "max_reference_margin_of_a_flip": max((f["reference_margin"] for f in flips), default=None),
            "by_type": by_type, "flips": flips,
            "latency_ms_p50": statistics.median(lat) if lat else None,
        }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    for path, e in report["engines"].items():
        print(f"{path}: {e['same']}/{e['compared']} same as the reference ({e['agreement']}), mean |dp| "
              f"{e['mean_abs_dp']}, max {e['max_abs_dp']}, accuracy {e['accuracy']} (reference "
              f"{report['reference_accuracy']}), ECE {e['ece']} (reference {report['reference_ece']}), flips "
              f"{len(e['flips'])}, at reference margin >= 0.2: {e['flips_at_reference_margin_ge_0.2']} of "
              f"{report['reference_margin_ge_0.2']}, largest reference margin of a flip "
              f"{e['max_reference_margin_of_a_flip']}, failed {len(e['failed'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cases", help="parquet of the dataset -> cases JSONL (request body + gold labels)")
    c.add_argument("parquet")
    c.add_argument("out")
    m = sub.add_parser("compare", help="answer files of basal-run: the first is the reference")
    m.add_argument("cases")
    m.add_argument("reference")
    m.add_argument("engines", nargs="+")
    m.add_argument("--out", default="parity.json")
    a = ap.parse_args()
    if a.cmd == "cases":
        cases(a.parquet, a.out)
    else:
        compare(a.cases, a.reference, a.engines, a.out)


if __name__ == "__main__":
    sys.exit(main())
