"""Agreement tables and the G1 / G2 gates of basal-strix-verified-plan.md, over basal-run answer files.

  python3 basal-strix-gates.py --dir ~/strix-verify --ref eager.jsonl \
      eager-packed.jsonl strix.jsonl strix-c8.jsonl strix-det-c8.jsonl

Every file is a JSONL of basal-run answers, one line per case ({"id", "answers": {"<question>": {"probabilities":
{...}}, ...}}); a typed-decisions case has five questions. A decision is one question of one case (as in parity.py and
pairwise.py), and its answer is the option with the highest probability. Flips are decisions whose answer differs from
the reference; the interval is Wilson 95% on the flip rate, reported as agreement (1 - rate). G2 needs only
strix.jsonl and runs without the reference. --json writes the same numbers to a file.
"""
import argparse
import json
import math
from pathlib import Path

BAR = 0.995            # the maintainers' bar: 99.5% of answers equal to --mode eager bf16
MAX_FLIPS_1000 = 5     # 99.5% of 1,000 decisions


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (proportion form)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def binom_cdf(k, n, p):
    """P(X <= k), X ~ Binomial(n, p): the chance that a run of n decisions has at most k flips."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(min(k, n) + 1))


def load(path):
    """JSONL -> ({(case id, question): {option: probability}}, ids of cases without answers, number of cases)."""
    out, failed, cases = {}, [], 0
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        cases += 1
        rec = json.loads(line)
        rid = rec.get("id", ln - 1)
        ans = rec.get("answers") or {}
        if not ans:
            failed.append(rid)
            continue
        for q, a in ans.items():
            out[(rid, q)] = {str(k): float(v) for k, v in a["probabilities"].items()}
    return out, failed, cases


def top(p):
    return max(p, key=p.get)


def margin(p):
    """Top-1 minus top-2 probability: how close the reference decision was."""
    v = sorted(p.values(), reverse=True)
    return (v[0] - v[1]) if len(v) > 1 else 1.0


def compare(ref, other):
    """Decisions in both, the ones whose answer differs, and the ones with the same answer but other probabilities."""
    keys = [k for k in ref if k in other]
    flips = [k for k in keys if top(ref[k]) != top(other[k])]
    prob = [k for k in keys if top(ref[k]) == top(other[k]) and other[k] != ref[k]]
    return keys, flips, prob


def label(k):
    return f"{k[0]}/{k[1]}"


def band(m):
    return "<0.02" if m < 0.02 else "0.02-0.05" if m < 0.05 else "0.05-0.10" if m < 0.10 else ">=0.10"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="answer files to compare against --ref (names, relative to --dir)")
    ap.add_argument("--dir", default=".", help="directory holding the answer files")
    ap.add_argument("--ref", default="eager.jsonl", help="reference answer file (default: eager.jsonl)")
    ap.add_argument("--json", default=None, help="write the numbers to this JSON file")
    a = ap.parse_args()
    d = Path(a.dir)
    ref_path = d / a.ref
    got, out = {}, {"ref": a.ref, "dir": str(d), "files": {}}
    if not ref_path.exists():
        print(f"no reference file {ref_path}: agreement tables and G1 skipped")
    else:
        ref, bad, ncases = load(ref_path)
        out["decisions"], out["cases"] = len(ref), ncases
        print(f"reference: {a.ref}  {ncases} cases, {len(ref)} decisions, {len(bad)} cases without answers")
        if bad:
            print(f"  the reference has {len(bad)} error lines: the numbers below are not usable until it is rerun")
        print()
        for name in a.files:
            p = d / name
            if not p.exists():
                print(f"{name:22s} missing")
                continue
            other, failed, _ = load(p)
            keys, flips, _ = compare(ref, other)
            n, k = len(keys), len(flips)
            lo, hi = wilson(k, n)
            agree = 1 - k / n if n else float("nan")
            ppass = binom_cdf(MAX_FLIPS_1000, 1000, k / n) if n else float("nan")
            bands = {}
            for key in keys:
                b = band(margin(ref[key]))
                ok, tot = bands.get(b, (0, 0))
                bands[b] = (ok + (top(ref[key]) == top(other[key])), tot + 1)
            got[name] = dict(n=n, flips=k)
            out["files"][name] = dict(decisions=n, flips=k, agreement=agree, agreement_ci=[1 - hi, 1 - lo],
                                      flip_rate=k / n if n else None, p_pass_1000=ppass, failed_cases=len(failed),
                                      bands={b: dict(agree=v[0], of=v[1]) for b, v in bands.items()},
                                      flip_ids=[label(f) for f in flips[:200]])
            print(f"{name:22s} n={n:5d}  flips={k:4d}  agree={agree:.4f}  "
                  f"95% {1 - hi:.4f}-{1 - lo:.4f}  P(pass on 1,000)={ppass:.2f}"
                  + (f"  FAILED CASES {len(failed)}" if failed else ""))
            for b in ("<0.02", "0.02-0.05", "0.05-0.10", ">=0.10"):
                if b in bands:
                    ok, tot = bands[b]
                    print(f"{'':22s}   reference margin {b:9s} {ok}/{tot} = {ok / tot:.4f}")
        print()

    if "eager-packed.jsonl" in got and "strix.jsonl" in got:
        F = got["eager-packed.jsonl"]["flips"] / got["eager-packed.jsonl"]["n"]
        S = got["strix.jsonl"]["flips"] / got["strix.jsonl"]["n"]
        print(f"G1  F (eager packed: bf16 noise of the reference's own packing) = {F * 100:.2f}%   "
              f"S (strix one at a time) = {S * 100:.2f}%   S - F = {(S - F) * 100:+.2f} pt")
        if S - F >= 0.003:
            verdict = "strix has error of its own beyond bf16 noise -> phase 2, then phase 3 is the main path"
        elif F >= 0.004:
            verdict = ("the bar is at or below the bf16 floor on this set: no packed bf16 engine passes reliably "
                       "-> phase 2, then phase 5 (evidence)")
        elif F < 0.003:
            verdict = "strix already passes most of the time -> phase 2, then phase 5"
        else:
            verdict = ("between the rows (0.3% <= F < 0.4%): more decisions, or the maintainers' own set, decide "
                       "whether F is at or below the bar")
        print(f"    {verdict}")
        ppass = binom_cdf(MAX_FLIPS_1000, 1000, S)
        print(f"    at S={S * 100:.2f}% a 1,000-decision run has at most {MAX_FLIPS_1000} flips with p={ppass:.2f};"
              f" the bar needs about 0.3%")
        out["G1"] = dict(F=F, S=S, verdict=verdict, p_pass_1000=ppass)
    else:
        print("G1  needs eager.jsonl, eager-packed.jsonl and strix.jsonl")

    if not (d / "strix.jsonl").exists():
        print("G2  needs strix.jsonl")
    # identity checks, every probability equal: G2 = batched vs one at a time; EXACT = --near-ties eager with every
    # decision computed again (--near-tie-margin 1.01) vs the eager reference
    for tag, base_name, name in (("G2", "strix.jsonl", "strix-c8.jsonl"), ("G2", "strix.jsonl", "strix-det-c8.jsonl"),
                                 ("G2", "strix-ties.jsonl", "strix-ties-c8.jsonl"),
                                 ("EXACT", "eager.jsonl", "strix-ties-all.jsonl")):
        if not ((d / base_name).exists() and (d / name).exists()):
            continue
        base, bfail, _ = load(d / base_name)
        other, ofail, _ = load(d / name)
        keys, dflips, pflips = compare(base, other)
        ok = not dflips and not pflips and not bfail and not ofail and len(keys) == len(base) == len(other)
        print(f"{tag:5s} {name} vs {base_name}: {len(keys)} decisions, {len(dflips)} answers differ, "
              f"{len(pflips)} same answer with different probabilities"
              + (f", failed cases {len(bfail)} / {len(ofail)}" if bfail or ofail else "")
              + f" -> {'PASS' if ok else 'FAIL'}")
        out.setdefault(tag, {})[name] = dict(base=base_name, decisions=len(keys),
                                             decision_diffs=[label(k) for k in dflips[:50]],
                                             probability_diffs=[label(k) for k in pflips[:50]], passed=ok)

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
