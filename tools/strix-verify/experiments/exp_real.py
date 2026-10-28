"""Real basal-2.0 weights, patched strix backend: does a 5-question case answer the same alone and packed with 7
others? Synthetic token ids of real-case size. Prints every differing (case, question)."""
import random
import sys

import torch

from basal.strix import StrixHaloFastBackend

M = sys.argv[1]
be = StrixHaloFastBackend(M, warm=False)
LETTERS = [[32, 33], [32, 33, 34], [35, 36, 37, 38], [32, 33], [39, 40, 41, 42, 43]]


def case(rng):
    state = [rng.randrange(1000, 30000) for _ in range(rng.randrange(300, 650))]
    toks = [state + [rng.randrange(1000, 30000) for _ in range(rng.randrange(25, 70))] for _ in range(5)]
    return toks, [list(x) for x in LETTERS]


total = 0
for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
    rng = random.Random(trial)
    cases = [case(rng) for _ in range(8)]
    alone = [be.run_shared([c])[0] for c in cases]
    together = be.run_shared(cases)
    units = be._units(cases)
    order = be._chunks_tok(units)
    rank = {units[k][2]: (ci, j) for ci, idx in enumerate(order) for j, k in enumerate(idx)}
    for c in range(8):
        diff = [q for q in range(5) if not torch.equal(torch.tensor(alone[c][q]), torch.tensor(together[c][q]))]
        if diff:
            total += 1
            dp = max(abs(a - b) for q in diff for a, b in zip(alone[c][q], together[c][q]))
            print(f"trial {trial} case {c} (chunk, place {rank[c]}, {len(units[c][0][0])} tokens): questions {diff}"
                  f" differ, max |dp| {dp:.2e}", flush=True)
print(f"{total} differing cases", flush=True)
