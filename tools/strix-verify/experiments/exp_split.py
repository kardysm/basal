"""Does a request's answer change when its readout rows are split across two gather blocks (BT=8) of the last
layer's attention? Tiny random Llama, the patched backend, alone vs after other requests."""
import random
import sys

import torch

sys.path.insert(0, sys.argv[1] + "/tests")
from test_strix import make, request, tiny  # noqa: E402

be = make(tiny())
BT = be.ATTN["BT"]
random.seed(11)
state = [random.randrange(2, 60) for _ in range(60)]
target = request(state, 5)  # 5 questions x 2 orders = 10 readout rows
alone = be.run_shared([target])[0]

# others: (length, number of questions); each other request reads out 2 rows per question
for others in [(), ((37, 1),), ((37, 2),), ((37, 3),), ((13, 1), (20, 1)), ((13, 1), (20, 1), (25, 1)),
               ((13, 3),), ((13, 2), (20, 1)), ((40, 4),), ((40, 1), (41, 2))]:
    groups = []
    for n, nq in others:
        st = [2 + i % 50 for i in range(n)]
        groups.append(request(st, nq, first=10))
    groups.append(target)
    units = be._units(groups)
    idx = be._chunks_tok(units)[0]
    reads_before = sum(len(units[k][1]) for k in idx[: [units[k][2] for k in idx].index(len(others))])
    together = be.run_shared(groups)[-1]
    same = [torch.equal(torch.tensor(a), torch.tensor(b)) for a, b in zip(alone, together)]
    first_block = [(reads_before + i) // BT for i in range(len(alone))]
    print(f"others={others!s:40s} reads before={reads_before:2d} blocks={first_block} "
          f"equal={'all' if all(same) else [i for i, s in enumerate(same) if not s]}")
