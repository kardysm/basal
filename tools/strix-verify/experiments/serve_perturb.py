"""basal server (strix) whose forward moves one number by one bf16 step: hidden element 0 of the first token after
the embedding (the <s> every prompt starts with). Everything else is unchanged."""
import sys
import torch
import torch.nn.functional as TF
from basal import strix


class _F:
    def __getattr__(self, name):
        return getattr(TF, name)

    @staticmethod
    def embedding(ids, w):
        h = TF.embedding(ids, w)
        h.view(torch.int16)[0, 0] += 1  # one bf16 step (one ulp) of one value
        return h


strix.F = _F()
from basal.server import main  # noqa: E402
sys.argv[0] = "basal-serve"
main()
