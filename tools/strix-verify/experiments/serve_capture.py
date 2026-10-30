"""basal server whose strix backend records every run_shared call (groups in, probabilities out) to $CAP."""
import os
import pickle
import sys
import threading

from basal import strix

_orig = strix.StrixHaloFastBackend.run_shared
_log = open(os.environ["CAP"], "ab")
_lock = threading.Lock()


def run_shared(self, groups, policy=None, **kw):
    out = _orig(self, groups, policy, **kw)
    with _lock:
        pickle.dump((groups, out), _log)
        _log.flush()
    return out


strix.StrixHaloFastBackend.run_shared = run_shared
from basal.server import main  # noqa: E402

sys.argv[0] = "basal-serve"
main()
