#!/usr/bin/env bash
# basal-strix-verify.sh -- phases 0-1 and the G1 / G2 gates of basal-strix-verified-plan.md.
#
# One server at a time on 127.0.0.1:$PORT, the six answer files of phase 1, then the agreement tables and the gates.
# Nothing else on the box is touched: the live :8022 server and basal-ask keep running.
#
#   ./basal-strix-verify.sh check          # the environment the plan assumes (python, worktree version, model, GPU)
#   ./basal-strix-verify.sh setup          # phase 0: parity.py from the fork branch, cases-test.jsonl / cases-train.jsonl
#   ./basal-strix-verify.sh all            # phase 1 on the test split: six configs, ~20-40 min of machine time
#   TRAIN=1 ./basal-strix-verify.sh all    # adds the four train-split runs
#   ./basal-strix-verify.sh ties           # --near-ties eager: strix-ties, strix-ties-c8 (8 in flight) and
#                                          #   strix-ties-all (every decision through eager: must equal eager.jsonl)
#   ./basal-strix-verify.sh gates          # pairwise.py (if present) + G1 / G2 from the answer files
#   ./basal-strix-verify.sh status         # what is in $WORK
#   ./basal-strix-verify.sh sdpa           # which SDPA kernel the eager reference really runs (one profiled forward)
#
# Env: PY, REPO (code under test), MODEL, PORT, WORK, PW (pairwise.py), SNAP (dataset snapshot), DRY=1, FORCE=1.
set -euo pipefail

PY=${PY:-$HOME/basal-env/bin/python}
REPO=${REPO:-$HOME/projects/basal-strix-warm}
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--Remek--basal-2.0-4.5B/snapshots/e5ba9149760d99b73b1963d8e0ccf948b58afacb}
PORT=${PORT:-8031}
WORK=${WORK:-$HOME/strix-verify}
PW=${PW:-/run/media/kardysm/4TB/projects/dangerous/basal-systems/tools/strix/pairwise.py}
SNAP=${SNAP:-$HOME/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots/7189f46b02248bd2569df7325c9df8e3b24e90a7/all}
NAME=${NAME:-basal-2.0-4.5B}
HOST=127.0.0.1
URL=http://$HOST:$PORT
PY3=${PY3:-$(command -v python3 || true)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GATES=$HERE/basal-strix-gates.py
DRY=${DRY:-0}
FORCE=${FORCE:-0}
TRAIN=${TRAIN:-0}
SERVER_PID=

say() { printf '\n== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then echo "   [dry] $*"; else "$@"; fi; }

# -- the environment the plan assumes ----------------------------------------------------------------------------
check_env() {
    [ -x "$PY" ] || die "no python at PY=$PY (the plan's env: \$HOME/basal-env/bin/python)"
    [ -d "$REPO/basal" ] || die "no worktree at REPO=$REPO (git worktree of strix-warm-shapes)"
    [ -f "$MODEL/config.json" ] || die "no model at MODEL=$MODEL"
    export PYTHONPATH=$REPO
    local v repo_real
    # -P everywhere: without it `python -c` puts the current directory first, and a basal checkout there wins over $REPO
    v=$("$PY" -P -c 'import basal, os; print(basal.__version__, os.path.realpath(basal.__file__))') || die "import basal failed from $REPO"
    echo "   code under test: $v"
    repo_real=$(realpath "$REPO")
    case "$v" in
        *" $repo_real/"*) ;;
        *) die "basal is not imported from REPO=$REPO ($repo_real)" ;;
    esac
    case "$v" in
        2.4.1\ *) ;;
        *) echo "   warning: expected 2.4.1 (the plan pins strix-warm-shapes @ 7f4628f)" ;;
    esac
    if ! "$PY" - 2>/dev/null <<'PY'
import torch, triton
print(f"   torch {torch.__version__}  hip {torch.version.hip}  device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"   triton {triton.__version__}")
PY
    then echo "   warning: torch / triton not importable from $PY"; fi
    [ -x "$PY3" ] || echo "   warning: no python3 for the gate tables (PY3=$PY3)"
}

# -- phase 0 -----------------------------------------------------------------------------------------------------
pyarrow_run() {  # run a script that needs pyarrow, with uv when there is one
    if command -v uv >/dev/null; then
        run uv run --no-project --with pyarrow python "$@"
    else
        run "$PY" "$@"
    fi
}

cmd_setup() {
    mkdir -p "$WORK/logs"
    if [ ! -s "$WORK/parity.py" ]; then
        say "parity.py from branch strix-parity-data"
        git -C "$REPO" show strix-parity-data:docs/benchmarks/parity.py > "$WORK/parity.py" 2>/dev/null \
            || git -C /run/media/kardysm/4TB/projects/basal show strix-parity-data:docs/benchmarks/parity.py > "$WORK/parity.py" 2>/dev/null \
            || die "cannot read parity.py (branch strix-parity-data) from $REPO or the main checkout"
    fi
    if [ -f "$WORK/parity.py" ]; then echo "   $WORK/parity.py"; fi
    for split in test train; do
        local pq="$SNAP/$split-00000-of-00001.parquet" cases="$WORK/cases-$split.jsonl"
        [ -f "$pq" ] || die "no $pq (set SNAP to the typed-decisions snapshot's all/ directory)"
        if [ -s "$cases" ] && [ "$FORCE" != 1 ]; then
            echo "   $cases exists, skipping (FORCE=1 to rebuild)"
            continue
        fi
        say "cases: $split"
        pyarrow_run "$WORK/parity.py" cases "$pq" "$cases"
        if [ -f "$cases" ]; then echo "   $(wc -l < "$cases") lines -> $cases"; fi
    done
    cat <<'EOF'

   phase 0, item 3 (by hand): check basal-2.0's model card for whether typed-decisions *train* was in its training
   or calibration data. If it was, report the train numbers separately and never as the headline.
EOF
}

# -- phase 1 -----------------------------------------------------------------------------------------------------
set_conf() {  # name -> FLAGS, CONC
    case "$1" in
        eager)        CONC=1; FLAGS=(--mode eager --soam off --max-batch 1) ;;
        eager-fp32)   CONC=1; FLAGS=(--mode eager --dtype float32 --soam off --max-batch 1) ;;
        eager-packed) CONC=1; FLAGS=(--mode eager) ;;
        strix)        CONC=1; FLAGS=(--mode strix) ;;
        strix-c8)     CONC=8; FLAGS=(--mode strix --wait-ms 2) ;;  # 2 ms: every forward holds several requests
        strix-det-c8) CONC=8; FLAGS=(--mode strix --deterministic) ;;
        strix-ties)     CONC=1; FLAGS=(--mode strix --near-ties eager) ;;
        strix-ties-c8)  CONC=8; FLAGS=(--mode strix --near-ties eager --wait-ms 2) ;;
        strix-ties-all) CONC=1; FLAGS=(--mode strix --near-ties eager --near-tie-margin 1.01) ;;  # every decision
        *) return 1 ;;
    esac
}

out_name() {  # name, split -> answer file
    if [ "$2" = test ]; then echo "$1.jsonl"; else echo "$1-$2.jsonl"; fi
}

start_server() {  # log tag, flags...
    local tag=$1; shift
    local log="$WORK/logs/$tag.log"
    echo "   server:$* (log $log)"
    if [ "$DRY" = 1 ]; then echo "   [dry] PYTHONPATH=$REPO $PY -P -c 'from basal.server import main; main()' --model $MODEL --name $NAME --host $HOST --port $PORT $*"; return 0; fi
    ( export PYTHONPATH="$REPO"; exec "$PY" -P -c 'from basal.server import main; main()' \
        --model "$MODEL" --name "$NAME" --host "$HOST" --port "$PORT" "$@" ) >"$log" 2>&1 &
    SERVER_PID=$!
    local i
    for i in $(seq 1 900); do
        if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
            echo "   up after ${i}s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "   server died at start-up:"; tail -30 "$log" >&2; SERVER_PID=; return 1
        fi
        if [ $((i % 30)) -eq 0 ]; then echo "   ... loading (${i}s)"; fi
        sleep 1
    done
    echo "   server did not answer /health in 900s:"; tail -30 "$log" >&2; return 1
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local i
    for i in $(seq 1 40); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
}
trap 'stop_server' EXIT INT TERM

run_one() {  # name, split
    local name=$1 split=$2 cases out
    set_conf "$name" || die "unknown config $name"
    cases="$WORK/cases-$split.jsonl"
    out="$WORK/$(out_name "$name" "$split")"
    [ -f "$cases" ] || die "no $cases (run: $0 setup)"
    if [ -s "$out" ] && [ "$FORCE" != 1 ]; then
        echo "   $out exists, skipping (FORCE=1 to redo)"
        return 0
    fi
    say "$name ($split split, ${CONC} in flight)"
    start_server "$name-$split" "${FLAGS[@]}" || die "server failed for $name"
    if [ "$DRY" = 1 ]; then
        echo "   [dry] PYTHONPATH=$REPO $PY -P -c 'from basal.run import main; main()' --input $cases --output $out --url $URL/v1/systemone --concurrency $CONC"
    else
        PYTHONPATH=$REPO "$PY" -P -c 'from basal.run import main; main()' \
            --input "$cases" --output "$out" --url "$URL/v1/systemone" --concurrency "$CONC"
        local n err
        n=$(wc -l < "$out"); err=$(grep -c '"error"' "$out" || true)
        echo "   $out: $n lines, ${err:-0} errors, $(wc -l < "$cases") cases"
        [ "$n" -eq "$(wc -l < "$cases")" ] || echo "   WARNING: line count differs from the case file"
        [ "${err:-0}" -eq 0 ] || echo "   WARNING: $err lines carry an error: the gates below are not usable until rerun"
    fi
    stop_server
    sleep 2  # let the GPU settle between configs
}

cmd_run() {
    [ $# -eq 1 ] || die "usage: $0 run eager|eager-fp32|eager-packed|strix|strix-c8|strix-det-c8|strix-ties|strix-ties-c8|strix-ties-all"
    check_env
    mkdir -p "$WORK/logs"
    run_one "$1" test
}

cmd_all() {
    check_env
    mkdir -p "$WORK/logs"
    local n
    for n in eager eager-fp32 eager-packed strix strix-c8 strix-det-c8; do run_one "$n" test; done
    if [ "$TRAIN" = 1 ]; then
        for n in eager eager-packed strix strix-c8; do run_one "$n" train; done
    fi
    say "answer files in $WORK -- now: $0 gates"
}

cmd_ties() {  # --near-ties eager on the test split
    check_env
    mkdir -p "$WORK/logs"
    local n
    for n in strix-ties strix-ties-c8 strix-ties-all; do run_one "$n" test; done
    say "answer files in $WORK -- now: $0 gates"
}

# -- gates -------------------------------------------------------------------------------------------------------
cmd_gates() {
    [ -x "$PY3" ] || die "need python3 for the gate tables"
    [ -f "$GATES" ] || die "no $GATES next to this script"
    say "G1 / G2 from the answer files"
    local use_train=0
    if [ -s "$WORK/eager-train.jsonl" ]; then use_train=1; fi
    "$PY3" "$GATES" --dir "$WORK" --ref eager.jsonl \
        eager-packed.jsonl strix.jsonl strix-c8.jsonl strix-det-c8.jsonl strix-ties.jsonl strix-ties-c8.jsonl \
        --json "$WORK/gates-bf16.json" || true
    if [ -s "$WORK/eager-fp32.jsonl" ]; then
        echo
        "$PY3" "$GATES" --dir "$WORK" --ref eager-fp32.jsonl \
            eager.jsonl eager-packed.jsonl strix.jsonl strix-ties.jsonl --json "$WORK/gates-fp32.json" || true
    fi
    if [ "$use_train" = 1 ]; then
        echo
        "$PY3" "$GATES" --dir "$WORK" --ref eager-train.jsonl \
            eager-packed-train.jsonl strix-train.jsonl strix-c8-train.jsonl --json "$WORK/gates-bf16-train.json" || true
    fi
    if [ -f "$PW" ]; then
        say "pairwise.py (the fork's tool)"
        # pairwise.py opens every file it is given: pass only the ones that exist, reference first
        local f bf16=() fp32=()
        for f in eager.jsonl eager-packed.jsonl strix.jsonl strix-c8.jsonl strix-det-c8.jsonl; do
            if [ -s "$WORK/$f" ]; then bf16+=("$f"); fi
        done
        for f in eager-fp32.jsonl eager.jsonl eager-packed.jsonl strix.jsonl; do
            if [ -s "$WORK/$f" ]; then fp32+=("$f"); fi
        done
        if [ -s "$WORK/eager.jsonl" ] && [ ${#bf16[@]} -ge 2 ]; then
            ( cd "$WORK" && run "$PY3" "$PW" "${bf16[@]}" --out pw-bf16.json ) || echo "   pairwise.py failed (bf16)"
        fi
        if [ -s "$WORK/eager-fp32.jsonl" ] && [ ${#fp32[@]} -ge 2 ]; then
            echo
            ( cd "$WORK" && run "$PY3" "$PW" "${fp32[@]}" --out pw-fp32.json ) || echo "   pairwise.py failed (fp32)"
        fi
    else
        echo
        echo "   no pairwise.py at PW=$PW (the fork's tool); the tables above come from basal-strix-gates.py"
    fi
}

cmd_status() {
    echo "WORK=$WORK"
    [ -d "$WORK" ] || { echo "   (does not exist)"; return 0; }
    local f
    for f in "$WORK"/*.jsonl; do
        [ -e "$f" ] || continue
        printf '   %-28s %6s lines  %s\n' "$(basename "$f")" "$(wc -l < "$f")" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
    done
    ls "$WORK"/logs/*.log 2>/dev/null | sed 's/^/   log /' || true
}

# -- one profiled eager forward: which SDPA kernel does the reference actually run? -------------------------------
cmd_sdpa() {
    check_env
    say "SDPA kernels of the eager reference (one forward, torch.profiler)"
    BASAL_MODEL=$MODEL PYTHONPATH=$REPO run "$PY" -P - <<'PY'
import contextlib, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from basal.engine import eager_sdpa_kernels
from torch.nn.attention import sdpa_kernel
from torch.profiler import ProfilerActivity, profile

m = os.environ["BASAL_MODEL"]
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).cuda().eval()
ids = tok("The storehouse is above its target and the lot is filling. Is the lot filling?", return_tensors="pt").input_ids.cuda()
kernels = eager_sdpa_kernels("cuda")
ctx = sdpa_kernel(kernels) if kernels else contextlib.nullcontext()
with torch.no_grad(), ctx:
    for _ in range(3):
        model(ids)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model(ids)
        torch.cuda.synchronize()
print("kernels allowed by the eager backend:", kernels or "every enabled one")
table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
print(table)
hits = [ln.strip() for ln in table.splitlines() if any(w in ln.lower() for w in ("flash", "efficient", "attn", "attention"))]
print("attention kernels seen:", hits or "none in the top 40 rows (raise row_limit)")
PY
}

# -- main --------------------------------------------------------------------------------------------------------
case "${1:-help}" in
    check)  check_env ;;
    setup)  cmd_setup ;;
    run)    shift; cmd_run "$@" ;;
    all)    cmd_all ;;
    ties)   cmd_ties ;;
    gates)  cmd_gates ;;
    status) cmd_status ;;
    sdpa)   cmd_sdpa ;;
    help|-h|--help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}" 
        ;;
    *) die "unknown command $1 (check, setup, run <name>, all, ties, gates, status, sdpa)" ;;
esac
