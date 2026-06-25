#!/usr/bin/env bash
# Sweep a trained GR00T doorman policy across every door in a door set, running a batch
# of parallel envs per door, and collect per-door success into a summary table.
#
# Run INSIDE the Arena container (needs Isaac Sim). A GR00T policy server must already be
# serving the checkpoint (see submodules/Isaac-GR00T: `uv run python gr00t/eval/run_gr00t_server.py`).
#
# Example:
#   isaaclab_arena_gr00t/eval/run_doorman_sweep.sh \
#     -H 192.168.1.50 \
#     -d /workspaces/isaaclab_arena/isaaclab_arena/assets/doorman_doors_combined/usd \
#     -e 10 -n 10 -t 14
#
# Disturbance-recovery sweep (random wrist poke after grasp commit) via -K passthrough:
#   isaaclab_arena_gr00t/eval/run_doorman_sweep.sh -H 127.0.0.1 \
#     -K "--poke --poke_random --poke_start_step 30 --poke_duration 20 --poke_force_range 10 30"
#
# Doors with index < NUM_TRAINED are labelled "trained", the rest "held-out".
set -uo pipefail

# --- defaults --------------------------------------------------------------
REMOTE_HOST="localhost"
REMOTE_PORT=5555
DOORS_DIR="/workspaces/isaaclab_arena/isaaclab_arena/assets/doorman_doors_combined/usd"
NUM_ENVS=10
NUM_EPISODES=10
NUM_TRAINED=14          # index boundary: < this == trained-on, >= this == held-out
OUT_DIR=""
POLICY_YAML="isaaclab_arena_gr00t/policy/config/alex_manip_gr00t_closedloop_config.yaml"
EMBODIMENT="alex_v2_ability_hands"
POKE_ARGS=""            # extra policy_runner args (e.g. --poke ... for disturbance-recovery runs)

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while getopts "H:p:d:e:n:t:o:K:h" opt; do
  case "$opt" in
    H) REMOTE_HOST="$OPTARG" ;;
    p) REMOTE_PORT="$OPTARG" ;;
    d) DOORS_DIR="$OPTARG" ;;
    e) NUM_ENVS="$OPTARG" ;;
    n) NUM_EPISODES="$OPTARG" ;;
    t) NUM_TRAINED="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    K) POKE_ARGS="$OPTARG" ;;
    *) usage ;;
  esac
done

[ -d "$DOORS_DIR" ] || { echo "ERROR: doors dir not found: $DOORS_DIR" >&2; exit 1; }
NUM_DOORS=$(find "$DOORS_DIR" -maxdepth 1 -name 'door_*.usd' | wc -l)
[ "$NUM_DOORS" -gt 0 ] || { echo "ERROR: no door_*.usd in $DOORS_DIR" >&2; exit 1; }

[ -n "$OUT_DIR" ] || OUT_DIR="$PWD/doorman_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf "door\tlabel\thandle\tsuccess_rate\tjoint_moved_rate\tik_retries\n" > "$SUMMARY"

echo "Doors dir : $DOORS_DIR ($NUM_DOORS doors)"
echo "Server    : $REMOTE_HOST:$REMOTE_PORT"
echo "Per door  : $NUM_ENVS envs x ($NUM_EPISODES episodes total)"
echo "Poke args : ${POKE_ARGS:-<none>}"
echo "Out dir   : $OUT_DIR"
echo

# Optional: read handle type per door from metadata.json (lever / pushbar / ...).
META="$DOORS_DIR/metadata.json"
get_handle() {  # $1 = zero-padded door key
  [ -f "$META" ] || { echo "?"; return; }
  grep -A20 "\"$1\"" "$META" | grep -m1 "doorHandleType" \
    | sed -E 's/.*"doorHandleType": *"([^"]+)".*/\1/'
}

# Extract a numeric metric value from the final "Metrics:" line of a run log.
get_metric() {  # $1 = log file, $2 = metric key
  grep -h "Metrics:" "$1" 2>/dev/null | tail -1 \
    | grep -oE "'$2': *[0-9.]+" | grep -oE "[0-9.]+$" | tail -1
}

for d in $(seq 0 $((NUM_DOORS - 1))); do
  key=$(printf "door_%04d" "$d")
  if [ "$d" -lt "$NUM_TRAINED" ]; then label="trained"; else label="held-out"; fi
  handle=$(get_handle "$key")
  log="$OUT_DIR/${key}.log"
  echo "===== door $d ($label, $handle) -> $log ====="

  ARENA_DOORMAN_DOORS_DIR="$DOORS_DIR" PYTHONUNBUFFERED=1 \
  /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
    --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy\
    --policy_config_yaml_path "$POLICY_YAML" \
    --remote_host "$REMOTE_HOST" --remote_port "$REMOTE_PORT" \
    --headless --enable_cameras --device cuda \
    --num_envs "$NUM_ENVS" --num_episodes "$NUM_EPISODES" \
    $POKE_ARGS \
    alex_doorman_teleop --embodiment "$EMBODIMENT" --door_index "$d" --fail_on_ik_error \
    > "$log" 2>&1

  sr=$(get_metric "$log" success_rate); sr=${sr:-NA}
  jm=$(get_metric "$log" revolute_joint_moved_rate); jm=${jm:-NA}
  rt=$(grep -h "IK-error retries" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+$"); rt=${rt:-0}
  printf "%d\t%s\t%s\t%s\t%s\t%s\n" "$d" "$label" "$handle" "$sr" "$jm" "$rt" >> "$SUMMARY"
  echo "   success_rate=$sr  joint_moved_rate=$jm  ik_retries=$rt"
done

echo
echo "================= SUMMARY ================="
column -t -s $'\t' "$SUMMARY"
echo
# Per-label mean success (ignoring NA runs).
awk -F'\t' 'NR>1 && $4!="NA" {s[$2]+=$4; n[$2]++}
            END {for (k in s) printf "mean success_rate (%-8s): %.3f  (%d doors)\n", k, s[k]/n[k], n[k]}' "$SUMMARY"
echo "Logs + summary.tsv in: $OUT_DIR"
