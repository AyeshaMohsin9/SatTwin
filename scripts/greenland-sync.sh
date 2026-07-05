#!/usr/bin/env bash
#
# greenland-sync.sh — Push code to / pull results from Greenland instance
# -----------------------------------------------------------------------
# Requires an active SSM port-forward tunnel (greenland-connect.sh tunnel).
#
# Usage:
#   ./scripts/greenland-sync.sh push              # push local code to Greenland
#   ./scripts/greenland-sync.sh pull              # pull results/outputs back
#   ./scripts/greenland-sync.sh run "command"     # run a command on Greenland
#   ./scripts/greenland-sync.sh deploy            # push + run setup
#

set -euo pipefail

# ---- Connection config (must match greenland-connect.sh) ----
LOCAL_PORT="1049"
SSH_USER="greenland-user"
SSH_HOST="localhost"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# ---- Paths ----
LOCAL_PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_PROJECT_DIR="~/SatTwin"

# ---- What to exclude from push (saves bandwidth) ----
EXCLUDE=(
    ".git"
    "__pycache__"
    "*.pyc"
    ".venv"
    ".venv-gl"
    "node_modules"
    "build"
    ".eggs"
    "*.egg-info"
    "wandb"
    ".DS_Store"
    "results"
)

# Build rsync exclude flags
EXCLUDE_FLAGS=""
for pattern in "${EXCLUDE[@]}"; do
    EXCLUDE_FLAGS="$EXCLUDE_FLAGS --exclude=$pattern"
done

SSH_CMD="ssh -p $LOCAL_PORT $SSH_OPTS"

check_tunnel() {
    if ! ssh -p "$LOCAL_PORT" $SSH_OPTS -o ConnectTimeout=3 "$SSH_USER@$SSH_HOST" "echo ok" > /dev/null 2>&1; then
        echo "✗ Cannot reach Greenland. Is the SSM tunnel running?"
        echo "  Start it with: ./scripts/greenland-connect.sh tunnel"
        exit 1
    fi
}

push() {
    echo ">> Pushing local code to Greenland..."
    check_tunnel
    rsync -avz --delete \
        $EXCLUDE_FLAGS \
        -e "$SSH_CMD" \
        "$LOCAL_PROJECT_DIR/" \
        "$SSH_USER@$SSH_HOST:$REMOTE_PROJECT_DIR/"
    echo ">> ✓ Code pushed to $REMOTE_PROJECT_DIR on Greenland"
}

pull() {
    local remote_path="${1:-code/HDTN-SCN/results/}"
    local local_path="${2:-$LOCAL_PROJECT_DIR/code/HDTN-SCN/results/}"
    echo ">> Pulling results from Greenland ($remote_path)..."
    check_tunnel
    mkdir -p "$local_path"
    rsync -avz \
        -e "$SSH_CMD" \
        "$SSH_USER@$SSH_HOST:$REMOTE_PROJECT_DIR/$remote_path" \
        "$local_path"
    echo ">> ✓ Results pulled to $local_path"
}

run_remote() {
    local cmd="$1"
    echo ">> Running on Greenland: $cmd"
    check_tunnel
    ssh -p "$LOCAL_PORT" $SSH_OPTS "$SSH_USER@$SSH_HOST" \
        "cd $REMOTE_PROJECT_DIR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate sattwin && $cmd"
}

# Like run_remote but WITHOUT the conda prefix — for bootstrap/setup commands
# that run before miniconda exists (e.g. first-time instance setup).
run_remote_raw() {
    local cmd="$1"
    echo ">> Running on Greenland (raw): $cmd"
    check_tunnel
    ssh -p "$LOCAL_PORT" $SSH_OPTS "$SSH_USER@$SSH_HOST" \
        "cd $REMOTE_PROJECT_DIR && $cmd"
}

deploy() {
    push
    echo ""
    echo ">> Running setup on Greenland (main node)..."
    run_remote_raw "bash scripts/greenland-instance-setup.sh quick"
}

case "${1:-help}" in
    push)    push ;;
    pull)    pull "${2:-code/HDTN-SCN/results/}" "${3:-$LOCAL_PROJECT_DIR/code/HDTN-SCN/results/}" ;;
    run)     run_remote "${2:?Usage: $0 run \"command\"}" ;;
    run-raw) run_remote_raw "${2:?Usage: $0 run-raw \"command\"}" ;;
    deploy)  deploy ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  push              Sync local code → Greenland main node"
        echo "  pull [path]       Pull results from Greenland → local (default: code/HDTN-SCN/results/)"
        echo "  run \"command\"     Execute a command on Greenland main node (conda env active)"
        echo "  run-raw \"command\" Execute a command on main node WITHOUT conda prefix (bootstrap)"
        echo "  deploy            Push code + run setup on main node"
        echo ""
        echo "Prerequisites:"
        echo "  1. SSM tunnel must be active: ./scripts/greenland-connect.sh tunnel"
        echo "  2. Instance must be set up:   ./scripts/greenland-sync.sh deploy"
        ;;
esac
