#!/usr/bin/env bash
#
# Greenland SDB connect helper
# -----------------------------
# Authenticates with Midway/Isengard, sets up the `greenland` AWS profile,
# and opens the SSM port-forward tunnel to your Scientist Development Box.
#
# Usage:
#   ./scripts/greenland-connect.sh          # auth + open SSM tunnel (default)
#   ./scripts/greenland-connect.sh auth     # auth + create profile only
#   ./scripts/greenland-connect.sh tunnel   # open SSM tunnel only
#   ./scripts/greenland-connect.sh ssh      # SSH into the box (run after tunnel)
#
# NOTE: credentials from `ada` are temporary — re-run `auth` before each session.

set -euo pipefail

# ---- Job-specific config (from Greenland Console > Job properties) ----
ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"                 # conduit is denied for this alias; isengard works
PROFILE="greenland"
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"
REMOTE_PORT="2222"
LOCAL_PORT="1049"
SSH_USER="greenland-user"
# Job: cmohsinm-workspace (EKS: cmohsinm-workspace-79e371d2)  -- SatTwin instance
# Instances: 1x p4d.24xlarge (single main-node), 8x A100 = 8 GPUs, us-east-2
# Initiative: KiroScienceInterns
# NOTE: keep this on port 1049 (unique from other experiments' instances).
#
# Single-node job: only the MAIN node exists and runs the SSM agent, so this is
# the single tunnel target. No worker nodes for this reservation.
SSM_TARGET="mi-0eeb767a3ef638218"   # main-node SsmManagedInstanceId (from job JSON)
# Node internal IP (NodesEniHostIP): main=10.3.39.252
MAIN_NODE_IP="10.3.39.252"
MAIN_NODE_INSTANCE_ID="i-0f18e7782c28a6aa8"
# ----------------------------------------------------------------------

auth() {
  echo ">> Authenticating with Midway + Isengard and configuring '$PROFILE' profile..."
  mwinit -f
  ada credentials update --account="$ACCOUNT" --provider="$PROVIDER" --role="$CUSTOMER_ROLE" --once
  aws configure set --profile "$PROFILE" source_profile default
  aws configure set --profile "$PROFILE" region "$REGION"
  aws configure set --profile "$PROFILE" role_arn "$JOB_ROLE_ARN"
  aws sts get-caller-identity --profile "$PROFILE"
  echo ">> Auth OK. '$PROFILE' profile is ready."
}

tunnel() {
  echo ">> Opening SSM port-forward $LOCAL_PORT -> $REMOTE_PORT (keep this terminal open)..."
  aws ssm start-session \
    --target "$SSM_TARGET" \
    --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"$REMOTE_PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" \
    --profile "$PROFILE" \
    --region "$REGION"
}

ssh_in() {
  echo ">> Connecting via SSH on localhost:$LOCAL_PORT (tunnel must be running)..."
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=60 \
      -p "$LOCAL_PORT" "$SSH_USER@localhost"
}

case "${1:-all}" in
  auth)   auth ;;
  tunnel) tunnel ;;
  ssh)    ssh_in ;;
  all)    auth; tunnel ;;
  *) echo "Usage: $0 [auth|tunnel|ssh|all]"; exit 1 ;;
esac
