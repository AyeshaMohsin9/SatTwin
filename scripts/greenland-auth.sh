#!/usr/bin/env bash
#
# greenland-auth.sh — Run this from your LOCAL LAPTOP daily
# --------------------------------------------------------
# Authenticates with Midway/Isengard and sets up the 'greenland' AWS profile
# so you can connect to your Greenland SDB instance.
#
# Usage:
#   ./scripts/greenland-auth.sh
#
# After this succeeds, use the Greenland console or VSCode extension to connect.

set -euo pipefail

# ---- Configuration (from Greenland Console > Reservation tab) ----
ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"
PROFILE="greenland"
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"
# ------------------------------------------------------------------

echo "============================================"
echo " Greenland Daily Auth (Local Laptop)"
echo "============================================"
echo ""

# Step 1: Midway authentication
echo "[1/4] Refreshing Midway credentials..."
mwinit -f
echo "  ✓ Midway OK"
echo ""

# Step 2: Assume customer role via Isengard
echo "[2/4] Assuming role '$CUSTOMER_ROLE' on account $ACCOUNT (provider: $PROVIDER)..."
ada credentials update \
  --account "$ACCOUNT" \
  --role "$CUSTOMER_ROLE" \
  --provider "$PROVIDER" \
  --once
echo "  ✓ Credentials updated in default profile"
echo ""

# Step 3: Configure greenland profile to chain from default -> job role
echo "[3/4] Configuring '$PROFILE' AWS profile..."
aws configure set --profile "$PROFILE" source_profile default
aws configure set --profile "$PROFILE" region "$REGION"
aws configure set --profile "$PROFILE" role_arn "$JOB_ROLE_ARN"
echo "  ✓ Profile '$PROFILE' configured"
echo ""

# Step 4: Verify the full auth chain works
echo "[4/4] Verifying auth chain (default -> greenland job role)..."
echo ""
echo "  Caller identity:"
aws sts get-caller-identity --profile "$PROFILE"
echo ""

echo "============================================"
echo " ✅ Auth complete! You can now connect to"
echo "    your Greenland instance."
echo "============================================"
echo ""
echo "Next steps:"
echo "  - Use the Greenland console 'Open in VS Code' button"
echo "  - Or run: ./scripts/greenland-connect.sh tunnel"
echo ""
