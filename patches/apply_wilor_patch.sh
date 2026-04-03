#!/usr/bin/env bash
set -e

WILOR_DIR="${1:-common/nets/WiLoR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/hand4wholepp_img_feat.patch"

if [ ! -d "$WILOR_DIR/.git" ]; then
  echo "Error: '$WILOR_DIR' is not a git repository."
  echo "Usage: bash patches/apply_wilor_patch.sh [path/to/WiLoR]"
  echo "Default: common/nets/WiLoR"
  exit 1
fi

git -C "$WILOR_DIR" apply "$PATCH_FILE"
echo "Applied WiLoR patch successfully to $WILOR_DIR"