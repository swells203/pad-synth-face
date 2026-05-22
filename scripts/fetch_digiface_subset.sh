#!/usr/bin/env bash
# Fetch a partition of the Microsoft Research DigiFace-1M "5 images per
# identity" subset. The dataset is publicly hosted on Azure blob storage
# (no auth required). One partition covers ~33k identities × 5 images each
# — far more than the 24 identities (8 Set A + 16 Set B) needed by our
# real-bonafide sweep. We fetch only the first partition for speed.
#
# Source (Microsoft Research, WACV 2023):
#   https://github.com/microsoft/DigiFace1M
#
# Idempotent: skips download if the marker file is present.
set -euo pipefail

OUT_DIR="datasets/_real/digiface_118k_raw"
ZIP_NAME="subjects_100000-133332_5_imgs.zip"
URL="https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/${ZIP_NAME}"
MARKER="$OUT_DIR/.fetch_complete"

mkdir -p "$OUT_DIR"

if [[ -f "$MARKER" ]]; then
  echo "DigiFace-1M partition already present at $OUT_DIR (marker exists)."
  exit 0
fi

echo "Fetching $URL"
echo "  -> $OUT_DIR/$ZIP_NAME"
curl -L --fail --progress-bar -o "$OUT_DIR/$ZIP_NAME" "$URL"

echo "Extracting..."
unzip -q "$OUT_DIR/$ZIP_NAME" -d "$OUT_DIR"

# Cleanup the zip after extraction (idempotency relies on the marker, not
# the zip's continued presence).
rm "$OUT_DIR/$ZIP_NAME"

# Verify layout: at least one identity dir with PNGs.
n_identities=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [[ "$n_identities" -lt 100 ]]; then
  echo "ERROR: expected many identity dirs under $OUT_DIR, found $n_identities" >&2
  exit 3
fi

touch "$MARKER"
echo "DigiFace-1M partition extracted to $OUT_DIR ($n_identities identity dirs)."
