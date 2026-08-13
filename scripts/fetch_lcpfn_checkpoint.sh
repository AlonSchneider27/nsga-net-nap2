#!/usr/bin/env bash
# Download the pretrained LC-PFN checkpoint (~95 MiB gz) into the gitignored
# trained_models/lcpfn/ directory. Run from the repo root (works on the BGU
# cluster too — it has direct internet access).
set -euo pipefail

NAME="pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt"
URL="https://ml.informatik.uni-freiburg.de/research-artifacts/lcpfn/${NAME}.gz"
DEST_DIR="trained_models/lcpfn"

mkdir -p "${DEST_DIR}"
if [ -f "${DEST_DIR}/${NAME}" ]; then
    echo "Already present: ${DEST_DIR}/${NAME}"
    exit 0
fi

echo "Downloading ${URL} ..."
curl -L --fail -o "${DEST_DIR}/${NAME}.gz" "${URL}"
echo "Decompressing ..."
gunzip -f "${DEST_DIR}/${NAME}.gz"
echo "Done: ${DEST_DIR}/${NAME}"
