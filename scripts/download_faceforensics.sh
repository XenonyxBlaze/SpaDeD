#!/usr/bin/env bash
# ==============================================================================
# SpaDeD FaceForensics++ Downloader (Bash Wrapper)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${1:-${PROJECT_ROOT}/data/FaceForensics++}"
SERVER="${2:-EU2}"
COMPRESSION="${3:-c23}"
DATASET="${4:-all}"

echo "================================================================================"
echo "  ♠ SpaDeD FACEFORENSICS++ DOWNLOAD RUNNER"
echo "================================================================================"
echo "[+] Target Output Directory: ${OUTPUT_DIR}"
echo "[+] Mirror Region: ${SERVER}"
echo "[+] Compression: ${COMPRESSION}"
echo "[+] Dataset Split: ${DATASET}"
echo "================================================================================"

python "${SCRIPT_DIR}/download_faceforensics.py" \
    "${OUTPUT_DIR}" \
    -d "${DATASET}" \
    -c "${COMPRESSION}" \
    -t videos \
    --server "${SERVER}" \
    --yes
