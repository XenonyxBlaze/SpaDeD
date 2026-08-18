#!/usr/bin/env bash
# ==============================================================================
# SpaDeD Dataset Extraction Engine (Bash / 7-Zip / Unzip)
# Extracts downloaded archives from ~/Downloads into the structured data folder
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
DOWNLOADS_DIR="${HOME}/Downloads"

# Detect Windows user profile path if in Git Bash / MSYS2
if [ -d "/c/Users/${USER}/Downloads" ]; then
    DOWNLOADS_DIR="/c/Users/${USER}/Downloads"
elif [ -n "${USERPROFILE:-}" ] && [ -d "${USERPROFILE}/Downloads" ]; then
    DOWNLOADS_DIR="${USERPROFILE}/Downloads"
fi

echo "================================================================================"
echo "  ♠ SpaDeD DATASET EXTRACTION ENGINE (BASH / 7-ZIP)"
echo "================================================================================"
echo "[+] Source Downloads: ${DOWNLOADS_DIR}"
echo "[+] Target Data Directory: ${DATA_DIR}"

mkdir -p "${DATA_DIR}/Celeb-DF-v2"
mkdir -p "${DATA_DIR}/DF40/EFS"
mkdir -p "${DATA_DIR}/DF40/FE"
mkdir -p "${DATA_DIR}/DF40/FR"
mkdir -p "${DATA_DIR}/DF40/FS"

# Detect extraction tool
if command -v 7z &> /dev/null; then
    EXTRACT_CMD="7z"
elif [ -f "/c/Program Files/7-Zip/7z.exe" ]; then
    EXTRACT_CMD="/c/Program Files/7-Zip/7z.exe"
elif [ -f "C:\\Program Files\\7-Zip\\7z.exe" ]; then
    EXTRACT_CMD="C:\\Program Files\\7-Zip\\7z.exe"
elif command -v unzip &> /dev/null; then
    EXTRACT_CMD="unzip"
else
    echo "[!] Error: Neither 7z nor unzip found."
    exit 1
fi

echo "[+] Extraction Binary: ${EXTRACT_CMD}"
echo "--------------------------------------------------------------------------------"

extract_file() {
    local zip_file="$1"
    local target_dir="$2"
    
    if [ ! -f "${zip_file}" ]; then
        return 0
    fi

    local filename=$(basename "${zip_file}")
    echo -e "\n[>] Extracting ${filename} -> ${target_dir}..."
    local start_time=$(date +%s)

    mkdir -p "${target_dir}"

    if [[ "${EXTRACT_CMD}" == *"7z"* ]]; then
        "${EXTRACT_CMD}" x -y -mmt=on "-o${target_dir}" "${zip_file}" > /dev/null
    else
        unzip -q -o "${zip_file}" -d "${target_dir}"
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo "[+] [SUCCESS] Extracted ${filename} in ${elapsed}s."
}

# 1. Celeb-DF-v2
extract_file "${DOWNLOADS_DIR}/Celeb-DF-v2.zip" "${DATA_DIR}/Celeb-DF-v2"

# 2. DF40 EFS (Entire Face Synthesis)
for f in ddim DiT pixart RDDM sd2.1 SiT StyleGAN2 StyleGAN3 StyleGANXL VQGAN; do
    extract_file "${DOWNLOADS_DIR}/${f}.zip" "${DATA_DIR}/DF40/EFS"
done

# 3. DF40 FE (Facial Expression / Audio-driven)
for f in danet one_shot_free pirender sadtalker wav2lip; do
    extract_file "${DOWNLOADS_DIR}/${f}.zip" "${DATA_DIR}/DF40/FE"
done

# 4. DF40 FR (Face Reenactment)
for f in facevid2vid fomm fsgan hyperreenact lia mcnet MRAA tpsm; do
    extract_file "${DOWNLOADS_DIR}/${f}.zip" "${DATA_DIR}/DF40/FR"
done

# 5. DF40 FS (Face Swap)
for f in blendface e4s facedancer faceswap inswap mobileswap simswap uniface; do
    extract_file "${DOWNLOADS_DIR}/${f}.zip" "${DATA_DIR}/DF40/FS"
done

echo -e "\n================================================================================"
echo "  EXTRACTION QUEUE COMPLETE!"
echo "================================================================================"
