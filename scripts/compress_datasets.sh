#!/usr/bin/env bash
# ==============================================================================
# SpaDeD Dataset Compression Engine (Bash / 7-Zip / Zip)
# Compresses extracted datasets into optimized archives for Google Drive upload
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${PROJECT_ROOT}/archives_for_upload"

echo "================================================================================"
echo "  ♠ SpaDeD DATASET COMPRESSION ENGINE (GOOGLE DRIVE UPLOAD PREPARATION)"
echo "================================================================================"
echo "[+] Project Root: ${PROJECT_ROOT}"
echo "[+] Data Directory: ${DATA_DIR}"

mkdir -p "${OUTPUT_DIR}"
echo "[+] Target Upload Directory: ${OUTPUT_DIR}"

# Detect fastest compression tool (7z or zip)
if command -v 7z &> /dev/null; then
    COMPRESS_CMD="7z"
elif [ -f "/c/Program Files/7-Zip/7z.exe" ]; then
    COMPRESS_CMD="/c/Program Files/7-Zip/7z.exe"
elif [ -f "C:\\Program Files\\7-Zip\\7z.exe" ]; then
    COMPRESS_CMD="C:\\Program Files\\7-Zip\\7z.exe"
elif command -v zip &> /dev/null; then
    COMPRESS_CMD="zip"
else
    echo "[!] Error: Neither 7z nor zip utility found."
    exit 1
fi

echo "[+] Compression Utility: ${COMPRESS_CMD}"
echo "--------------------------------------------------------------------------------"

compress_folder() {
    local folder_name="$1"
    local archive_name="$2"
    local source_path="${DATA_DIR}/${folder_name}"
    local target_archive="${OUTPUT_DIR}/${archive_name}"

    if [ ! -d "${source_path}" ]; then
        echo "[-] Skipping ${folder_name} (Directory not found at ${source_path})"
        return 0
    fi

    echo -e "\n[>] Compressing ${folder_name} -> ${target_archive}..."
    local start_time=$(date +%s)

    if [[ "${COMPRESS_CMD}" == *"7z"* ]]; then
        "${COMPRESS_CMD}" a -tzip -mx=4 -mmt=on -y "${target_archive}" "${source_path}"
    else
        (cd "${DATA_DIR}" && zip -r -q -4 "${target_archive}" "${folder_name}")
    fi

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    if [ -f "${target_archive}" ]; then
        local size_mb=$(du -m "${target_archive}" | cut -f1)
        echo "[+] [SUCCESS] ${archive_name} created (${size_mb} MB) in ${elapsed}s."
    else
        echo "[!] [ERROR] Failed to create ${archive_name}."
    fi
}

# 1. Compress Celeb-DF-v2
compress_folder "Celeb-DF-v2" "Celeb-DF-v2.zip"

# 2. Compress DF40 Paradigms (can compress together or individually)
compress_folder "DF40" "DF40_all.zip"

# 3. Compress FaceForensics++ (if downloaded)
compress_folder "FaceForensics++" "FaceForensics_c23.zip"

echo -e "\n================================================================================"
echo "  ALL DATASETS COMPRESSED & READY FOR GOOGLE DRIVE UPLOAD!"
echo "  Upload files in '${OUTPUT_DIR}/' to: 'My Drive/SpaDeD_Datasets/'"
echo "================================================================================"
