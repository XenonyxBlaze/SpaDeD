#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" High-Performance Dataset Extraction and Queuing Engine
Extracts downloaded Celeb-DF-v2 and DF40 family zip archives from Downloads to proper data directories
using multi-threaded 7-Zip with sequential queueing to prevent disk thrashing.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 7-Zip executable path
SEVEN_ZIP_PATH = r"C:\Program Files\7-Zip\7z.exe"
WINRAR_PATH = r"C:\Program Files\WinRAR\rar.exe"

# Base target directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# DF40 Category Mappings
DF40_CATEGORY_MAP = {
    # Entire Face Synthesis (EFS)
    "stylegan2": "EFS",
    "stylegan3": "EFS",
    "styleganxl": "EFS",
    "ddim": "EFS",
    "dit": "EFS",
    "sit": "EFS",
    "pixart": "EFS",
    "sd2.1": "EFS",
    "vqgan": "EFS",
    "rddm": "EFS",

    # Facial Expression / Audio-driven (FE)
    "sadtalker": "FE",
    "wav2lip": "FE",
    "pirender": "FE",
    "danet": "FE",
    "one_shot_free": "FE",

    # Face Reenactment (FR)
    "fomm": "FR",
    "facevid2vid": "FR",
    "fsgan": "FR",
    "hyperreenact": "FR",
    "lia": "FR",
    "mcnet": "FR",
    "mraa": "FR",
    "tpsm": "FR",

    # Face Swap (FS)
    "simswap": "FS",
    "inswap": "FS",
    "facedancer": "FS",
    "faceswap": "FS",
    "blendface": "FS",
    "e4s": "FS",
    "mobileswap": "FS",
    "uniface": "FS",
}


def get_archive_tool():
    """Finds the fastest installed extraction binary (7-Zip preferred)."""
    if os.path.isfile(SEVEN_ZIP_PATH):
        return "7z", SEVEN_ZIP_PATH
    if os.path.isfile(WINRAR_PATH):
        return "winrar", WINRAR_PATH
    return None, None


def get_target_directory(archive_name: str) -> Path:
    """Determines the target folder in data/ based on the dataset archive name."""
    name_clean = archive_name.lower().replace(".zip", "").strip()

    if "celeb-df" in name_clean:
        return DATA_DIR / "Celeb-DF-v2"

    if name_clean in DF40_CATEGORY_MAP:
        category = DF40_CATEGORY_MAP[name_clean]
        return DATA_DIR / "DF40" / category

    # Default to DF40 root if it's a known manipulation method
    return DATA_DIR / "DF40"


def find_downloaded_archives(downloads_dir: Path):
    """Scans downloads folder for dataset zip files, filtering out 0-byte or active part files."""
    valid_archives = []
    
    for f in downloads_dir.glob("*.zip"):
        # Ignore 0-byte files or corrupted downloads
        if f.stat().st_size < 1024 * 1024:  # Under 1MB
            continue
        
        # Check if there is an active .part or .crdownload for this file
        part_files = list(downloads_dir.glob(f"{f.name}.*part*")) + list(downloads_dir.glob(f"{f.name}.*crdownload*"))
        if part_files:
            print(f"[-] Skipping {f.name} (Active in-progress download detected: {part_files[0].name})")
            continue

        name_lower = f.name.lower()
        if "celeb-df" in name_lower or any(k in name_lower for k in DF40_CATEGORY_MAP):
            valid_archives.append(f)

    # Sort archives by size so larger ones are predictable
    return sorted(valid_archives, key=lambda x: x.name.lower())


def extract_archive_7z(seven_zip_exe: str, archive_path: Path, target_dir: Path):
    """Extracts an archive using multi-threaded 7-Zip."""
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        seven_zip_exe,
        "x",
        "-y",               # Auto-overwrite without prompt
        "-mmt=on",          # Enable multi-threading on Ryzen 9700X
        f"-o{str(target_dir)}",
        str(archive_path)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        return True, elapsed
    else:
        return False, result.stderr or result.stdout


def main():
    print("=" * 80)
    print("  SpaDeD HIGH-PERFORMANCE DATASET EXTRACTION & QUEUING ENGINE")
    print("=" * 80)

    tool_type, tool_path = get_archive_tool()
    if not tool_type:
        print("[!] Error: Neither 7-Zip nor WinRAR was found at standard installation paths.")
        sys.exit(1)

    print(f"[+] Extraction Engine: {tool_type.upper()} ({tool_path})")
    print(f"[+] Hardware Optimization: Multi-Threading Active (-mmt=on)")

    downloads_dir = Path(os.path.expanduser("~")) / "Downloads"
    if not downloads_dir.exists():
        print(f"[!] Error: Downloads directory not found at {downloads_dir}")
        sys.exit(1)

    print(f"[+] Scanning: {downloads_dir}")
    archives = find_downloaded_archives(downloads_dir)

    if not archives:
        print("[!] No completed dataset zip archives found in Downloads.")
        return

    total_size_bytes = sum(f.stat().st_size for f in archives)
    total_size_gb = total_size_bytes / (1024 ** 3)

    print(f"[+] Found {len(archives)} dataset archives ({total_size_gb:.2f} GB total payload)")
    print("=" * 80)
    print("QUEUED EXTRACTIONS:")
    for idx, arc in enumerate(archives, 1):
        target = get_target_directory(arc.name)
        size_mb = arc.stat().st_size / (1024 ** 2)
        print(f"  [{idx:02d}/{len(archives):02d}] {arc.name:<25} ({size_mb:>8.1f} MB) -> {target.relative_to(PROJECT_ROOT)}")
    print("=" * 80)

    start_all = time.time()
    completed = 0
    failed = 0

    for idx, arc in enumerate(archives, 1):
        target = get_target_directory(arc.name)
        size_mb = arc.stat().st_size / (1024 ** 2)
        
        print(f"\n[>] [{idx:02d}/{len(archives):02d}] Extracting {arc.name} ({size_mb:.1f} MB)...")
        print(f"    Target Destination: {target}")

        ok, out = extract_archive_7z(tool_path, arc, target)

        if ok:
            elapsed = out
            speed_mb = size_mb / max(elapsed, 0.01)
            print(f"[+] [SUCCESS] {arc.name} extracted in {elapsed:.1f}s ({speed_mb:.1f} MB/s throughput)")
            completed += 1
        else:
            print(f"[!] [FAILED] Could not extract {arc.name}: {out}")
            failed += 1

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 80)
    print(f"  EXTRACTION QUEUE COMPLETE: {completed}/{len(archives)} archives extracted successfully.")
    print(f"  Total Time: {total_elapsed / 60:.1f} minutes | Average Speed: {total_size_gb * 1024 / max(total_elapsed, 1):.1f} MB/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
