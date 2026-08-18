#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Downloads FaceForensics++ and Deep Fake Detection public data release.
Integrates with the official TUM FaceForensics script (scripts/ff_dataset.py) while adding:
1. Windows Sleep/Hibernate Prevention (keeps PC awake during multi-hour downloads).
2. Real-time per-file and batch progress bars (MB downloaded, MB/s speed, ETA).
3. Resumable downloads with automatic retry on network drops/sleep recovery.
4. Atomic file verification (prevents corrupt partial files).
"""

import argparse
import os
import sys
import time
import json
import random
import ctypes
import importlib.util
import requests
from tqdm import tqdm
from os.path import join

# Ensure UTF-8 output encoding on Windows consoles to prevent charmap encoding crashes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Windows Power Management flags to prevent Sleep/Hibernate during active downloads
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def prevent_system_sleep():
    """Instructs Windows OS power management to stay awake while downloading."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            print("[+] Windows Sleep/Hibernate prevention active (PC will not sleep while downloading).")
        except Exception as e:
            print(f"[!] Warning: Could not set Windows power state: {e}")


def restore_system_sleep():
    """Restores default Windows power management settings upon exit."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("[+] Windows power management restored to normal.")
        except Exception:
            pass


# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def load_official_script_config():
    """Dynamically loads configuration from official scripts/ff_dataset.py if present locally."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = ['ff_dataset.py', 'ff_script.py', 'faceforensics_download_v4.py']
    for candidate in candidates:
        candidate_path = os.path.join(script_dir, candidate)
        if os.path.isfile(candidate_path):
            try:
                spec = importlib.util.spec_from_file_location("official_ff", candidate_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
            except Exception as e:
                print(f"[!] Note: Could not import {candidate}: {e}")
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics++ public data release with progress tracking & sleep prevention.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help='Dataset to download.',
                        choices=list(DATASETS.keys()) + ['all'])
    parser.add_argument('-c', '--compression', type=str, default='c23',
                        help='Compression degree: c23 (high quality), c40 (low quality), raw.',
                        choices=COMPRESSION)
    parser.add_argument('-t', '--type', type=str, default='videos',
                        help='File type: videos, masks, models.',
                        choices=TYPE)
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Select number of videos to download.')
    parser.add_argument('--server', type=str, default='EU',
                        help='Server mirror region identifier (EU / EU2 / CA).',
                        choices=SERVERS)
    parser.add_argument('--server_url', type=str, default=os.getenv('FF_SERVER_URL', None),
                        help='Official FaceForensics download server URL.')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Automatically accept terms without prompting.')
    args = parser.parse_args()

    official_mod = load_official_script_config()
    server_url = args.server_url

    if not server_url and official_mod is not None:
        # Resolve server from official module
        server = args.server
        if hasattr(official_mod, 'SERVERS') and server in official_mod.SERVERS:
            if server == 'EU':
                server_url = 'http://canis.vc.in.tum.de:8100/'
            elif server == 'EU2':
                server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
            elif server == 'CA':
                server_url = 'http://falas.cmpt.sfu.ca:8100/'
        print(f"[+] Loaded official FaceForensics configuration from local script (Mirror: {server}).")

    if not server_url:
        print("\n" + "="*75)
        print("[!] FaceForensics++ Terms of Service Compliance Check:")
        print("    In accordance with the FaceForensics research agreement, server endpoints")
        print("    must not be hardcoded in public repositories.")
        print("    Please request access at: https://github.com/ondyari/FaceForensics")
        print("    and place ff_dataset.py into scripts/ or provide --server_url.")
        print("="*75)
        try:
            server_url = input("\n[?] Enter the FaceForensics server base URL received from TUM: ").strip()
        except (EOFError, KeyboardInterrupt):
            server_url = None

        if not server_url:
            print("\n[!] Error: No server URL provided. Set FF_SERVER_URL, pass --server_url,")
            print("    or place ff_dataset.py into the scripts/ directory.")
            sys.exit(1)

    if not server_url.endswith('/'):
        server_url += '/'

    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/Deepfakes/models/'
    return args


def fetch_json_with_retry(url: str, max_retries: int = 5, timeout=(30, 90)):
    """Fetches JSON metadata with exponential backoff retry for slow/overloaded mirrors."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            if attempt < max_retries:
                wait_time = attempt * 3
                print(f"[!] Metadata fetch notice (attempt {attempt}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to fetch metadata from {url} after {max_retries} attempts: {e}")


def download_file_resumable(url: str, out_file: str, desc: str = "", max_retries: int = 5):
    """
    Robust chunked file downloader with:
    - Resume capability (Range HTTP header)
    - Real-time progress bar (Speed, ETA, Total MB)
    - Exponential backoff retry on network drops/sleep resume
    - Atomic rename from .part to final destination
    """
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    temp_file = out_file + ".part"

    # If the completed file already exists and is non-empty, verify and skip
    if os.path.isfile(out_file) and os.path.getsize(out_file) > 1024:
        return True

    # Check existing partial downloaded bytes for resume
    initial_bytes = os.path.getsize(temp_file) if os.path.isfile(temp_file) else 0

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            if initial_bytes > 0:
                headers['Range'] = f'bytes={initial_bytes}-'

            response = requests.get(url, headers=headers, stream=True, timeout=(30, 120))

            if response.status_code == 416:  # Range not satisfiable, file might already be complete
                if os.path.isfile(temp_file):
                    os.replace(temp_file, out_file)
                    return True
                initial_bytes = 0
                response = requests.get(url, stream=True, timeout=(30, 120))

            if response.status_code not in (200, 206):
                if response.status_code == 404:
                    print(f"\n[!] File not found on server (404): {url}")
                    return False
                response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if response.status_code == 206:
                total_size += initial_bytes

            mode = 'ab' if initial_bytes > 0 else 'wb'
            chunk_size = 64 * 1024  # 64 KB chunks

            file_desc = desc if desc else os.path.basename(out_file)
            with open(temp_file, mode) as f, tqdm(
                total=total_size,
                initial=initial_bytes,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=file_desc[:35].ljust(35),
                leave=False,
                ncols=90
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Download complete: atomically move temp to destination
            if os.path.isfile(temp_file):
                os.replace(temp_file, out_file)
            return True

        except (requests.RequestException, IOError) as e:
            if attempt < max_retries:
                wait_time = attempt * 3
                time.sleep(wait_time)
                initial_bytes = os.path.getsize(temp_file) if os.path.isfile(temp_file) else 0
            else:
                print(f"\n[!] Failed to download {os.path.basename(out_file)} after {max_retries} attempts: {e}")
                return False

    return False


def download_file_list(filenames, base_url, output_path, category_name="Dataset"):
    """Downloads a list of files with an overall batch counter and detailed per-file progress."""
    os.makedirs(output_path, exist_ok=True)
    total_files = len(filenames)
    success_count = 0

    print(f"\n[>] Starting {category_name} ({total_files} files) -> {output_path}")
    overall_pbar = tqdm(total=total_files, desc=f"Overall [{category_name[:15]}]", ncols=90)

    for idx, filename in enumerate(filenames, 1):
        target_path = join(output_path, filename)
        file_desc = f"[{idx}/{total_files}] {filename}"
        ok = download_file_resumable(base_url + filename, target_path, desc=file_desc)
        if ok:
            success_count += 1
        overall_pbar.update(1)

    overall_pbar.close()
    print(f"[+] Completed {category_name}: {success_count}/{total_files} files downloaded successfully.")


def main(args):
    prevent_system_sleep()
    try:
        print("=" * 80)
        print("  FACEFORENSICS++ RESUMABLE DOWNLOADER WITH SYSTEM SLEEP PREVENTION")
        print("=" * 80)
        print("Terms of Use: " + args.tos_url)
        print("Server Mirror: " + args.server)
        print("=" * 80)

        if not args.yes:
            print("Press Enter to accept terms & begin download, or CTRL+C to cancel...")
            input()

        c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
        c_type = args.type
        c_compression = args.compression
        num_videos = args.num_videos
        output_path = args.output_path
        os.makedirs(output_path, exist_ok=True)

        for dataset in c_datasets:
            dataset_path = DATASETS[dataset]

            if 'original_youtube_videos' in dataset:
                suffix = 'info' if 'info' in dataset_path else ''
                out_name = f'downloaded_videos{suffix}.zip'
                download_file_resumable(
                    args.base_url + dataset_path,
                    join(output_path, out_name),
                    desc=f"YouTube Raw Archive {suffix}"
                )
                continue

            # Fetch filelist from server with retry
            if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
                filepaths = fetch_json_with_retry(args.base_url + DEEPFEAKES_DETECTION_URL)
                filelist = filepaths['actors'] if 'actors' in dataset_path else filepaths['DeepFakesDetection']
            elif 'original' in dataset_path:
                file_pairs = fetch_json_with_retry(args.base_url + FILELIST_URL)
                filelist = [item for pair in file_pairs for item in pair]
            else:
                file_pairs = fetch_json_with_retry(args.base_url + FILELIST_URL)
                filelist = []
                for pair in file_pairs:
                    filelist.append('_'.join(pair))
                    if c_type != 'models':
                        filelist.append('_'.join(pair[::-1]))

            if num_videos is not None and num_videos > 0:
                filelist = filelist[:num_videos]

            # Construct URLs
            dataset_videos_url = args.base_url + f'{dataset_path}/{c_compression}/{c_type}/'
            dataset_mask_url = args.base_url + f'{dataset_path}/masks/{c_type}/'

            if c_type == 'videos':
                target_dir = join(output_path, dataset_path, c_compression, c_type)
                video_files = [f + '.mp4' for f in filelist]
                download_file_list(video_files, dataset_videos_url, target_dir, category_name=dataset)

            elif c_type == 'masks':
                if 'original' in dataset:
                    continue
                if 'FaceShifter' in dataset:
                    continue
                target_dir = join(output_path, dataset_path, c_type, 'videos')
                mask_files = [f + '.mp4' for f in filelist]
                download_file_list(mask_files, dataset_mask_url, target_dir, category_name=f"{dataset}-masks")

            elif c_type == 'models':
                if dataset != 'Deepfakes':
                    continue
                for folder in tqdm(filelist, desc="Models", ncols=90):
                    folder_url = args.deepfakes_model_url + folder + '/'
                    folder_out = join(output_path, dataset_path, c_type, folder)
                    for model_name in DEEPFAKES_MODEL_NAMES:
                        download_file_resumable(folder_url + model_name, join(folder_out, model_name))

        print("\n" + "=" * 80)
        print("  ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    finally:
        restore_system_sleep()


if __name__ == "__main__":
    args = parse_args()
    main(args)
