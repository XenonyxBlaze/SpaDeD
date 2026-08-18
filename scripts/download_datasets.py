"""
Dataset Download & Directory Setup Helper for Spatiotemporal Deepfake Detection.

This script creates the standardized folder hierarchy for:
1. FaceForensics++ (FF++)
2. DF40 Benchmark (FS, FR, EFS, FE)
3. Celeb-DF v2

It also provides an option to generate sample synthetic dataset clips for testing.
"""

import os
import argparse
import random


def create_directory_hierarchy(base_dir: str = "data"):
    """Creates the standard directory layout for all required datasets."""
    dirs = [
        # FaceForensics++
        os.path.join(base_dir, "FaceForensics++", "original_sequences", "youtube", "c23", "videos"),
        os.path.join(base_dir, "FaceForensics++", "manipulated_sequences", "Deepfakes", "c23", "videos"),
        os.path.join(base_dir, "FaceForensics++", "manipulated_sequences", "Face2Face", "c23", "videos"),
        os.path.join(base_dir, "FaceForensics++", "manipulated_sequences", "FaceSwap", "c23", "videos"),
        os.path.join(base_dir, "FaceForensics++", "manipulated_sequences", "NeuralTextures", "c23", "videos"),
        
        # DF40 Benchmark
        os.path.join(base_dir, "DF40", "FS"),   # Face Swapping (10 methods)
        os.path.join(base_dir, "DF40", "FR"),   # Face Reenactment (13 methods)
        os.path.join(base_dir, "DF40", "EFS"),  # Entire Face Synthesis (12 methods)
        os.path.join(base_dir, "DF40", "FE"),   # Face Editing (5 methods)
        
        # Celeb-DF v2
        os.path.join(base_dir, "Celeb-DF-v2", "Celeb-real"),
        os.path.join(base_dir, "Celeb-DF-v2", "Celeb-synthesis"),
        os.path.join(base_dir, "Celeb-DF-v2", "YouTube-real"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[+] Successfully created standard dataset directory hierarchy under '{base_dir}/'.")


def generate_dummy_sample_data(base_dir: str = "data/sample_dataset", num_identities: int = 10, num_frames: int = 20):
    """
    Generates dummy synthetic image sequences for unit-testing the 5-fold training pipeline.
    """
    print(f"[+] Generating synthetic test samples in '{base_dir}'...")
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        from PIL import Image
        import numpy as np
        
        for ident_idx in range(num_identities):
            ident_id = f"id_{ident_idx:03d}"
            for label, class_name in [(0, "real"), (1, "fake")]:
                sample_dir = os.path.join(base_dir, f"{ident_id}_{class_name}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Generate T synthetic frames
                base_color = np.random.randint(50, 200, size=3, dtype=np.uint8)
                for frame_idx in range(num_frames):
                    frame_noise = np.random.randint(-15, 16, size=(256, 256, 3), dtype=np.int16)
                    frame_img = np.clip(base_color + frame_noise, 0, 255).astype(np.uint8)
                    
                    # Add synthetic artifact for fake class
                    if label == 1:
                        frame_img[100:150, 100:150, :] = (frame_img[100:150, 100:150, :] * 0.8).astype(np.uint8)
                        
                    img_pil = Image.fromarray(frame_img)
                    img_pil.save(os.path.join(sample_dir, f"frame_{frame_idx:03d}.png"))
                    
        print(f"[+] Generated {num_identities * 2} synthetic sequences across {num_identities} unique identities.")
    except ImportError:
        print("[!] PIL/Numpy not installed yet. Skipping sample frame image creation. Directories are ready.")


def print_dataset_instructions():
    """Prints direct download instructions, dataset sizes, and transfer estimates."""
    print("=" * 85)
    print("  OFFICIAL DATASET ACCESS, STORAGE REQUIREMENTS & DOWNLOAD GUIDE")
    print("=" * 85)
    
    print("""
STORAGE & DOWNLOAD TIME ESTIMATES:
-------------------------------------------------------------------------------------
Dataset / Split          | Size      | 50 Mbps   | 100 Mbps  | 300 Mbps  | 1 Gbps
-------------------------------------------------------------------------------------
FaceForensics++ (HQ, c23)| ~38-40 GB | ~1h 45m   | ~55 min   | ~18 min   | ~5-6 min
FaceForensics++ (LQ, c40)| ~10-12 GB | ~30 min   | ~15 min   | ~5 min    | ~1.5 min
FaceForensics++ (Raw)    | ~500-1000G| ~24-48 hrs| ~12-24 hrs| ~4-8 hrs  | ~1.5-2.5 hrs
Celeb-DF (v2)            | ~36-40 GB | ~1h 45m   | ~55 min   | ~18 min   | ~5-6 min
DF40 Benchmark (All 40)  | ~60-80 GB | ~3 hrs    | ~1h 30m   | ~30 min   | ~10 min
Single DF40 Family (FS)  | ~15-20 GB | ~45 min   | ~25 min   | ~8 min    | ~2.5 min
-------------------------------------------------------------------------------------

1. DF40 BENCHMARK (Yan et al., NeurIPS 2024)
   - Scope: 40 manipulation methods across FS, FR, EFS, and FE (~60 - 80 GB total).
   - Official GitHub: https://github.com/YZY-stack/DF40
   - DeepfakeBench Portal: https://github.com/SCLBD/DeepfakeBench
   - Download Instructions:
     1. Clone DeepfakeBench or DF40 repository.
     2. Follow dataset download instructions provided in the DF40 README to obtain the
        manipulated and pristine test splits.
     3. Place extracted categories in:
        - data/DF40/FS/   (SimSwap, InSwapper, FaceShifter, etc., ~20 GB)
        - data/DF40/FR/   (Wav2Lip, SadTalker, Face2Face, etc., ~25 GB)
        - data/DF40/EFS/  (StyleGAN2, Stable Diffusion, DiT, etc., ~15 GB)
        - data/DF40/FE/   (StarGAN, StyleCLIP, e4e, etc., ~10 GB)

2. FACEFORENSICS++ (FF++) (Rössler et al., ICCV 2019)
   - Scope: 1,000 YouTube original sequences + 4 manipulation methods (~38 GB in c23).
   - Official Downloader Included: scripts/download_faceforensics.py
   - Download Command (Standard c23 High-Quality):
        python scripts/download_faceforensics.py data/FaceForensics++ -d all -c c23 -t videos
   - Fast Test Sample (10 videos):
        python scripts/download_faceforensics.py data/FaceForensics++ -d all -c c23 -t videos -n 10
   - Directory Layout:
        data/FaceForensics++/manipulated_sequences/[Method]/c23/videos/
        data/FaceForensics++/original_sequences/youtube/c23/videos/

3. CELEB-DF (v2) (Li et al., CVPR 2020)
   - Scope: 590 celebrity identities, 5,639 high-quality swapped videos (~38 GB).
   - Official GitHub: https://github.com/yuezunli/celeb-deepfakeforensics
   - Access Procedure:
     1. Request access via the official form linked on the GitHub page.
     2. Download and unzip the dataset into:
        data/Celeb-DF-v2/Celeb-real/
        data/Celeb-DF-v2/Celeb-synthesis/
        data/Celeb-DF-v2/YouTube-real/
""")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Setup and Download Guide")
    parser.add_argument("--setup_dirs", action="store_true", help="Create directory hierarchy")
    parser.add_argument("--gen_sample", action="store_true", help="Generate synthetic test data")
    args = parser.parse_args()
    
    print_dataset_instructions()
    create_directory_hierarchy()
    
    if args.gen_sample:
        generate_dummy_sample_data()
