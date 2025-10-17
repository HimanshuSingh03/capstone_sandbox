"""
Script to convert only test-clean LibriSpeech dataset to Scaper format for quick testing.
"""

import os
import argparse
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir', type=str, default='data/MixLibriSpeech',
        help='Path to datasets')

    args = parser.parse_args()

    dset = 'test-clean'
    dset_dir = os.path.join('LibriSpeech', dset)
    print(f"Processing {dset} for testing...")
    
    for speaker in tqdm(os.listdir(os.path.join(args.root_dir, dset_dir))):
        speaker_dir = os.path.join(dset_dir, speaker)
        out_dir = os.path.join(args.root_dir, 'librispeech_scaper_fmt', dset, speaker)
        
        # Skip if already exists
        if os.path.exists(out_dir):
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        
        for chapter in os.listdir(os.path.join(args.root_dir, speaker_dir)):
            chapter_dir = os.path.join(speaker_dir, chapter)
            for audiofile in os.listdir(os.path.join(args.root_dir, chapter_dir)):
                if not audiofile.endswith('.flac'):
                    continue
                audiofile_path = os.path.join(chapter_dir, audiofile)
                out_path = os.path.join(out_dir, audiofile)
                abs_src = os.path.abspath(os.path.join(args.root_dir, audiofile_path))
                abs_dst = os.path.abspath(out_path)
                
                if not os.path.exists(abs_dst):
                    try:
                        os.link(abs_src, abs_dst)  # Create hard link
                    except OSError:
                        # Fallback to copy if hard link fails
                        shutil.copy2(abs_src, abs_dst)