#!/usr/bin/env python3
"""
Check progress and generate progress file for existing processed videos
"""

import os
from pathlib import Path
import argparse

def check_processed_videos(input_dir, output_dir):
    """
    Check which videos have already been processed and generate progress file
    
    Args:
        input_dir: Directory containing original UCF-101 videos
        output_dir: Directory containing processed frame images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all original video files
    original_videos = []
    for video_file in input_path.rglob("*.avi"):
        original_videos.append(video_file)
    
    print(f"Found {len(original_videos)} original video files")
    
    # Check which videos have been processed
    processed_videos = []
    missing_videos = []
    
    for video_file in original_videos:
        # Create expected output path
        rel_path = video_file.relative_to(input_path)
        video_name = video_file.stem  # Remove .avi extension
        video_output_dir = output_path / rel_path.parent / video_name
        
        # Check if output directory exists and has frame files
        if video_output_dir.exists():
            # Check if it has the expected frame files
            frame_files = list(video_output_dir.glob("frame*.jpg"))
            if len(frame_files) >= 28:  # Should have at least 28 frames
                processed_videos.append(str(video_file))
            else:
                missing_videos.append(str(video_file))
                print(f"Incomplete: {video_file} (only {len(frame_files)} frames)")
        else:
            missing_videos.append(str(video_file))
    
    print(f"\nProgress Summary:")
    print(f"✅ Processed videos: {len(processed_videos)}")
    print(f"❌ Missing videos: {len(missing_videos)}")
    print(f"📊 Progress: {len(processed_videos)/len(original_videos)*100:.1f}%")
    
    # Save progress file
    progress_file = os.path.join(output_dir, "processed_videos.txt")
    with open(progress_file, 'w') as f:
        for video in processed_videos:
            f.write(f"{video}\n")
    
    print(f"\nProgress saved to: {progress_file}")
    
    return processed_videos, missing_videos

def main():
    parser = argparse.ArgumentParser(description="Check preprocessing progress")
    parser.add_argument("--input_dir", type=str, default="./jpegs_256/UCF-101",
                       help="Input directory containing UCF-101 videos")
    parser.add_argument("--output_dir", type=str, default="./jpegs_256_processed",
                       help="Output directory for processed frames")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist")
        return
    
    processed, missing = check_processed_videos(args.input_dir, args.output_dir)
    
    if len(missing) > 0:
        print(f"\nTo continue processing, run:")
        print(f"python preprocess_videos.py --input_dir {args.input_dir} --output_dir {args.output_dir}")
    else:
        print(f"\n🎉 All videos have been processed!")

if __name__ == "__main__":
    main() 