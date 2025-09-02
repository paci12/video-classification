#!/usr/bin/env python3
"""
UCF-101 Video Preprocessing Script
Convert .avi videos to frame images for video classification
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import pickle

def extract_frames(video_path, output_dir, max_frames=28, resize_shape=(256, 256)):
    """
    Extract frames from video and save as images
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract
        resize_shape: Target size for resizing frames (width, height)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame interval to extract max_frames
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        # Uniformly sample frames
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    
    frame_count = 0
    success = True
    
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        frame = cv2.resize(frame, resize_shape)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame with format: frame000001.jpg, frame000002.jpg, etc.
        # Note: frame indices start from 1 to match the project's expected format
        frame_filename = f"frame{i+1:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        frame_count += 1
    
    cap.release()
    return frame_count > 0

def load_progress(progress_file):
    """Load progress from file"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return set(f.read().splitlines())
        except:
            return set()
    return set()

def save_progress(progress_file, processed_videos):
    """Save progress to file"""
    with open(progress_file, 'w') as f:
        for video in processed_videos:
            f.write(f"{video}\n")

def process_ucf101_dataset(input_dir, output_dir, max_frames=28, resize_shape=(256, 256), resume=True):
    """
    Process entire UCF-101 dataset with resume capability
    
    Args:
        input_dir: Directory containing UCF-101 videos
        output_dir: Directory to save processed frames
        max_frames: Maximum number of frames per video
        resize_shape: Target size for resizing frames
        resume: Whether to resume from previous progress
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Progress tracking
    progress_file = os.path.join(output_dir, "processed_videos.txt")
    processed_videos = load_progress(progress_file) if resume else set()
    
    # Get all video files
    video_files = []
    for video_file in input_path.rglob("*.avi"):
        video_files.append(video_file)
    
    print(f"Found {len(video_files)} video files")
    print(f"Already processed: {len(processed_videos)} videos")
    
    # Filter out already processed videos
    remaining_videos = [v for v in video_files if str(v) not in processed_videos]
    print(f"Remaining to process: {len(remaining_videos)} videos")
    
    if len(remaining_videos) == 0:
        print("All videos have been processed!")
        return
    
    # Process each video
    success_count = 0
    failed_count = 0
    
    try:
        for video_file in tqdm(remaining_videos, desc="Processing videos"):
            # Create relative path structure
            rel_path = video_file.relative_to(input_path)
            video_name = video_file.stem  # Remove .avi extension
            
            # Create output directory for this video
            video_output_dir = output_path / rel_path.parent / video_name
            
            # Extract frames
            success = extract_frames(
                str(video_file), 
                str(video_output_dir), 
                max_frames, 
                resize_shape
            )
            
            if success:
                success_count += 1
                processed_videos.add(str(video_file))
                # Save progress every 100 videos
                if success_count % 100 == 0:
                    save_progress(progress_file, processed_videos)
            else:
                failed_count += 1
                print(f"Failed to process: {video_file}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user!")
        print("Progress has been saved. You can resume later.")
        save_progress(progress_file, processed_videos)
        raise
    
    # Save final progress
    save_progress(progress_file, processed_videos)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} videos")
    print(f"Failed: {failed_count} videos")
    print(f"Total processed: {len(processed_videos)} videos")
    print(f"Output directory: {output_dir}")
    print(f"Progress saved to: {progress_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess UCF-101 videos to frames")
    parser.add_argument("--input_dir", type=str, default="./jpegs_256/UCF-101",
                       help="Input directory containing UCF-101 videos")
    parser.add_argument("--output_dir", type=str, default="./jpegs_256_processed",
                       help="Output directory for processed frames")
    parser.add_argument("--max_frames", type=int, default=28,
                       help="Maximum number of frames to extract per video")
    parser.add_argument("--width", type=int, default=256,
                       help="Target width for frame resizing")
    parser.add_argument("--height", type=int, default=256,
                       help="Target height for frame resizing")
    parser.add_argument("--no_resume", action="store_true",
                       help="Don't resume from previous progress (start fresh)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset
    process_ucf101_dataset(
        args.input_dir,
        args.output_dir,
        args.max_frames,
        (args.width, args.height),
        resume=not args.no_resume
    )

if __name__ == "__main__":
    main() 