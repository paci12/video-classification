#!/usr/bin/env python3
"""
Test script to verify data format compatibility with the video classification project
"""

import os
import numpy as np
from PIL import Image
import argparse

def test_data_format(data_path, action_name="ApplyEyeMakeup", video_name="v_ApplyEyeMakeup_g01_c01"):
    """
    Test if the data format matches the project's expectations
    
    Args:
        data_path: Path to the processed data directory
        action_name: Name of an action category to test
        video_name: Name of a video to test
    """
    print(f"Testing data format in: {data_path}")
    
    # Test directory structure
    video_dir = os.path.join(data_path, action_name, video_name)
    print(f"Looking for video directory: {video_dir}")
    
    if not os.path.exists(video_dir):
        print(f"❌ Error: Video directory does not exist: {video_dir}")
        return False
    
    print(f"✅ Video directory exists: {video_dir}")
    
    # Test frame files
    expected_frames = list(range(1, 29))  # frames 1-28
    missing_frames = []
    found_frames = []
    
    for frame_idx in expected_frames:
        frame_filename = f"frame{frame_idx:06d}.jpg"
        frame_path = os.path.join(video_dir, frame_filename)
        
        if os.path.exists(frame_path):
            found_frames.append(frame_idx)
            
            # Test if it's a valid image
            try:
                img = Image.open(frame_path)
                if img.size != (256, 256):
                    print(f"⚠️  Warning: Frame {frame_idx} has size {img.size}, expected (256, 256)")
            except Exception as e:
                print(f"❌ Error: Frame {frame_idx} is not a valid image: {e}")
                return False
        else:
            missing_frames.append(frame_idx)
    
    print(f"✅ Found {len(found_frames)} frames: {found_frames[:5]}...{found_frames[-5:] if len(found_frames) > 10 else ''}")
    
    if missing_frames:
        print(f"❌ Missing frames: {missing_frames}")
        return False
    else:
        print("✅ All expected frames found!")
    
    # Test frame loading with PIL (as the project does)
    print("\nTesting frame loading with PIL...")
    try:
        test_frame_path = os.path.join(video_dir, "frame000001.jpg")
        img = Image.open(test_frame_path)
        print(f"✅ Successfully loaded frame: {img.size}, mode: {img.mode}")
    except Exception as e:
        print(f"❌ Error loading test frame: {e}")
        return False
    
    return True

def test_multiple_videos(data_path, max_tests=5):
    """
    Test multiple videos to ensure consistency
    """
    print(f"\nTesting multiple videos (max {max_tests})...")
    
    success_count = 0
    total_count = 0
    
    for action_dir in os.listdir(data_path):
        action_path = os.path.join(data_path, action_dir)
        if not os.path.isdir(action_path):
            continue
            
        for video_dir in os.listdir(action_path):
            video_path = os.path.join(action_path, video_dir)
            if not os.path.isdir(video_path):
                continue
                
            total_count += 1
            if total_count > max_tests:
                break
                
            print(f"\nTesting: {action_dir}/{video_dir}")
            if test_data_format(data_path, action_dir, video_dir):
                success_count += 1
            else:
                print(f"❌ Failed to test: {action_dir}/{video_dir}")
                
        if total_count > max_tests:
            break
    
    print(f"\n📊 Test Results: {success_count}/{total_count} videos passed format validation")
    return success_count == total_count

def main():
    parser = argparse.ArgumentParser(description="Test data format compatibility")
    parser.add_argument("--data_path", type=str, default="./jpegs_256_processed",
                       help="Path to processed data directory")
    parser.add_argument("--test_all", action="store_true",
                       help="Test multiple videos instead of just one")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data path does not exist: {args.data_path}")
        print("Please run the preprocessing script first:")
        print("python preprocess_videos.py")
        return
    
    if args.test_all:
        success = test_multiple_videos(args.data_path)
    else:
        success = test_data_format(args.data_path)
    
    if success:
        print("\n🎉 Data format validation passed! The dataset is ready for training.")
    else:
        print("\n❌ Data format validation failed! Please check the preprocessing script.")

if __name__ == "__main__":
    main() 