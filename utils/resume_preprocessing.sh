#!/bin/bash

# Smart Resume UCF-101 Video Preprocessing
# This script can handle both cases:
# 1. New version with progress file
# 2. Old version without progress file (will generate progress file first)

echo "=== Smart Resume UCF-101 Video Preprocessing ==="

data_path="/data2/lpq/video-classification"

# Check if processed directory exists
if [ ! -d "$data_path/jpegs_256_processed" ]; then
    echo "Error: No processed directory found. Please run the full preprocessing first."
    exit 1
fi

# Check if progress file exists
if [ -f "$data_path/jpegs_256_processed/processed_videos.txt" ]; then
    processed_count=$(wc -l < "$data_path/jpegs_256_processed/processed_videos.txt")
    echo "Found progress file with $processed_count processed videos"
    echo "Resuming from existing progress..."
else
    echo "No progress file found. Checking existing processed videos..."
    echo "Step 1: Checking existing progress and generating progress file..."
    python3 check_progress.py --input_dir "./jpegs_256/UCF-101" --output_dir "$data_path/jpegs_256_processed"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to check progress!"
        exit 1
    fi
fi

# Resume preprocessing
echo "Step 2: Resuming video preprocessing..."
python3 preprocess_videos.py \
    --input_dir "./jpegs_256/UCF-101" \
    --output_dir "$data_path/jpegs_256_processed" \
    --max_frames 28 \
    --width 256 \
    --height 256

if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully!"
    
    # Continue with validation and training setup
    echo "Step 3: Validating data format..."
    python3 test_data_format.py --data_path "$data_path/jpegs_256_processed" --test_all
    
    if [ $? -eq 0 ]; then
        echo "Step 4: Updating data paths in model files..."
        
        # Update ResNetCRNN model
        sed -i "s|data_path = \"./jpegs_256/\"|data_path = \"$data_path/jpegs_256_processed/\"|g" ResNetCRNN/UCF101_ResNetCRNN.py
        
        # Update CRNN model  
        sed -i "s|data_path = \"./jpegs_256/\"|data_path = \"$data_path/jpegs_256_processed/\"|g" CRNN/UCF101_CRNN.py
        
        # Update Conv3D model
        sed -i "s|data_path = \"./jpegs_256/\"|data_path = \"$data_path/jpegs_256_processed/\"|g" Conv3D/UCF101_3DCNN.py
        
        echo "Data paths updated!"
        echo ""
        echo "🎉 Ready to run training!"
        echo ""
        echo "Choose a model to train:"
        echo "1. ResNetCRNN (Recommended - best accuracy ~85%)"
        echo "2. CRNN (Medium accuracy ~54%)"
        echo "3. Conv3D (Basic accuracy ~50%)"
        echo ""
        echo "To run training, use one of these commands:"
        echo "  cd ResNetCRNN && python UCF101_ResNetCRNN.py"
        echo "  cd CRNN && python UCF101_CRNN.py"
        echo "  cd Conv3D && python UCF101_3DCNN.py"
    else
        echo "Data validation failed!"
        exit 1
    fi
else
    echo "Preprocessing failed!"
    exit 1
fi 