#!/bin/bash

# UCF-101 Video Classification Pipeline
# This script preprocesses videos and prepares for training

echo "=== UCF-101 Video Classification Pipeline ==="

# Set data path
data_path="/data2/lpq/video-classification"

# Check if required packages are installed
echo "Checking required packages..."

python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing OpenCV..."
    pip install opencv-python
fi

python3 -c "import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing tqdm..."
    pip install tqdm
fi

python3 -c "import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing PyYAML..."
    pip install PyYAML
fi

echo "All required packages are installed!"

# Step 1: Preprocess videos to frames
echo ""
echo "Step 1: Preprocessing videos to frames..."
echo "Note: This process can be interrupted with Ctrl+C and resumed later."
echo "Progress is automatically saved every 100 videos."
echo ""

python3 preprocess_videos.py \
    --input_dir "./jpegs_256/UCF-101" \
    --output_dir "$data_path/jpegs_256_processed" \
    --max_frames 28 \
    --width 256 \
    --height 256

if [ $? -ne 0 ]; then
    echo "Error: Video preprocessing failed!"
    exit 1
fi

echo "✅ Video preprocessing completed successfully!"

# Step 2: Validate data format
echo ""
echo "Step 2: Validating data format..."
python3 test_data_format.py --data_path "$data_path/jpegs_256_processed" --test_all

if [ $? -ne 0 ]; then
    echo "Error: Data format validation failed!"
    exit 1
fi

echo "✅ Data format validation passed!"

# Step 3: Check configuration files
echo ""
echo "Step 3: Checking configuration files..."
echo "Verifying that all model configs exist..."

configs=(
    "configs/Conv3D_train.yaml"
    "configs/CRNN_train.yaml"
    "configs/ResNetCRNN_train.yaml"
    "configs/ResNetCRNN_varylength_train.yaml"
    "configs/swintransformer-RNN_train.yaml"
)

for config in "${configs[@]}"; do
    if [ -f "$config" ]; then
        echo "✅ $config exists"
    else
        echo "❌ $config missing"
    fi
done

# Step 4: Display training commands
echo ""
echo "Step 4: Ready to run training!"
echo ""
echo "All models are now configured to use the processed data at:"
echo "  $data_path/jpegs_256_processed"
echo ""
echo "To run training, use one of these commands:"
echo ""
echo "  # ResNetCRNN (Recommended - best accuracy ~85%)"
echo "  python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml"
echo ""
echo "  # CRNN (Medium accuracy ~54%)"
echo "  python train.py --model CRNN --config configs/CRNN_train.yaml"
echo ""
echo "  # Conv3D (Basic accuracy ~50%)"
echo "  python train.py --model Conv3D --config configs/Conv3D_train.yaml"
echo ""
echo "  # SwinTransformer-RNN (Latest architecture)"
echo "  python train.py --model swintransformer-RNN --config configs/swintransformer-RNN_train.yaml"
echo ""
echo "Optional training parameters:"
echo "  --epochs 100          # Override epochs from config"
echo "  --batch_size 16       # Override batch size from config"
echo "  --lr 0.0001          # Override learning rate from config"
echo ""
echo "Example with custom parameters:"
echo "  python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --epochs 100 --batch_size 16"
echo ""

# Step 5: Check GPU availability
echo "Step 5: Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  No GPU detected - training will be slow on CPU')
    print('   Consider using a GPU for better performance')
"

echo ""
echo "=== Pipeline Setup Completed! ==="
echo ""
echo "Next steps:"
echo "1. Review the configuration files in configs/ directory"
echo "2. Adjust training parameters if needed"
echo "3. Run training with one of the commands above"
echo "4. Monitor training progress and results"
echo ""
echo "For evaluation after training:"
echo "  python eval.py --model ResNetCRNN --epoch 50"
echo "  python eval.py --all  # Evaluate all models"
echo ""
echo "For single sample prediction:"
echo "  python predict.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --checkpoint results/ResNetCRNN/result/ResNetCRNN_ckpt/best_model.pth --sample /path/to/video/frames --topk 5" 