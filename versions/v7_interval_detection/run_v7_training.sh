#!/bin/bash
# V7 Training Quick Start Script

echo "=================================================="
echo "V7 Interval Detection Training - RTX 5090"
echo "=================================================="

# Check GPU
echo -e "\n1. Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run motion features test
echo -e "\n2. Testing Motion Features implementation..."
python test_v7_motion_features.py

if [ $? -ne 0 ]; then
    echo "❌ Motion features test failed!"
    exit 1
fi

# Memory estimation
echo -e "\n3. Estimating memory usage..."
python estimate_v7_memory.py

# Ask user for configuration
echo -e "\n4. Select training configuration:"
echo "  1) Standard (batch=32, recommended) ⭐"
echo "  2) Maximum (batch=48, aggressive)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        CONFIG="configs/config_v7_5090.yaml"
        echo "Using standard configuration (batch=32)"
        ;;
    2)
        CONFIG="configs/config_v7_5090_max.yaml"
        echo "Using maximum configuration (batch=48)"
        ;;
    *)
        echo "Invalid choice, using standard configuration"
        CONFIG="configs/config_v7_5090.yaml"
        ;;
esac

# Confirm
echo -e "\n5. Training Configuration:"
echo "  Config file: $CONFIG"
grep "batch_size" $CONFIG
grep "learning_rate" $CONFIG
grep "mixed_precision" $CONFIG

read -p "Continue? [y/n]: " confirm

if [ "$confirm" != "y" ]; then
    echo "Training cancelled"
    exit 0
fi

# Start training
echo -e "\n6. Starting training..."
echo "=================================================="

# Monitor GPU in background
watch -n 5 nvidia-smi > gpu_monitor.log &
WATCH_PID=$!

# Run training
python train_v7_local.py --config $CONFIG

# Kill GPU monitor
kill $WATCH_PID

echo -e "\n=================================================="
echo "Training completed!"
echo "Check checkpoint: checkpoints/v7_5090/best_model_v7.pth"
echo "GPU log: gpu_monitor.log"
echo "=================================================="
