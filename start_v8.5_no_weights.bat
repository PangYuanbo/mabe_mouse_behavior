@echo off
echo ========================================
echo V8.5 No Class Weights Training
echo 38 Classes, Focal Loss Only (like V8.1)
echo ========================================
echo.

python train_v8.5_local.py --config configs/config_v8.5_no_weights.yaml

pause
