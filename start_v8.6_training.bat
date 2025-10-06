@echo off
echo ============================================================
echo Starting V8.6 MARS-Enhanced Training
echo ============================================================
echo.
echo Target: Interval F1 = 0.35+ (from V8.5's 0.255)
echo Key improvements:
echo   - MARS features (370 dims)
echo   - Multi-scale temporal conv
echo   - Freeze detection branch
echo   - Rare behavior oversampling (3x)
echo.
echo Expected training time: ~2 hours on RTX 5090
echo ============================================================
echo.

python train_v8_6_local.py --config configs/config_v8.6_mars.yaml

echo.
echo ============================================================
echo Training completed!
echo Check: checkpoints/v8_6/v8_6_best.pt
echo ============================================================
pause
