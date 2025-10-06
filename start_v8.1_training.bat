@echo off
echo ============================================================
echo V8.1 Optimized Training Launcher
echo ============================================================
echo.
echo Starting V8.1 training with:
echo - Motion gating (velocity thresholds)
echo - Fine-tuned class thresholds
echo - Kaggle F1 tracking every epoch
echo - Best model saved based on Kaggle F1
echo.
echo Press Ctrl+C to stop training
echo ============================================================
echo.

python train_v8.1_local.py

echo.
echo ============================================================
echo Training finished!
echo Check: checkpoints/v8.1_optimized/best_model.pth
echo ============================================================
pause
