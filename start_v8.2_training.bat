@echo off
echo ============================================================
echo V8.2 Fine-Tuned Training Launcher
echo ============================================================
echo.
echo V8.2 Key Improvements:
echo - Smoothing kernel = 3 (better for short segments)
echo - Lower sniff* thresholds (0.25-0.28) for recall
echo - Larger sniffbody merge_gap (8) for better merging
echo - Motion gating maintained
echo - Kaggle F1 tracking every epoch
echo.
echo Press Ctrl+C to stop training
echo ============================================================
echo.

python train_v8.2_local.py

echo.
echo ============================================================
echo Training finished!
echo Best model: checkpoints/v8.2_fine_tuned/best_model.pth
echo ============================================================
pause
