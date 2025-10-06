@echo off
REM V8.5 Full Behavior Coverage Training
REM ALL 37 behaviors (not just 27 like V8/V8.1)

echo ========================================
echo V8.5 Full Behavior Coverage Training
echo ALL 37 Behaviors Included
echo ========================================
echo.

python train_v8.5_local.py --config configs/config_v8.5_full_behaviors.yaml

echo.
echo ========================================
echo Training Complete!
echo Check: checkpoints/v8.5_full_behaviors/
echo ========================================
pause
