@echo off
call conda activate Fed_ADP(2)
python train_v7_local.py --config configs/config_v7_5090.yaml
