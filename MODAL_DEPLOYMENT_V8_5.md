# Modal 部署指南 - V8.5 No Class Weights

## 快速部署（3 步）

### 1. 设置 Modal Workspace
```bash
# 设置为 ybpang-1 workspace
modal profile set ybpang-1
```

### 2. 上传代码和配置
```bash
# 上传 src, configs, versions 到 Modal volume
modal run upload_code_to_modal.py
```

### 3. 启动训练
```bash
# 在 A10G GPU 上训练 V8.5 无权重版本
modal run modal_train_v8_5_no_weights.py
```

---

## 详细说明

### 环境要求
- Modal CLI 已安装: `pip install modal`
- 已登录 Modal: `modal token set`
- 已设置 workspace: `modal profile set ybpang-1`

### 训练配置
| 项目 | 配置 |
|-----|------|
| **GPU** | A10G (24GB VRAM) |
| **RAM** | 32GB |
| **Timeout** | 10 小时 |
| **Volume** | mabe-data |
| **Config** | config_v8.5_no_weights.yaml |
| **Batch Size** | 256 |
| **Epochs** | 100 |
| **类权重** | 无（Focal Loss only, γ=2.0） |

### 目录结构（Modal Volume）
```
/vol/
├── data/kaggle/              # 训练数据
│   ├── train_annotation/
│   └── train_keypoints/
├── code/                     # 源代码
│   ├── configs/
│   │   └── config_v8.5_no_weights.yaml
│   ├── src/
│   └── versions/
│       └── v8_5_full_behaviors/
└── checkpoints/              # 模型保存
    └── v8_5_no_weights_best_f1.pt
```

### 上传的文件清单
运行 `modal run upload_code_to_modal.py` 会上传：

1. **configs/** - 所有 .yaml 配置文件
   - ✅ config_v8.5_no_weights.yaml
   - config_v8.5_full_behaviors.yaml
   - 其他配置...

2. **src/** - 所有源代码
   - data/
   - models/
   - utils/
   - ...

3. **versions/** - 所有版本代码（新增！）
   - ✅ v8_5_full_behaviors/
     - v8_5_model.py
     - v8_5_dataset.py
     - action_mapping.py
     - submission_utils.py
   - v8_fine_grained/
   - v8_1_optimized_postproc/
   - ...

### 训练流程

#### Step 1: 上传代码
```bash
modal run upload_code_to_modal.py
```

**预期输出**:
```
============================================================
Uploading source code to Modal volume 'mabe-data'
============================================================
  Found: src/data/kaggle_dataset.py
  Found: configs/config_v8.5_no_weights.yaml
  Found: versions/v8_5_full_behaviors/v8_5_model.py
  ...
Total files: 150+

Uploading to Modal...
✓ Uploaded: src/data/kaggle_dataset.py
✓ Uploaded: configs/config_v8.5_no_weights.yaml
...
============================================================
✓ Code upload complete!
============================================================
```

#### Step 2: 启动训练
```bash
modal run modal_train_v8_5_no_weights.py
```

**预期输出**:
```
============================================================
V8.5 No Class Weights Training on Modal A10G
38 Behaviors, Focal Loss Only (γ=2.0)
============================================================
Config: config_v8.5_no_weights.yaml
Device: cuda
GPU: NVIDIA A10G
VRAM: 24.0 GB

Model: v8.5_multitask
NUM_ACTIONS: 38
Batch Size: 256
Use Class Weights: False
Focal Gamma: 2.0

Loading data...
[OK] Train batches: 1234
[OK] Val batches: 234
[OK] Input dimension: 112

Creating V8.5 model...
[OK] Total parameters: 8,234,567
[OK] Loss: Focal (γ=2.0), No Class Weights

============================================================
Starting Training
============================================================

Epoch 1/100
------------------------------------------------------------
Training: 100%|██████████| 1234/1234 [02:15<00:00]
Train Loss: 1.2345 | Acc: 0.7654
Validation: 100%|██████████| 234/234 [00:30<00:00]
Val Loss: 1.3456 | Acc: 0.7543

Top 10 Behaviors (Frame-level F1):
  [ 0] background          : F1=0.8765
  [ 1] attack              : F1=0.5432
  ...

Macro F1: 0.2345
[SAVE] Best F1 model: 0.2345

...

============================================================
Training Complete!
Best F1: 0.2543
Best Acc: 0.7821
============================================================
[COMMIT] Final volume save complete
```

### 模型检查点

训练完成后，模型保存在：
```
/vol/checkpoints/v8_5_no_weights_best_f1.pt    # 最佳 F1
/vol/checkpoints/v8_5_no_weights_best_acc.pt   # 最佳准确率
```

### 下载模型（可选）
```python
# download_checkpoints.py
import modal

app = modal.App("download-checkpoints")
volume = modal.Volume.from_name("mabe-data")

@app.function(volumes={"/vol": volume})
def download():
    import shutil
    shutil.copy(
        "/vol/checkpoints/v8_5_no_weights_best_f1.pt",
        "/tmp/v8_5_no_weights_best_f1.pt"
    )
    return open("/tmp/v8_5_no_weights_best_f1.pt", "rb").read()

@app.local_entrypoint()
def main():
    data = download.remote()
    with open("v8_5_no_weights_best_f1.pt", "wb") as f:
        f.write(data)
    print("✓ Downloaded: v8_5_no_weights_best_f1.pt")
```

---

## 监控训练

### 查看运行中的任务
```bash
modal app list
```

### 查看日志
```bash
modal app logs mabe-v8-5-no-weights
```

### 停止训练
```bash
modal app stop mabe-v8-5-no-weights
```

---

## 成本估算

| 项目 | 详情 |
|-----|-----|
| **GPU** | A10G @ $0.80/小时 |
| **预计时长** | 6-8 小时（100 epochs） |
| **预计成本** | **$4.80-$6.40** |

比 H100 节省 ~70%（H100 约 $2.90/小时）

---

## 对比：V8.5 原版 vs 无权重

### 启动命令对比

**V8.5 原版（有类权重）**:
```bash
# 需要创建对应的 Modal 脚本
modal run modal_train_v8_5_full.py
```

**V8.5 无权重**:
```bash
modal run modal_train_v8_5_no_weights.py
```

### 预期性能对比
| 指标 | V8.5 原版 | V8.5 无权重 |
|-----|----------|------------|
| **训练时间/epoch** | ~2.5 分钟 | ~2.3 分钟（快 10%） |
| **常见行为 F1** | 0.4-0.6 | 0.4-0.6（相同） |
| **稀有行为 F1** | 0.15-0.25 | 0.05-0.15（降低） |
| **整体 F1** | 0.2554 | 0.22-0.25（略低） |
| **稳定性** | 中等（极端权重） | 高（无极端权重） |

---

## Workspace 配置

### 切换到 ybpang-1
```bash
# 查看当前 workspace
modal profile current

# 切换到 ybpang-1
modal profile set ybpang-1

# 验证
modal profile current
# 应输出: ybpang-1
```

### Volume 检查
```bash
# 查看 volume
modal volume list

# 应包含:
# mabe-data (shared volume)
```

---

## 故障排除

### 问题 1: ModuleNotFoundError
**错误**: `ModuleNotFoundError: No module named 'versions.v8_5_full_behaviors'`

**解决**:
```bash
# 重新上传代码
modal run upload_code_to_modal.py
```

### 问题 2: Config 文件未找到
**错误**: `FileNotFoundError: config_v8.5_no_weights.yaml`

**解决**:
```bash
# 确保配置文件存在
ls configs/config_v8.5_no_weights.yaml

# 重新上传
modal run upload_code_to_modal.py
```

### 问题 3: Volume 数据缺失
**错误**: `No such file or directory: '/vol/data/kaggle'`

**解决**:
```bash
# 检查是否已上传 Kaggle 数据
# 如果没有，需要先上传数据（使用单独的数据上传脚本）
```

### 问题 4: Workspace 错误
**错误**: `Workspace not found: ybpang-1`

**解决**:
```bash
# 列出所有 workspace
modal profile list

# 使用正确的 workspace 名称
modal profile set <correct-workspace-name>
```

---

## 完整部署检查清单

- [ ] Modal CLI 已安装并登录
- [ ] Workspace 设置为 ybpang-1
- [ ] 配置文件已创建: `configs/config_v8.5_no_weights.yaml`
- [ ] Modal 训练脚本已创建: `modal_train_v8_5_no_weights.py`
- [ ] 上传脚本已更新（包含 versions 目录）
- [ ] 运行 `modal run upload_code_to_modal.py`
- [ ] 验证代码已上传到 volume
- [ ] 运行 `modal run modal_train_v8_5_no_weights.py`
- [ ] 监控训练日志
- [ ] 训练完成后下载模型

---

## 快速参考

### 一键部署
```bash
# 1. 设置 workspace
modal profile set ybpang-1

# 2. 上传代码
modal run upload_code_to_modal.py

# 3. 启动训练
modal run modal_train_v8_5_no_weights.py
```

### 监控
```bash
# 实时日志
modal app logs mabe-v8-5-no-weights --follow

# 任务状态
modal app list
```

### 清理
```bash
# 停止任务
modal app stop mabe-v8-5-no-weights

# 删除 volume（慎用！）
# modal volume delete mabe-data
```

---

**准备好了吗？开始部署！** 🚀

```bash
modal profile set ybpang-1 && modal run upload_code_to_modal.py && modal run modal_train_v8_5_no_weights.py
```
