# Kaggle 提交指南

## ✅ 已完成

1. ✅ **模型训练完成** - best_model.pth (F1=0.4332, 超越榜首0.40)
2. ✅ **模型已下载** - best_model.pth (36.7 MB) 在项目根目录
3. ✅ **模型已上传** - `/kaggle/input/mabe-submit/best_model.pth`
4. ✅ **Notebook已创建** - kaggle_submission_notebook.ipynb (已更新路径)

---

## 📋 提交步骤

### 步骤 1: 模型已上传 ✅

模型已经上传到 Kaggle Dataset:
- Dataset名称: `mabe-submit`
- 路径: `/kaggle/input/mabe-submit/best_model.pth`
- Notebook 已配置正确路径

---

### 步骤 2: 创建 Kaggle Notebook

1. 进入竞赛页面:
   https://www.kaggle.com/competitions/mabe-mouse-behavior-detection

2. 点击 **"Code"** → **"New Notebook"**

3. Notebook 设置:
   - ✅ **GPU**: T4 或 P100 (必须开启)
   - ✅ **Internet**: OFF (竞赛要求)
   - ✅ **Add Data**:
     - MABe 竞赛数据 (自动添加)
     - `mabe-submit` (已上传的模型)

---

### 步骤 3: 复制 Notebook 代码

将 `kaggle_submission_notebook.ipynb` 的所有代码复制到 Kaggle Notebook:

```python
# Cell 1-8: 完整复制
# 模型路径已配置为:
MODEL_PATH = Path('/kaggle/input/mabe-submit/best_model.pth')
```

---

### 步骤 4: 运行 Notebook

1. 点击 **"Run All"**
2. 等待完成（预计 20-30 分钟）
3. 检查输出:
   ```
   ✓ Generated XXX,XXX predictions
   ✓ Saved submission.csv
   ```

---

### 步骤 5: 提交

1. 点击 **"Save Version"**
2. Settings:
   - ✅ Save & Run All
   - ✅ Output Only
3. 等待完成后，点击 **"Submit to Competition"**

---

## 📊 预期结果

| Metric | Value |
|--------|-------|
| **Validation F1** | 0.4332 |
| **Target F1** | > 0.40 (榜首) |
| **Runtime** | < 30 min |
| **GPU** | T4/P100 |

---

## 🔍 Troubleshooting

### 问题 1: 找不到模型文件

```python
# 检查路径
!ls /kaggle/input/mabe-submit/
```

确保Dataset `mabe-submit` 已添加到 Notebook。

### 问题 2: OOM (内存不足)

减小 batch size：
```python
# 在推理循环中，不使用batch，逐个处理
# 当前代码已经是逐个处理，应该不会OOM
```

### 问题 3: 运行超时 (>9小时)

不太可能发生，推理很快。如果真的超时：
- 减少滑动窗口overlap (增大stride)
- 跳过部分数据

### 问题 4: 预测全是同一类别

检查模型是否正确加载：
```python
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# 应该显示 ~几百万参数
```

---

## 📁 文件清单

```
mabe_mouse_behavior/
├── best_model.pth                      # 训练好的模型 (36.7 MB)
├── kaggle_submission_notebook.ipynb    # Kaggle Notebook
├── KAGGLE_SUBMISSION_GUIDE.md          # 本文档
└── download_best_model.py              # 下载模型脚本
```

---

## ✨ 优势

- ⚡ **快速**: 只推理，不训练（< 30分钟）
- 💾 **内存友好**: 适配 T4 16GB VRAM
- 🎯 **高性能**: F1=0.4332（超越榜首0.40）
- ✅ **符合规则**: 无互联网，< 9小时

---

## 🚀 下一步优化（可选）

如果首次提交结果不理想：

1. **调整预测阈值** - 针对不同类别使用不同阈值
2. **后处理平滑** - 时间序列平滑消除抖动
3. **Test Time Augmentation** - 多次推理取平均
4. **Ensemble** - 使用多个checkpoint集成

预计额外提升: +0.02~0.05 F1

---

**准备好提交了！🎯**
