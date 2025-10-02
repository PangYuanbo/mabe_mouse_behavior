# 在 Modal 上直接下载 Kaggle 数据

## 步骤 1: 获取 Kaggle API 凭证

1. 访问 https://www.kaggle.com/settings/account
2. 滚动到 "API" 部分
3. 点击 "Create New Token"
4. 下载 `kaggle.json` 文件

`kaggle.json` 内容示例：
```json
{
  "username": "your_username",
  "key": "your_api_key_here"
}
```

## 步骤 2: 创建 Modal Secret

使用 `kaggle.json` 中的信息创建 Modal secret：

```bash
# 方法 1: 直接从命令行创建
modal secret create kaggle-secret \
  KAGGLE_USERNAME=your_username \
  KAGGLE_KEY=your_api_key_here

# 方法 2: 从 kaggle.json 文件创建（推荐）
# 首先提取信息
KAGGLE_USERNAME=$(cat ~/Downloads/kaggle.json | python3 -c "import sys, json; print(json.load(sys.stdin)['username'])")
KAGGLE_KEY=$(cat ~/Downloads/kaggle.json | python3 -c "import sys, json; print(json.load(sys.stdin)['key'])")

# 然后创建 secret
modal secret create kaggle-secret \
  KAGGLE_USERNAME=$KAGGLE_USERNAME \
  KAGGLE_KEY=$KAGGLE_KEY
```

## 步骤 3: 接受竞赛规则

**重要**: 必须在 Kaggle 网站上接受竞赛规则才能下载数据！

1. 访问竞赛页面: https://www.kaggle.com/competitions/mabe-2022-behavior-challenge
2. 点击 "Join Competition" 或 "I Understand and Accept"
3. 接受规则和条款

## 步骤 4: 下载数据到 Modal

```bash
# 运行下载脚本（直接在 Modal 云端下载）
modal run download_kaggle_data.py
```

这会：
- ✅ 在 Modal 云端安装 kaggle
- ✅ 使用你的 Kaggle 凭证
- ✅ 下载竞赛数据
- ✅ 自动解压到 `/vol/data/kaggle/`
- ✅ 保存到 Modal Volume（持久化存储）

## 步骤 5: 验证下载

下载完成后，数据将存储在 Modal Volume 的以下位置：

```
/vol/data/kaggle/
├── train/              # 训练数据
├── test/               # 测试数据
└── sample_submission.csv
```

## 步骤 6: 更新训练配置

编辑 `modal_train_advanced.py`，确保数据路径正确：

```python
# 数据路径
train_data_dir = "/vol/data/kaggle/train"
val_data_dir = "/vol/data/kaggle/train"  # 将自动分割
```

## 步骤 7: 开始训练

```bash
modal run modal_train_advanced.py
```

## 常见问题

### Q: 出现 "403 Forbidden" 错误

**A**: 你还没有接受竞赛规则。访问竞赛页面并点击 "Join Competition"。

### Q: 出现 "Could not find kaggle.json" 错误

**A**: 检查 Modal secret 是否正确创建：
```bash
modal secret list
```
应该能看到 `kaggle-secret`。

### Q: 如何查看已下载的文件？

**A**: 创建一个简单的 Modal 函数查看：
```bash
modal run download_kaggle_data.py::list_files
```

### Q: 数据下载太慢

**A**: Modal 云端下载速度通常很快（10-100 MB/s）。如果竞赛数据很大（几GB），可能需要几分钟。

### Q: 如何重新下载数据？

**A**: 数据已存储在 Modal Volume 中，除非删除 Volume，否则不需要重新下载。如需重新下载：
```bash
# 选项 1: 删除 Volume 重新创建
modal volume delete mabe-data
modal run download_kaggle_data.py

# 选项 2: 在 Modal 函数中删除旧数据
# 修改脚本添加清理逻辑
```

## 数据规模参考

MABe 2022 竞赛数据：
- 训练数据: ~2-3 GB
- 测试数据: ~1 GB
- 总计: ~4 GB

下载时间估算：
- Modal 云端: 5-10 分钟
- 本地下载再上传: 20-60 分钟（取决于网速）

**建议**: 直接在 Modal 下载可节省大量时间！
