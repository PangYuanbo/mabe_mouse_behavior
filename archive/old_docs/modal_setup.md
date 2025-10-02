# Modal 部署指南

## 1. 安装 Modal

```bash
pip install modal
```

## 2. Modal 认证

```bash
modal setup
```

这会打开浏览器进行认证。

## 3. 配置 Kaggle 凭证（可选）

如果需要从 Kaggle 下载数据：

```bash
# 从 kaggle.json 获取用户名和 API key
# 然后创建 Modal secret:
modal secret create kaggle-credentials \
    KAGGLE_USERNAME=<your_username> \
    KAGGLE_KEY=<your_api_key>
```

或者手动上传数据到 Modal volume（见方法 2）。

## 4. 准备代码

Modal 需要能访问源代码。有两种方式：

### 方法 1: 使用 Modal Mounts（推荐）

更新 `modal_train.py`，添加代码挂载：

```python
@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    mounts=[modal.Mount.from_local_dir("src", remote_path="/root/src")],
    secrets=[modal.Secret.from_name("kaggle-credentials")],
)
```

### 方法 2: 手动上传数据

如果不想从 Kaggle 下载，可以先上传本地数据：

```python
# 创建上传脚本 upload_data.py
import modal

app = modal.App("mabe-upload")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

@app.function(volumes={"/data": volume})
def upload_local_data():
    from pathlib import Path
    import shutil

    # 将本地数据复制到 volume
    local_data = Path("data")
    remote_data = Path("/data")

    for item in local_data.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(local_data)
            dest = remote_data / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    volume.commit()
    print("Data uploaded successfully!")

if __name__ == "__main__":
    with modal.enable_output():
        upload_local_data.remote()
```

## 5. 运行训练

### 使用示例数据（本地已有）:

```bash
# 首先上传本地数据到 Modal volume
modal run upload_data.py

# 然后训练
modal run modal_train.py
```

### 从 Kaggle 下载数据并训练:

```bash
modal run modal_train.py --download-data
```

### 使用自定义配置:

```bash
modal run modal_train.py --config configs/custom_config.yaml
```

## 6. 下载训练好的模型

```bash
# 在 modal_train.py 中已经定义了 download_checkpoint 函数
# 可以通过以下方式下载：

modal run modal_train.py::download_checkpoint --checkpoint-name best_model.pth > best_model.pth
```

或者创建下载脚本：

```python
import modal

app = modal.App("mabe-download")
volume = modal.Volume.from_name("mabe-data")

@app.function(volumes={"/data": volume})
def download():
    with open("/data/checkpoints/best_model.pth", "rb") as f:
        return f.read()

if __name__ == "__main__":
    data = download.remote()
    with open("best_model.pth", "wb") as f:
        f.write(data)
    print("Downloaded best_model.pth")
```

## 7. 监控训练

Modal 提供实时日志查看：

```bash
# 查看最近的运行
modal app logs mabe-mouse-behavior

# 实时查看
modal app logs mabe-mouse-behavior --follow
```

## GPU 选项

在 `modal_train.py` 中可以更改 GPU 类型：

```python
@app.function(
    gpu="T4",        # 便宜，适合实验
    # gpu="A10G",    # 中等性能
    # gpu="A100",    # 高性能
)
```

## 成本估算

- T4 GPU: ~$0.60/小时
- A10G GPU: ~$1.20/小时
- A100 GPU: ~$4.00/小时

根据配置，训练 10-100 epochs 大约需要 1-4 小时。

## 故障排除

### 找不到模块

确保在函数内部导入模块，而不是在文件顶部：

```python
@app.function(...)
def train_model():
    import torch  # ✓ 正确
    from src.data.dataset import get_dataloaders  # ✓ 正确
```

### 数据找不到

检查 volume 是否正确挂载，路径是否正确：

```bash
modal volume ls mabe-data
```

### Kaggle 认证失败

确保 secret 创建正确：

```bash
modal secret list
```