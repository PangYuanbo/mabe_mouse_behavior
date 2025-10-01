# Kaggle API 配置指南

## 获取 Kaggle API Token

1. 访问 https://www.kaggle.com/settings/account
2. 滚动到 "API" 部分
3. 点击 "Create New Token"
4. 会下载一个 `kaggle.json` 文件

## 配置步骤

```bash
# 创建配置目录
mkdir -p ~/.kaggle

# 移动 kaggle.json 到配置目录
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置正确的权限
chmod 600 ~/.kaggle/kaggle.json
```

## 验证配置

```bash
kaggle competitions list
```

如果配置成功，应该能看到 Kaggle 竞赛列表。

## 下载 MABe 竞赛数据

```bash
cd /Users/aaronpang/PycharmProjects/mabe_mouse_behavior
kaggle competitions download -c mabe-mouse-behavior-detection -p data
cd data
unzip mabe-mouse-behavior-detection.zip
```