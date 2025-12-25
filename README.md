# Marketing Material Tools

这是一个专注于营销素材文件与信息处理的工具集，主要用于自动化生成贴纸、印刷文件和图像处理。

## 核心功能

本项目目前包含以下核心模块：

### 1. 智能贴纸生成器 (`sticker/`)

基于 `rembg` 和 `opencv-python` 的自动化贴纸生成工具，支持：

- **自动抠图**: 使用 AI 模型 (U2NET) 自动移除图片背景。
- **智能白边**: 自动计算并生成平滑的贴纸白边 (Outline)。
- **阴影效果**: 生成带有立体感的阴影效果，适合 Web 展示。
- **印刷级 PDF 导出**: 生成分层 PDF 文件，包含：
  - Layer 1: White Mask (矢量白底，用于铺白墨)
  - Layer 2: Artwork (原图)
  - Layer 3: CutLine (矢量刀线，用于切割机)
- **SVG 刀模导出**: 独立导出 SVG 格式的切割路径。

## 环境依赖

本项目要求 **Python >= 3.11, < 3.13**。

主要依赖库：

- `numpy`
- `opencv-python`
- `pillow`
- `rembg` (用于 AI 抠图)
- `reportlab` (用于 PDF 生成)
- `jsonschema`
- `pymatting`

## 快速开始

### 1. 安装依赖

推荐使用 `uv` 进行包管理，或者使用标准的 `pip`：

```bash
# 使用 pip
pip install -r requirements.txt
# (如果没有 requirements.txt，参考 pyproject.toml 安装)
pip install "rembg[gpu]" opencv-python pillow reportlab numpy
```

### 2. 贴纸生成示例

核心代码位于 `sticker/sticker-1.py`。该文件可以直接作为脚本运行，或在重命名为有效 Python 模块名（如 `sticker_generator.py`）后被引用。

**直接运行示例：**

```bash
# 确保在项目根目录下
python sticker/sticker-1.py
```

或者，如果你想在其他代码中使用 `StickerGenerator` 类，建议将 `sticker/sticker-1.py` 重命名为 `sticker_generator.py`，然后：

```python
# 假设已经重命名为 sticker_generator.py
from sticker.sticker_generator import StickerGenerator
import os

# 初始化生成器
generator = StickerGenerator()

# ... (后续调用逻辑同上)
```

## 目录结构

```
.
├── main.py               # 项目入口示例
├── pyproject.toml        # 项目依赖配置
├── sticker/              # 贴纸生成模块
│   ├── sticker-1.py      # 核心逻辑实现 (StickerGenerator)
│   ├── sticker-Rembg.py  # Rembg 相关测试代码
│   └── ...
└── ...
```

## 注意事项

- **Rembg 模型**: 首次运行 `rembg` 时会自动通过网络下载 `u2net` 模型，请确保网络通畅。
- **OpenCV**: 图像处理主要依赖 OpenCV，处理高分辨率图片时请注意内存消耗。
