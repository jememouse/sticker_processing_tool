# Sticker Processing Tool

智能贴纸生成与印刷文件处理工具。

## 核心功能

### 智能贴纸生成器

支持两种 AI 抠图引擎：
- **BiRefNet**: 高精度边缘处理
- **Rembg (U2NET)**: 轻量级快速处理

**功能特点**：
- 自动抠图 & 智能白边生成
- 3mm 出血线裁切
- 印刷级 PDF 导出 (白墨层 + 图稿层 + 刀线层)
- SVG 刀模导出

## 快速开始

```bash
# 安装依赖
uv sync

# 启动服务 (前端 + 后端)
./start.sh
```

**访问地址**：
- 前端：http://localhost:5173
- 后端 API：http://localhost:8080
- API 文档：http://localhost:8080/docs

## 目录结构

```
.
├── backend/           # 后端 API 服务
│   ├── api/           # FastAPI 路由
│   └── sticker/       # 贴纸生成核心模块
├── frontend/          # Vue 3 前端
├── start.sh           # 统一启动脚本
└── pyproject.toml     # Python 依赖
```

## 环境要求

- Python >= 3.11, < 3.13
- Node.js (前端开发)
