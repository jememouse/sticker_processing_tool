"""
FastAPI 应用入口 - 营销素材工具 API 服务
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from api.routers import sticker

# 创建 FastAPI 应用
app = FastAPI(
    title="营销素材工具 API",
    description="智能贴纸生成器 API 服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加 CORS 中间件 (允许前端跨域访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(sticker.router)


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "营销素材工具 API 服务",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


# 启动命令: uv run uvicorn api.server:app --reload --port 8080
