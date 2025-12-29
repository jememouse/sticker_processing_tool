"""
贴纸生成 API 的 Pydantic 模型定义
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AIModelType(str, Enum):
    """AI 抠图模型类型"""
    BIREFNET = "birefnet"
    REMBG = "rembg"


class StickerGenerateRequest(BaseModel):
    """贴纸生成请求参数"""
    file_id: str = Field(..., description="上传文件的临时 ID")
    outline_width: int = Field(default=15, ge=5, le=50, description="白边宽度 (5-50 像素)")
    model_type: AIModelType = Field(default=AIModelType.BIREFNET, description="AI 抠图模型类型")


class UploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str = Field(..., description="临时文件 ID")
    filename: str = Field(..., description="原始文件名")
    preview_url: str = Field(..., description="原图预览 URL")


class GenerateResponse(BaseModel):
    """贴纸生成响应"""
    task_id: str = Field(..., description="生成任务 ID")
    status: str = Field(..., description="任务状态: processing, completed, failed")
    preview_url: Optional[str] = Field(None, description="预览图 URL")
    pdf_url: Optional[str] = Field(None, description="PDF 下载 URL")
    svg_url: Optional[str] = Field(None, description="SVG 下载 URL")
    error: Optional[str] = Field(None, description="错误信息")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    preview_url: Optional[str] = None
    pdf_url: Optional[str] = None
    svg_url: Optional[str] = None
    error: Optional[str] = None
