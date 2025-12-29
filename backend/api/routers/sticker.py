"""
贴纸生成相关 API 路由
"""
import os
import uuid
import shutil
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from api.schemas.sticker import (
    AIModelType,
    StickerGenerateRequest,
    UploadResponse,
    GenerateResponse,
    TaskStatusResponse,
)

router = APIRouter(prefix="/api/sticker", tags=["贴纸生成"])

# 临时文件存储目录
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp_uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp_outputs")

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 任务状态存储 (生产环境应使用 Redis 等)
task_store: dict = {}

# 延迟加载的生成器实例
_birefnet_generator = None
_rembg_generator = None


def get_birefnet_generator():
    """延迟加载 BiRefNet 生成器"""
    global _birefnet_generator
    if _birefnet_generator is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from sticker import sticker_birefnet
        _birefnet_generator = sticker_birefnet.StickerGenerator()
    return _birefnet_generator


def get_rembg_generator():
    """延迟加载 rembg 生成器"""
    global _rembg_generator
    if _rembg_generator is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from sticker import sticker_rembg
        _rembg_generator = sticker_rembg.StickerGenerator()
    return _rembg_generator


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片文件
    
    - 支持格式: jpg, jpeg, png, webp
    - 最大文件大小: 20MB
    """
    # 验证文件类型
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件类型: {file.content_type}。支持: jpg, png, webp"
        )
    
    # 验证文件大小 (20MB)
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小超过 20MB 限制")
    
    # 生成唯一文件 ID
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    # 保存文件
    with open(save_path, "wb") as f:
        f.write(content)
    
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        preview_url=f"/api/sticker/preview/original/{file_id}"
    )


@router.get("/preview/original/{file_id}")
async def get_original_preview(file_id: str):
    """获取上传的原图预览"""
    # 查找文件
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="文件未找到")


def process_sticker_task(task_id: str, file_id: str, outline_width: int, model_type: AIModelType):
    """后台任务: 处理贴纸生成"""
    try:
        task_store[task_id]["status"] = "processing"
        task_store[task_id]["progress"] = 10
        
        # 查找上传的文件
        input_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            if os.path.exists(path):
                input_path = path
                break
        
        if not input_path:
            task_store[task_id]["status"] = "failed"
            task_store[task_id]["error"] = "上传的文件未找到"
            return
        
        task_store[task_id]["progress"] = 20
        
        # 选择生成器
        if model_type == AIModelType.BIREFNET:
            generator = get_birefnet_generator()
        else:
            generator = get_rembg_generator()
        
        task_store[task_id]["progress"] = 30
        
        # 生成贴纸数据
        process_data = generator.process_image_to_data(input_path, outline_width=outline_width)
        task_store[task_id]["progress"] = 60
        
        # 创建输出目录
        task_output_dir = os.path.join(OUTPUT_DIR, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 生成 PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(task_output_dir, f"sticker_{timestamp}.pdf")
        generator.generate_pdf(process_data, pdf_path)
        task_store[task_id]["progress"] = 80
        
        # 生成 SVG
        svg_path = os.path.join(task_output_dir, f"cutline_{timestamp}.svg")
        generator.generate_svg_cutline(process_data["outline_mask"], svg_path)
        task_store[task_id]["progress"] = 90
        
        # 生成预览 PNG
        preview_path = os.path.join(task_output_dir, f"preview_{timestamp}.png")
        generator.process_image(input_path, preview_path, outline_width=outline_width)
        
        # 更新任务状态
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["progress"] = 100
        task_store[task_id]["preview_url"] = f"/api/sticker/download/{task_id}/preview"
        task_store[task_id]["pdf_url"] = f"/api/sticker/download/{task_id}/pdf"
        task_store[task_id]["svg_url"] = f"/api/sticker/download/{task_id}/svg"
        
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)


@router.post("/generate", response_model=GenerateResponse)
async def generate_sticker(request: StickerGenerateRequest, background_tasks: BackgroundTasks):
    """
    生成贴纸
    
    - 接收文件 ID 和参数
    - 后台异步处理
    - 返回任务 ID 用于轮询状态
    """
    # 验证文件是否存在
    file_exists = False
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if os.path.exists(os.path.join(UPLOAD_DIR, f"{request.file_id}{ext}")):
            file_exists = True
            break
    
    if not file_exists:
        raise HTTPException(status_code=404, detail="上传的文件未找到，请重新上传")
    
    # 创建任务
    task_id = str(uuid.uuid4())
    task_store[task_id] = {
        "status": "pending",
        "progress": 0,
        "preview_url": None,
        "pdf_url": None,
        "svg_url": None,
        "error": None,
    }
    
    # 添加后台任务
    background_tasks.add_task(
        process_sticker_task,
        task_id,
        request.file_id,
        request.outline_width,
        request.model_type
    )
    
    return GenerateResponse(
        task_id=task_id,
        status="pending"
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    task = task_store[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        preview_url=task.get("preview_url"),
        pdf_url=task.get("pdf_url"),
        svg_url=task.get("svg_url"),
        error=task.get("error"),
    )


@router.get("/download/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """
    下载生成的文件
    
    - file_type: preview, pdf, svg
    """
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    task = task_store[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    task_output_dir = os.path.join(OUTPUT_DIR, task_id)
    
    # 查找对应文件
    if file_type == "preview":
        pattern = "preview_"
        media_type = "image/png"
    elif file_type == "pdf":
        pattern = "sticker_"
        media_type = "application/pdf"
    elif file_type == "svg":
        pattern = "cutline_"
        media_type = "image/svg+xml"
    else:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    
    # 查找文件
    for filename in os.listdir(task_output_dir):
        if filename.startswith(pattern):
            file_path = os.path.join(task_output_dir, filename)
            return FileResponse(
                file_path, 
                media_type=media_type,
                filename=filename
            )
    
    raise HTTPException(status_code=404, detail="文件未找到")
