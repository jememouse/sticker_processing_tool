"""
贴纸生成器 - 统一版本
支持 rembg 和 birefnet 两种抠图引擎，共用后处理和 PDF 生成逻辑。
"""
import cv2
import numpy as np
from PIL import Image
import io
import os
import warnings
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import magenta, white, cyan
from datetime import datetime
from typing import Literal

# 出血线距离裁切线的偏移量 (3mm)
BLEED_MARGIN_MM = 3
# 假设 72 DPI，1mm ≈ 2.83 像素
PX_PER_MM = 72 / 25.4

# Backend 类型定义
BackendType = Literal["rembg", "birefnet"]


class StickerGenerator:
    """
    统一的贴纸生成器类
    
    Args:
        backend: 抠图引擎，可选 "rembg"（默认）或 "birefnet"
    """
    
    def __init__(self, backend: BackendType = "rembg"):
        self.backend = backend
        self._model = None  # BiRefNet 模型（延迟加载）
        self._device = None  # BiRefNet 设备
        self._transform = None  # BiRefNet 预处理
        
        if backend == "birefnet":
            self._init_birefnet()
        elif backend == "rembg":
            print("初始化 Rembg 贴纸生成器...")
        else:
            raise ValueError(f"不支持的 backend: {backend}，可选: 'rembg', 'birefnet'")
    
    # ========== 引擎初始化 ==========
    
    def _init_birefnet(self):
        """初始化 BiRefNet 模型"""
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        
        # 设备选择
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        print(f"正在加载 BiRefNet 模型 (使用设备: {self._device})...")
        
        try:
            self._model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
            self._model.to(self._device)
            self._model.eval()
            print("模型加载完成。")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
        self._transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # ========== 抠图引擎 ==========
    
    def _remove_background_rembg(self, input_image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        使用 Rembg 抠图
        
        Returns:
            (原图 RGBA, 紧贴主体的 Mask)
        """
        from rembg import remove
        
        with open(input_image_path, 'rb') as f:
            input_data = f.read()
            
            # 读取原图
            original_pil = Image.open(io.BytesIO(input_data)).convert("RGBA")
            img_rgba = np.array(original_pil)
            
            # 使用 Rembg 抠图
            output_data = remove(input_data)
            rembg_pil = Image.open(io.BytesIO(output_data))
            rembg_val = np.array(rembg_pil)
        
        # 提取 Tight Mask
        tight_mask = rembg_val[:, :, 3]
        
        return img_rgba, tight_mask
    
    def _remove_background_birefnet(self, input_image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        使用 BiRefNet 抠图
        
        Returns:
            (原图 RGBA, 紧贴主体的 Mask)
        """
        import torch
        from torchvision import transforms
        
        # 读取原图
        original_pil = Image.open(input_image_path).convert("RGB")
        w, h = original_pil.size
        
        # 预处理
        input_tensor = self._transform(original_pil).unsqueeze(0).to(self._device)
        
        # 推理
        with torch.no_grad():
            preds = self._model(input_tensor)[-1].sigmoid().cpu()
        
        # 后处理 Mask
        pred_mask = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred_mask)
        mask_pil = pred_pil.resize((w, h), resample=Image.BILINEAR)
        
        # 转换为 RGBA
        original_pil = original_pil.convert("RGBA")
        img_rgba = np.array(original_pil)
        tight_mask = np.array(mask_pil)
        
        return img_rgba, tight_mask
    
    def _remove_background(self, input_image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """统一抠图接口，根据 backend 分发"""
        if self.backend == "birefnet":
            return self._remove_background_birefnet(input_image_path)
        else:
            return self._remove_background_rembg(input_image_path)
    
    # ========== 共用处理逻辑 ==========
    
    def process_image_to_data(self, input_image_path: str, outline_width: int = 15) -> dict:
        """
        核心处理：抠图 + 生成轮廓、出血线数据
        
        Args:
            input_image_path: 输入图片路径
            outline_width: 轮廓扩展宽度（像素）
            
        Returns:
            包含处理结果的字典
        """
        print(f"正在处理图片 ({self.backend}): {input_image_path}")
        
        # 1. 抠图
        img_rgba, tight_mask = self._remove_background(input_image_path)
        h, w = img_rgba.shape[:2]
        
        # 2. 生成平滑轮廓 Mask
        # 步骤 A: 扩展
        kernel_size = outline_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(tight_mask, kernel, iterations=1)
        
        # 步骤 B: 平滑
        blur_radius = outline_width
        if blur_radius % 2 == 0:
            blur_radius += 1
        blurred = cv2.GaussianBlur(dilated, (blur_radius, blur_radius), 0)
        
        # 步骤 C: 硬化 (得到白边 Mask)
        _, outline_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # 步骤 D: 计算出血线边界框 (裁切线 bounding box + 3mm)
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            
            bleed_px = int(BLEED_MARGIN_MM * PX_PER_MM)
            bleed_x = max(0, x - bleed_px)
            bleed_y = max(0, y - bleed_px)
            bleed_x2 = min(w, x + bw + bleed_px)
            bleed_y2 = min(h, y + bh + bleed_px)
            bleed_rect = (bleed_x, bleed_y, bleed_x2 - bleed_x, bleed_y2 - bleed_y)
        else:
            bleed_rect = (0, 0, w, h)
        
        return {
            "original_image_rgba": img_rgba,
            "tight_mask": tight_mask,
            "outline_mask": outline_mask,
            "bleed_rect": bleed_rect,
            "width": w,
            "height": h
        }
    
    def process_image(self, input_image_path: str, output_path: str, 
                      outline_width: int = 15, outline_color: tuple = (255, 255, 255),
                      shadow_opacity: float = 0.3) -> np.ndarray:
        """
        生成带阴影和白边的 PNG 图片（兼容旧接口）
        """
        data = self.process_image_to_data(input_image_path, outline_width)
        img_rgba = data["original_image_rgba"]
        tight_mask = data["tight_mask"]
        outline_mask = data["outline_mask"]
        h, w = data["height"], data["width"]
        
        # 准备带有透明度的原图
        subject_layer = img_rgba.copy()
        subject_layer[:, :, 3] = tight_mask
        
        # 背景层 (白色)
        result = np.full((h, w, 4), 255, dtype=np.uint8)
        
        # 叠加主体
        result = self._overlay_image(result, subject_layer)
        
        # 绘制裁切线 (红色描边)
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255, 255), 2)
        
        # 保存
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        print(f"贴纸预览图已生成: {output_path}")
        return outline_mask
    
    def generate_pdf(self, process_data: dict, output_pdf_path: str):
        """
        生成分层 PDF:
        - Layer 1: Artwork (裁切后的主体图像)
        - Layer 2: CutLine (矢量裁切线 - 洋红色)
        - Layer 3: BleedLine (出血线矩形 - 青色)
        """
        img_rgba = process_data["original_image_rgba"]
        tight_mask = process_data["tight_mask"]
        outline_mask = process_data["outline_mask"]
        bleed_rect = process_data["bleed_rect"]
        orig_w, orig_h = process_data["width"], process_data["height"]
        
        # 裁切到出血线范围
        bx, by, bw, bh = bleed_rect
        
        cropped_img = img_rgba[by:by+bh, bx:bx+bw].copy()
        cropped_tight_mask = tight_mask[by:by+bh, bx:bx+bw].copy()
        cropped_outline_mask = outline_mask[by:by+bh, bx:bx+bw].copy()
        
        page_w, page_h = bw, bh
        
        c = canvas.Canvas(output_pdf_path, pagesize=(page_w, page_h))
        
        # Layer 1: Artwork
        masked_img = cropped_img.copy()
        masked_img[:, :, 3] = cropped_tight_mask
        
        temp_img_path = f"/tmp/temp_artwork_{self.backend}_cropped.png"
        cv2.imwrite(temp_img_path, cv2.cvtColor(masked_img, cv2.COLOR_RGBA2BGRA))
        
        c.drawImage(temp_img_path, 0, 0, width=page_w, height=page_h, mask='auto', preserveAspectRatio=True)
        
        # Layer 2: CutLine (洋红色)
        self._draw_contours_as_path(c, cropped_outline_mask, fill=False, stroke=True,
                                     color=magenta, stroke_width=0.5*mm, h_page=page_h, smooth_factor=0.0005)
        
        # Layer 3: BleedLine (青色矩形)
        c.setStrokeColor(cyan)
        c.setLineWidth(0.3*mm)
        c.rect(0, 0, page_w, page_h, stroke=1, fill=0)
        
        c.showPage()
        c.save()
        
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        
        print(f"分层 PDF 已生成 (裁切至出血线): {output_pdf_path}")
        print(f"  原始尺寸: {orig_w}x{orig_h} -> 裁切后: {page_w}x{page_h}")
    
    def _draw_contours_as_path(self, c, mask, fill=False, stroke=True, 
                                color=None, stroke_width=1, h_page=0, smooth_factor=0.0005):
        """将 OpenCV 轮廓转换为平滑的贝塞尔曲线"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if color:
            if fill: c.setFillColor(color)
            if stroke: c.setStrokeColor(color)
        
        if stroke: c.setLineWidth(stroke_width)
        
        p = c.beginPath()
        
        for contour in contours:
            epsilon = smooth_factor * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            pts = []
            for pt in approx_contour:
                x, y_cv = pt[0]
                pts.append((x, h_page - y_cv))
            
            if len(pts) < 3: continue
            
            n = len(pts)
            
            p_last = pts[n-1]
            p_curr = pts[0]
            mid_start_x = (p_last[0] + p_curr[0]) / 2.0
            mid_start_y = (p_last[1] + p_curr[1]) / 2.0
            
            p.moveTo(mid_start_x, mid_start_y)
            
            for i in range(n):
                ctrl_pt = pts[i]
                next_pt = pts[(i+1)%n]
                
                mid_end_x = (ctrl_pt[0] + next_pt[0]) / 2.0
                mid_end_y = (ctrl_pt[1] + next_pt[1]) / 2.0
                
                p0_x, p0_y = mid_start_x, mid_start_y
                
                cp1_x = p0_x + (2/3) * (ctrl_pt[0] - p0_x)
                cp1_y = p0_y + (2/3) * (ctrl_pt[1] - p0_y)
                
                cp2_x = mid_end_x + (2/3) * (ctrl_pt[0] - mid_end_x)
                cp2_y = mid_end_y + (2/3) * (ctrl_pt[1] - mid_end_y)
                
                p.curveTo(cp1_x, cp1_y, cp2_x, cp2_y, mid_end_x, mid_end_y)
                
                mid_start_x, mid_start_y = mid_end_x, mid_end_y
            
            p.close()
        
        c.drawPath(p, fill=fill, stroke=stroke)
    
    def generate_svg_cutline(self, mask: np.ndarray, output_svg_path: str, smooth_factor: float = 0.0005):
        """将 Mask 转换为 SVG 路径（用于切割机/刀模）"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(output_svg_path, "w+") as f:
            h, w = mask.shape
            f.write(f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">')
            f.write('<path d="')
            
            for contour in contours:
                epsilon = smooth_factor * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx_contour) < 3: continue
                
                for i, point in enumerate(approx_contour):
                    x, y = point[0]
                    if i == 0:
                        f.write(f"M {x} {y} ")
                    else:
                        f.write(f"L {x} {y} ")
                f.write("Z ")
            
            f.write('" stroke="red" stroke-width="1" fill="none" />')
            f.write("</svg>")
            print(f"刀模线 SVG 已生成: {output_svg_path}")
    
    def _overlay_image(self, background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        """Alpha 混合辅助函数"""
        alpha_background = background[:,:,3] / 255.0
        alpha_foreground = foreground[:,:,3] / 255.0
        
        for color in range(0, 3):
            background[:,:,color] = (
                alpha_foreground * foreground[:,:,color] +
                alpha_background * background[:,:,color] * (1 - alpha_foreground)
            )
        
        background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
        return background


# --- 使用示例 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="贴纸生成器")
    parser.add_argument("--backend", type=str, default="rembg", choices=["rembg", "birefnet"],
                        help="抠图引擎 (默认: rembg)")
    parser.add_argument("--input", type=str, default="sticker/111.jpg",
                        help="输入图片路径")
    parser.add_argument("--outline-width", type=int, default=20,
                        help="轮廓宽度 (默认: 20)")
    args = parser.parse_args()
    
    generator = StickerGenerator(backend=args.backend)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "Outfile")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    input_path = args.input
    
    if os.path.exists(input_path):
        process_data = generator.process_image_to_data(input_path, outline_width=args.outline_width)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pdf_path = os.path.join(output_dir, f"{input_name}_sticker_{args.backend}_{timestamp}.pdf")
        generator.generate_pdf(process_data, output_pdf_path)
        
        print("Done!")
    else:
        print(f"Error: Input file not found: {input_path}")
