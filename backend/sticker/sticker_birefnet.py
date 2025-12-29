"""
贴纸生成器 - BiRefNet 版本
使用 BiRefNet 模型进行 AI 抠图

⚠️ DEPRECATED: 此文件已弃用，请使用 sticker_generator.py 代替。
使用示例: StickerGenerator(backend="birefnet")
"""
import warnings
warnings.warn(
    "sticker_birefnet.py 已弃用，请使用 sticker_generator.py 代替。"
    "示例: from sticker.sticker_generator import StickerGenerator; StickerGenerator(backend='birefnet')",
    DeprecationWarning,
    stacklevel=2
)
import cv2
import numpy as np
from PIL import Image
import io
import os
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import magenta, white, cyan
from datetime import datetime

# 出血线距离裁切线的偏移量 (3mm)
BLEED_MARGIN_MM = 3
# 假设 72 DPI，1mm ≈ 2.83 像素
PX_PER_MM = 72 / 25.4

class StickerGenerator:
    def __init__(self):
        self.device = self._get_device()
        self.model = self._load_model()
        self.transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_model(self):
        print(f"正在加载 BiRefNet 模型 (使用设备: {self.device})...")
        try:
            model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
            model.to(self.device)
            model.eval()
            print("模型加载完成。")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

    def process_image_to_data(self, input_image_path, outline_width=15):
        """
        核心处理：使用 BiRefNet 抠图，返回处理后的数据
        """
        # 1. 读取并抠图 (使用 BiRefNet)
        print(f"正在处理图片: {input_image_path}")
        
        # 读取原图
        original_pil = Image.open(input_image_path).convert("RGB")
        w, h = original_pil.size
        
        # 预处理
        input_tensor = self.transform_image(original_pil).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        # 后处理 Mask
        pred_mask = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred_mask)
        mask_pil = pred_pil.resize((w, h), resample=Image.BILINEAR)
        
        # 应用 Mask 到原图
        original_pil = original_pil.convert("RGBA")
        img_rgba = np.array(original_pil)
        mask_np = np.array(mask_pil)
        
        # 将 Mask 应用为 Alpha 通道
        # [修改] 不再修改 Alpha 通道，保留完整原图数据 (用于后续反向遮罩)
        # img_rgba[:, :, 3] = mask_np 
        
        # 2. 生成平滑轮廓 Mask (基于抠图后的 Alpha 通道)
        # 注意: BiRefNet 输出的 mask 已经是灰度图 (0-255)，可以直接作为 alpha 使用
        # 但为了生成轮廓，我们可能需要二值化一下，或者直接使用 alpha
        
        # 步骤 A: 扩展
        kernel_size = outline_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        
        # 步骤 B: 平滑
        # 步骤 B: 平滑
        blur_radius = outline_width
        if blur_radius % 2 == 0: blur_radius += 1
        # [优化] 增加高斯模糊的强度可以让轮廓更圆润，减少微小锯齿
        blurred = cv2.GaussianBlur(dilated, (blur_radius, blur_radius), 0)
        
        # 步骤 C: 硬化 (得到白边 Mask)
        _, outline_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # 步骤 D: 计算出血线边界框 (裁切线 bounding box + 3mm)
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 合并所有轮廓的 bounding box
            all_points = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_points)
            
            # 出血线外扩 3mm (像素)
            bleed_px = int(BLEED_MARGIN_MM * PX_PER_MM)
            bleed_x = max(0, x - bleed_px)
            bleed_y = max(0, y - bleed_px)
            bleed_x2 = min(w, x + bw + bleed_px)
            bleed_y2 = min(h, y + bh + bleed_px)
            bleed_rect = (bleed_x, bleed_y, bleed_x2 - bleed_x, bleed_y2 - bleed_y)
        else:
            bleed_rect = (0, 0, w, h)
        
        # 返回关键数据字典
        return {
            "original_image_rgba": img_rgba,  # 完整原图
            "tight_mask": mask_np,            # 紧贴主体的 Mask
            "outline_mask": outline_mask,     # 扩充后的 Mask (用于裁切线)
            "bleed_rect": bleed_rect,         # 出血线边界框 (x, y, w, h)
            "width": w,
            "height": h
        }

    def process_image(self, input_image_path, output_path, outline_width=15, outline_color=(255, 255, 255), shadow_opacity=0.3):
        """
        保持原有接口兼容：生成带阴影和白边的 PNG 图片
        """
        data = self.process_image_to_data(input_image_path, outline_width)
        img_rgba = data["original_image_rgba"]
        tight_mask = data["tight_mask"] # [新增]
        outline_mask = data["outline_mask"]
        h, w = data["height"], data["width"]
        
        # 3. 合成图像 (PNG 预览: 白边贴纸效果)
        # 最终效果: 主体(Tight) + 白边 + 裁切线
        
        # 3.1 准备带有透明度的原图 (应用 Tight Mask)
        # 为了不破坏 original_image_rgba，我们创建一个副本
        subject_layer = img_rgba.copy()
        subject_layer[:, :, 3] = tight_mask # 将 tight_mask 应用为 Alpha

        # 3.2 背景层 (白色)
        result = np.full((h, w, 4), 255, dtype=np.uint8) # White background
        
        # 3.3 叠加主体
        result = self._overlay_image(result, subject_layer)
        
        # 3.4 绘制裁切线 (红色/洋红色描边)
        contours, _ = cv2.findContours(outline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255, 255), 2) # Red line
        
        # 保存
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        print(f"贴纸预览图已生成: {output_path}")
        return outline_mask

    def generate_pdf(self, process_data, output_pdf_path):
        """
        生成分层 PDF: 
        - 图像裁切到出血线范围 (矩形)
        - Layer 1: Artwork (裁切后的主体图像)
        - Layer 2: CutLine (矢量裁切线 - 洋红色)
        - Layer 3: BleedLine (出血线矩形 - 青色)
        """
        img_rgba = process_data["original_image_rgba"]
        tight_mask = process_data["tight_mask"]
        outline_mask = process_data["outline_mask"]
        bleed_rect = process_data["bleed_rect"]  # (x, y, w, h)
        orig_w, orig_h = process_data["width"], process_data["height"]
        
        # 裁切到出血线范围
        bx, by, bw, bh = bleed_rect
        
        # 裁切图像和 mask
        cropped_img = img_rgba[by:by+bh, bx:bx+bw].copy()
        cropped_tight_mask = tight_mask[by:by+bh, bx:bx+bw].copy()
        cropped_outline_mask = outline_mask[by:by+bh, bx:bx+bw].copy()
        
        # PDF 页面尺寸为裁切后的尺寸
        page_w, page_h = bw, bh
        
        c = canvas.Canvas(output_pdf_path, pagesize=(page_w, page_h))
        
        # --- Layer 1: Artwork (裁切后的主体图像) ---
        masked_img = cropped_img.copy()
        masked_img[:, :, 3] = cropped_tight_mask
        
        temp_img_path = "/tmp/temp_artwork_cropped.png"
        cv2.imwrite(temp_img_path, cv2.cvtColor(masked_img, cv2.COLOR_RGBA2BGRA))
        
        c.drawImage(temp_img_path, 0, 0, width=page_w, height=page_h, mask='auto', preserveAspectRatio=True)
        
        # --- Layer 2: CutLine (矢量裁切线 - 洋红色) ---
        self._draw_contours_as_path(c, cropped_outline_mask, fill=False, stroke=True, 
                                     color=magenta, stroke_width=0.5*mm, h_page=page_h, smooth_factor=0.0005)
        
        # --- Layer 3: BleedLine (出血线矩形 - 青色) ---
        # 出血线是整个页面的边界矩形
        c.setStrokeColor(cyan)
        c.setLineWidth(0.3*mm)
        c.rect(0, 0, page_w, page_h, stroke=1, fill=0)
        
        c.showPage()
        c.save()
        
        # 清理临时文件
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        print(f"分层 PDF 已生成 (裁切至出血线): {output_pdf_path}")
        print(f"  原始尺寸: {orig_w}x{orig_h} -> 裁切后: {page_w}x{page_h}")

    def _draw_contours_as_path(self, c, mask, fill=False, stroke=True, color=None, stroke_width=1, h_page=0, smooth_factor=0.0005):
        """
        [优化] 将 OpenCV 轮廓转换为 平滑的贝塞尔曲线 (Quadratic Bezier) 并在 Canvas 上绘制
        使用 Midpoint Smoothing 算法 (类似 Chaikin's Corner Cutting)，消除转角处的过冲和打结。
        smooth_factor: 控制点的稀疏程度。0.0005 可以在保持细节的同时，通过 Midpoint 算法实现非常圆润的转角。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if color:
            if fill: c.setFillColor(color)
            if stroke: c.setStrokeColor(color)
        
        if stroke: c.setLineWidth(stroke_width)
        
        p = c.beginPath()
        
        for contour in contours:
            # 1. 预处理：适当稀疏化点，去除微小抖动，但保留大致形状
            epsilon = smooth_factor * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # 提取点集 (x, y_pdf)
            # 注意: OpenCV y 向下，PDF y 向上。转换: y_pdf = h_page - y_cv
            pts = []
            for pt in approx_contour:
                x, y_cv = pt[0]
                pts.append((x, h_page - y_cv))
            
            if len(pts) < 3: continue

            # 2. 绘制平滑曲线 (Quadratic Bezier / Corner Cutting)
            # 算法: 连接中点。
            # 对于角点 P1，我们从 P0-P1 的中点 M1 画一条二次贝塞尔曲线到 P1-P2 的中点 M2，控制点为 P1。
            # 这保证了曲线永远在凸包内，绝对不会产生“打结”或奇异变形。
            
            # 为了处理闭合轮廓，我们需要环绕处理
            # 构造点集: 
            # 我们需要的段数等于点数 n。
            # 第 i 段曲线: 起点 = Mid(P[i-1], P[i]), 终点 = Mid(P[i], P[i+1]), 控制点 = P[i]
            
            n = len(pts)
            
            # 计算第一段的起点 (即 P[n-1] 和 P[0] 的中点)
            p_last = pts[n-1]
            p_curr = pts[0]
            mid_start_x = (p_last[0] + p_curr[0]) / 2.0
            mid_start_y = (p_last[1] + p_curr[1]) / 2.0
            
            p.moveTo(mid_start_x, mid_start_y)
            
            for i in range(n):
                # 当前控制点
                ctrl_pt = pts[i]
                # 下一个点
                next_pt = pts[(i+1)%n]
                
                # 终点 (当前点与下一点的中点)
                mid_end_x = (ctrl_pt[0] + next_pt[0]) / 2.0
                mid_end_y = (ctrl_pt[1] + next_pt[1]) / 2.0
                
                # ReportLab 只有 curveTo (三次贝塞尔)。我们需要将 二次 升阶为 三次。
                # Quadratic(P0, P1, P2) -> Cubic(P0, CP1, CP2, P2)
                # CP1 = P0 + 2/3 * (P1 - P0)
                # CP2 = P2 + 2/3 * (P1 - P2)
                
                # 当前段起点 P0 即原本的 mid_start
                # P1 即 ctrl_pt
                # P2 即 mid_end
                
                # P0 (current pen position)
                p0_x, p0_y = mid_start_x, mid_start_y # 逻辑上的起点
                
                cp1_x = p0_x + (2/3) * (ctrl_pt[0] - p0_x)
                cp1_y = p0_y + (2/3) * (ctrl_pt[1] - p0_y)
                
                cp2_x = mid_end_x + (2/3) * (ctrl_pt[0] - mid_end_x)
                cp2_y = mid_end_y + (2/3) * (ctrl_pt[1] - mid_end_y)
                
                p.curveTo(cp1_x, cp1_y, cp2_x, cp2_y, mid_end_x, mid_end_y)
                
                # 更新下一段的起点
                mid_start_x, mid_start_y = mid_end_x, mid_end_y
            
            p.close()
        
        c.drawPath(p, fill=fill, stroke=stroke)



    def generate_svg_cutline(self, mask, output_svg_path, smooth_factor=0.0005):
        """
        将 Mask 转换为 SVG 路径 (用于切割机/刀模)
        smooth_factor: approxPolyDP 的 epsilon 参数系数
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        with open(output_svg_path, "w+") as f:
            h, w = mask.shape
            f.write(f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">')
            f.write(f'<path d="')
            
            for contour in contours:
                # 平滑处理
                epsilon = smooth_factor * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx_contour) < 3: continue

                for i, point in enumerate(approx_contour):
                    x, y = point[0]
                    if i == 0:
                        f.write(f"M {x} {y} ")
                    else:
                        f.write(f"L {x} {y} ")
                f.write("Z ") # 闭合路径
            
            f.write(f'" stroke="red" stroke-width="1" fill="none" />')
            f.write("</svg>")
            print(f"刀模线 SVG 已生成: {output_svg_path}")

    def _overlay_image(self, background, foreground):
        """
        简单的 Alpha 混合辅助函数
        """
        alpha_background = background[:,:,3] / 255.0
        alpha_foreground = foreground[:,:,3] / 255.0

        for color in range(0, 3):
            background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
                                    alpha_background * background[:,:,color] * (1 - alpha_foreground)

        background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
        return background

# --- 使用示例 ---
if __name__ == "__main__":
    generator = StickerGenerator()
    
    # 假设你有一张名为 product.jpg 的商品图
    # 1. 生成贴纸 PNG
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 定义输出目录
    output_dir = os.path.join(base_dir, "Outfile")
    
    # 如果目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 1. 生成 PNG (包含阴影效果)
    # output_png_path = os.path.join(output_dir, "output_sticker.png")
    # mask = generator.process_image("/Users/wang/code-project/marketing_material_files_and_information_processing_tools/sticker/SCR-20241128-mbke.jpeg", output_png_path, outline_width=20)
    
    # 2. 生成 PDF (多图层：白底、图案、切线)
    input_path = "sticker/111.jpg"
    
    if os.path.exists(input_path):
        # 获取处理后的中间数据
        # 第一次运行会下载模型，可能需要一点时间
        process_data = generator.process_image_to_data(input_path, outline_width=20)
        
        # 生成 PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pdf_path = os.path.join(output_dir, f"{input_name}_sticker_{timestamp}.pdf")
        generator.generate_pdf(process_data, output_pdf_path)
        
        print("Done!")
    else:
        print(f"Error: Input file not found: {input_path}")