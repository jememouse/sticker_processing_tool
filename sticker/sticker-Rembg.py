import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io
import os
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import magenta, white
from reportlab.pdfbase.pdfdoc import PDFCatalog
from datetime import datetime

class StickerGenerator:
    def __init__(self):
        pass

    def process_image_to_data(self, input_image_path, outline_width=15):
        """
        核心处理：不直接保存文件，而是返回处理后的数据，用于后续生成 PDF 或 PNG
        """
        # 1. 读取并抠图 (使用 Rembg 识别主体)
        with open(input_image_path, 'rb') as i:
            input_data = i.read()
            
            # 视觉层: 使用原图 (保留背景)
            original_pil = Image.open(io.BytesIO(input_data)).convert("RGBA")
            img_rgba = np.array(original_pil)
            
            # 遮罩层: 使用 Rembg 抠图结果来提取轮廓
            output_data = remove(input_data)
            rembg_pil = Image.open(io.BytesIO(output_data))
            rembg_val = np.array(rembg_pil)
        
        # 2. 生成平滑轮廓 Mask (基于抠图后的 Alpha 通道)
        alpha = rembg_val[:, :, 3]
        
        # 步骤 A: 扩展
        kernel_size = outline_width * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(alpha, kernel, iterations=1)
        
        # 步骤 B: 平滑
        blur_radius = outline_width
        if blur_radius % 2 == 0: blur_radius += 1
        blurred = cv2.GaussianBlur(dilated, (blur_radius, blur_radius), 0)
        
        # 步骤 C: 硬化 (得到白边 Mask)
        _, outline_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # 返回关键数据字典
        return {
            "original_image_rgba": img_rgba, # 抠好的原图
            "outline_mask": outline_mask,   # 白边/刀线 Mask
            "width": img_rgba.shape[1],
            "height": img_rgba.shape[0]
        }

    def process_image(self, input_image_path, output_path, outline_width=15, outline_color=(255, 255, 255), shadow_opacity=0.3):
        """
        保持原有接口兼容：生成带阴影和白边的 PNG 图片
        """
        data = self.process_image_to_data(input_image_path, outline_width)
        img_rgba = data["original_image_rgba"]
        outline_mask = data["outline_mask"]
        h, w = data["height"], data["width"]
        
        # 3. 合成图像 (PNG 专用逻辑: 加阴影, 合成层)
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 绘制投影
        shadow_offset = 10
        shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_layer[outline_mask > 0] = [0, 0, 0, int(255 * shadow_opacity)]
        shadow_layer = cv2.GaussianBlur(shadow_layer, (15, 15), 0)
        M = np.float32([[1, 0, shadow_offset], [0, 1, shadow_offset]])
        shadow_layer = cv2.warpAffine(shadow_layer, M, (w, h))
        
        # 绘制轮廓层
        outline_layer = np.zeros((h, w, 4), dtype=np.uint8)
        outline_layer[outline_mask > 0] = [outline_color[0], outline_color[1], outline_color[2], 255]
        
        # 叠加
        result = self._overlay_image(result, shadow_layer)
        result = self._overlay_image(result, outline_layer)
        result = self._overlay_image(result, img_rgba)
        
        # 保存
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        print(f"贴纸已生成: {output_path}")
        return outline_mask

    def generate_pdf(self, process_data, output_pdf_path):
        """
        生成分层 PDF: 
        Layer 1: White Mask (矢量白底)
        Layer 2: Artwork (原图)
        Layer 3: CutLine (矢量刀线)
        """
        img_rgba = process_data["original_image_rgba"]
        outline_mask = process_data["outline_mask"]
        w_px, h_px = process_data["width"], process_data["height"]
        
        # 转换尺寸 (假设 72 DPI，1 px = 1 point)
        # 实际生产可能需要更高 DPI，这里保持 1:1 像素映射到 PDF点
        c = canvas.Canvas(output_pdf_path, pagesize=(w_px, h_px))
        
        # 1. 定义 OCG 层
        # 注意: ReportLab 的 OCG API 比较底层，这里使用 beginOCG/endOCG
        
        # --- Layer 1: White Border (作为矢量填充) ---
        # c.beginOCG("White Border", on=1)
        self._draw_contours_as_path(c, outline_mask, fill=True, stroke=False, color=white, h_page=h_px)
        # c.endOCG()
        
        # --- Layer 2: Artwork (位图) ---
        # c.beginOCG("Artwork", on=1)
        # 将 numpy array 转为临时 PNG 以供 ReportLab 使用 (保持透明度)
        temp_img_path = "/tmp/temp_artwork.png"
        cv2.imwrite(temp_img_path, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
        # drawImage 参数: image, x, y, width, height, mask
        # 坐标系: PDF 原点在左下角，OpenCV 在左上角。需要翻转或计算 Y
        c.drawImage(temp_img_path, 0, 0, width=w_px, height=h_px, mask='auto', preserveAspectRatio=True)
        # c.endOCG()
        
        # --- Layer 3: CutLine (矢量描边) ---
        # c.beginOCG("CutLine", on=1)
        # 描边通常用洋红色 (Magenta)
        self._draw_contours_as_path(c, outline_mask, fill=False, stroke=True, color=magenta, stroke_width=0.3*mm, h_page=h_px)
        # c.endOCG()
        
        c.showPage()
        c.save()
        
        # 清理临时文件
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
        print(f"分层 PDF 已生成: {output_pdf_path}")

    def _draw_contours_as_path(self, c, mask, fill=False, stroke=True, color=None, stroke_width=1, h_page=0, smooth_factor=0.0005):
        """
        将 OpenCV 轮廓转换为 PDF 路径并在 Canvas 上绘制
        注意坐标转换: OpenCV y 向下，PDF y 向上
        smooth_factor: approxPolyDP 的 epsilon 参数系数 (epsilon = arcLength * smooth_factor)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if color:
            if fill: c.setFillColor(color)
            if stroke: c.setStrokeColor(color)
        
        if stroke: c.setLineWidth(stroke_width)
        
        p = c.beginPath()
        
        for contour in contours:
            # 平滑处理
            epsilon = smooth_factor * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx_contour) < 3: continue # 忽略过小的轮廓

            # 起点
            start_pt = approx_contour[0][0]
            # 翻转 Y 坐标: y_pdf = h_page - y_cv
            p.moveTo(start_pt[0], h_page - start_pt[1])
            
            for i in range(1, len(approx_contour)):
                pt = approx_contour[i][0]
                p.lineTo(pt[0], h_page - pt[1])
            
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
    
    # 获取处理后的中间数据
    process_data = generator.process_image_to_data(input_path, outline_width=20)
    
    # 生成 PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_pdf_path = os.path.join(output_dir, f"sticker_layers_{timestamp}.pdf")
    generator.generate_pdf(process_data, output_pdf_path)
    
    # 也顺便生成一个 PNG 看看 (复用 process_image 的调用方式，或者我们前面已经验证过了)
    # 这里我们主要演示 PDF