import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from typing import Optional, Union, List, Dict
from pathlib import Path
from .watermark_detector import WatermarkDetector
from .simple_lama import SimpleLama
from ..utils import image_utils

class WatermarkRemover:
    """水印移除器类"""
    
    class SaveMode:
        """保存模式枚举"""
        OVERWRITE = "overwrite"  # 原位置覆盖
        OUTPUT_DIR = "output_dir"  # 保存到输出目录
        OUTPUT_DIR_ALL = "output_dir_all"  # 保存到输出目录(包括无水印图片)
        
    class DetectResultMode:
        """水印检测结果保存模式枚举"""
        NONE = "none"           # 不保存
        YOLO_TXT = "yolo_txt"   # 保存为YOLO格式txt文件
        JSON = "json"           # 保存为JSON文件
    
    def __init__(self):
        """初始化水印移除器"""
        self.detector = WatermarkDetector()
        self.inpainter = SimpleLama()
    
    def _get_save_format(self, image_path: str) -> Dict:
        """根据输入图片类型决定保存格式和参数
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            包含格式和参数的字典
        """
        ext = Path(image_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            return {
                'format': 'JPEG',
                'quality': 92,
                'optimize': True
            }
        elif ext == '.webp':
            return {
                'format': 'WEBP',
                'lossless': True
            }
        elif ext == '.png':
            return {
                'format': 'PNG',
                'optimize': True
            }
        else:
            return {
                'format': 'WEBP',
                'lossless': True
            }
    
    def _process_image(self, image_path: str) -> tuple[Image.Image, list]:
        """处理单张图片，返回处理结果和检测框
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            (处理后的图片, 检测到的水印框列表)
        """
        # 检测水印
        boxes = self.detector.detect(image_path)
        
        if not boxes:
            return Image.open(image_path), boxes
            
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        # 创建掩码
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for xmin, ymin, xmax, ymax in boxes:
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=1, thickness=-1)
            
        # 转换为PIL格式
        mask_pil = Image.fromarray(mask * 255).convert('L')
        image_pil = Image.open(image_path)
        
        # 执行修复
        result = self.inpainter(image_pil, mask_pil)
        return result, boxes

    def _save_detect_result(self, image_path: str, save_path: str, detect_mode: str) -> None:
        """保存检测结果
        
        Args:
            image_path: 输入图片路径
            save_path: 结果保存路径
            detect_mode: 检测结果保存模式
        """
        if detect_mode == self.DetectResultMode.YOLO_TXT:
            txt_path = str(Path(save_path).with_suffix('.yolo_txt'))
            self.detector.detect(
                image_path,
                output_type=WatermarkDetector.OutputType.YOLO_TXT,
                save_path=txt_path
            )
        elif detect_mode == self.DetectResultMode.JSON:
            output_dir = os.path.dirname(save_path)
            json_path = os.path.join(output_dir, 'detect_result.json')
            self.detector.detect(
                image_path,
                output_type=WatermarkDetector.OutputType.JSON,
                save_path=json_path
            )

    def _save_image(self, image: Image.Image, save_path: str, format_params: dict) -> None:
        """保存图片
        
        Args:
            image: 要保存的图片
            save_path: 保存路径
            format_params: 保存参数
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path, **format_params)

    def _get_output_path(self, image_path: str, output_dir: Optional[str], has_watermark: bool, save_mode: str) -> Optional[str]:
        """获取输出路径
        
        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            has_watermark: 是否有水印
            save_mode: 保存模式
            
        Returns:
            输出路径，如果不需要保存则返回None
        """
        if save_mode == self.SaveMode.OVERWRITE:
            return image_path if has_watermark else None
        elif save_mode in [self.SaveMode.OUTPUT_DIR, self.SaveMode.OUTPUT_DIR_ALL]:
            if not output_dir:
                raise ValueError("使用OUTPUT_DIR模式时必须指定output_dir")
            if has_watermark or save_mode == self.SaveMode.OUTPUT_DIR_ALL:
                rel_path = os.path.relpath(image_path, start=os.path.dirname(image_path))
                return os.path.join(output_dir, rel_path)
        return None

    def run_one(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_mode: str = SaveMode.OVERWRITE,
        detect_mode: str = DetectResultMode.NONE
    ) -> Optional[Image.Image]:
        """处理单张图片
        
        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            save_mode: 保存模式
            detect_mode: 检测结果保存模式
            
        Returns:
            如果不保存则返回处理后的PIL Image对象
        """
        # 处理图片
        result, boxes = self._process_image(image_path)
        has_watermark = bool(boxes)
        
        # 获取输出路径
        output_path = self._get_output_path(image_path, output_dir, has_watermark, save_mode)
        
        # 如果不需要保存，直接返回
        if not output_path:
            return result if has_watermark else None
            
        # 如果没有水印且需要保存，直接复制原文件
        if not has_watermark and save_mode == self.SaveMode.OUTPUT_DIR_ALL:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            copyfile(image_path, output_path)
            return None
            
        # 保存检测结果
        if detect_mode != self.DetectResultMode.NONE:
            self._save_detect_result(image_path, output_path, detect_mode)
            
        # 保存处理后的图片
        save_params = self._get_save_format(image_path)
        self._save_image(result, output_path, save_params)
        
        return None
    
    def run_images(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        save_mode: str = SaveMode.OVERWRITE,
        detect_mode: str = DetectResultMode.NONE
    ) -> None:
        """批量处理图片列表
        
        Args:
            image_paths: 图片路径列表
            output_dir: 输出目录
            save_mode: 保存模式
            detect_mode: 检测结果保存模式
        """
        for image_path in tqdm(image_paths, desc="处理图片"):
            try:
                self.run_one(
                    image_path,
                    output_dir=output_dir,
                    save_mode=save_mode,
                    detect_mode=detect_mode
                )
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")
    
    def run_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        recursive: bool = False,
        save_mode: str = SaveMode.OVERWRITE,
        detect_mode: str = DetectResultMode.NONE,
        extensions: Optional[List[str]] = None
    ) -> None:
        """处理整个目录
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            recursive: 是否递归处理子目录
            save_mode: 保存模式
            detect_mode: 检测结果保存模式
            extensions: 要处理的文件扩展名列表
        """
        # 获取所有图片路径
        image_paths = image_utils.get_image_paths(
            input_dir,
            recursive=recursive,
            extensions=extensions
        )
        
        # 如果使用输出目录模式，确保输出目录存在
        if save_mode == self.SaveMode.OUTPUT_DIR and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.run_images(
            image_paths,
            output_dir=output_dir,
            save_mode=save_mode,
            detect_mode=detect_mode
        )