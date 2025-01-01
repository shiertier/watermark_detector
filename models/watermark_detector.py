import os
from ultralytics import YOLO
import cv2
from typing import List, Tuple, Optional, Dict, Union
from tqdm import tqdm
from pathlib import Path
from ..utils import download_utils
from ..utils import image_utils
import json
import numpy as np

class WatermarkDetector:
    """水印检测模型类"""
    
    MODEL_FILENAME = "watermark_detect_2024.pt"
    DEFAULT_MODEL_URL = "https://huggingface.co/shiertier/watermark_detect/resolve/main/best.pt"

    class OutputType:
        """输出类型枚举"""
        BOXES = "boxes"  # 仅返回边界框列表
        YOLO_TXT = "yolo_txt"  # 保存YOLO格式的txt文件
        JSON = "json"  # 保存为JSON文件
        SHOW = "show"  # 显示或保存可视化结果

    def __init__(self, model_path: str = None):
        """初始化水印检测器
        
        Args:
            model_path: 模型路径,如果为None则自动下载默认模型
        """
        if model_path is None:
            model_path = os.environ.get("WATERMARK_MODEL")
            if not model_path or not os.path.exists(model_path):
                model_path = download_utils.download_model(
                    self.DEFAULT_MODEL_URL, 
                    filename=self.MODEL_FILENAME
                )
                
        self.yolo_model = YOLO(model_path)

    def detect_one(
        self, 
        image_path: str,
        output_type: str = "boxes",
        save_path: Optional[str] = None,
        show_window: bool = False,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2
    ) -> Union[List[Tuple[int, int, int, int]], np.ndarray]:
        """检测图像中的水印位置
        
        Args:
            image_path: 输入图像路径
            output_type: 输出类型，可选值：
                - "boxes": 返回边界框列表
                - "yolo_txt": 保存YOLO格式的txt文件
                - "show": 返回可视化结果图像
            save_path: 保存路径（用于yolo_txt和show类型）
            show_window: 是否显示窗口（仅用于show类型）
            box_color: 边界框颜色（仅用于show类型）
            box_thickness: 边界框粗细（仅用于show类型）
            
        Returns:
            根据output_type返回不同类型的结果
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        detect_results = self.yolo_model(image)
        boxes = []
        
        for result in detect_results:
            boxes.extend([
                tuple(map(int, box.xyxy[0].tolist()))
                for box in result.boxes
            ])
        
        if output_type == self.OutputType.BOXES:
            return boxes
        elif output_type == self.OutputType.YOLO_TXT:
            img_height, img_width = image.shape[:2]
            txt_path = save_path or str(Path(image_path).with_suffix('.txt'))
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                for xmin, ymin, xmax, ymax in boxes:
                    x_center = (xmin + xmax) / (2 * img_width)
                    y_center = (ymin + ymax) / (2 * img_height)
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            return boxes
        elif output_type == self.OutputType.SHOW:
            result_image = image.copy()
            for xmin, ymin, xmax, ymax in boxes:
                cv2.rectangle(
                    result_image,
                    (xmin, ymin),
                    (xmax, ymax),
                    box_color,
                    box_thickness
                )
            
            if save_path:
                cv2.imwrite(save_path, result_image)
            
            if show_window:
                window_name = "检测结果"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return result_image
        else:
            raise ValueError(f"不支持的输出类型: {output_type}")

    def detect(
        self,
        input_path: Union[str, List[str]],
        output_type: str = "boxes",
        save_path: Optional[str] = None,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        show_window: bool = False,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2
    ) -> Union[
        List[Tuple[int, int, int, int]],
        Dict[str, List[List[int]]],
        np.ndarray,
        None
    ]:
        """统一的检测接口
        
        Args:
            input_path: 输入路径，可以是单个图片路径、图片路径列表或目录路径
            output_type: 输出类型，可选值：
                - "boxes": 返回边界框列表（仅用于单图片）
                - "yolo_txt": 保存YOLO格式的txt文件
                - "json": 保存为JSON文件（需要指定save_path）
                - "show": 返回可视化结果（仅用于单图片）
            save_path: 保存路径
            recursive: 处理目录时是否递归
            extensions: 处理目录时的文件扩展名列表
            show_window: 是否显示窗口（仅用于show类型）
            box_color: 边界框颜色（仅用于show类型）
            box_thickness: 边界框粗细（仅用于show类型）
        
        Returns:
            根据输入和output_type返回不同类型的结果
        """
        # 处理单个图片
        if isinstance(input_path, str) and os.path.isfile(input_path):
            return self.detect_one(
                input_path,
                output_type=output_type,
                save_path=save_path,
                show_window=show_window,
                box_color=box_color,
                box_thickness=box_thickness
            )
        
        # 获取图片路径列表
        if isinstance(input_path, str) and os.path.isdir(input_path):
            image_paths = image_utils.get_image_paths(input_path, recursive, extensions)
        elif isinstance(input_path, list):
            image_paths = input_path
        else:
            raise ValueError("input_path必须是图片路径、图片路径列表或目录路径")
        
        # 处理多个图片
        if output_type == self.OutputType.YOLO_TXT:
            with tqdm(total=len(image_paths), desc="检测水印") as pbar:
                for image_path in image_paths:
                    try:
                        self.detect_one(image_path, output_type="yolo_txt")
                    except Exception as e:
                        print(f"处理 {image_path} 时出错: {e}")
                    pbar.update(1)
        elif output_type == self.OutputType.JSON:
            if not save_path:
                raise ValueError("使用JSON输出类型时必须指定save_path")
                
            results = {}
            with tqdm(total=len(image_paths), desc="检测水印") as pbar:
                for image_path in image_paths:
                    try:
                        boxes = self.detect_one(image_path)
                        results[image_path] = [list(box) for box in boxes]
                    except Exception as e:
                        print(f"处理 {image_path} 时出错: {e}")
                    pbar.update(1)
                    
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持对多个图片使用输出类型: {output_type}")

    def detect_images_and_save_txts(
        self,
        image_paths: List[str]
    ) -> None:
        """检测多个图片的水印并保存为YOLO格式的txt文件
        
        Args:
            image_paths: 图片路径列表
        """
        with tqdm(total=len(image_paths), desc="检测水印") as pbar:
            for image_path in image_paths:
                try:
                    self.detect_one_and_save(image_path)
                except Exception as e:
                    print(f"处理 {image_path} 时出错: {e}")
                pbar.update(1)

    def detect_images_and_save_json(
        self,
        image_paths: List[str],
        output_path: str
    ) -> None:
        """检测多个图片的水印并保存为JSON文件
        
        Args:
            image_paths: 图片路径列表
            output_path: 输出JSON文件路径
        """
        results: Dict[str, List[List[int]]] = {}
        
        with tqdm(total=len(image_paths), desc="检测水印") as pbar:
            for image_path in image_paths:
                try:
                    boxes = self.detect_one(image_path)
                    results[image_path] = [list(box) for box in boxes]
                except Exception as e:
                    print(f"处理 {image_path} 时出错: {e}")
                pbar.update(1)
                
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def detect_directory_and_save_txts(
        self,
        directory: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None
    ) -> None:
        """检测目录中所有图片的水印，并保存为YOLO格式的txt文件
        
        Args:
            directory: 图片目录路径
            recursive: 是否递归处理子目录
            extensions: 要处理的图片扩展名列表,默认为['.jpg', '.jpeg', '.png', '.webp']
        """
        image_paths = image_utils.get_image_paths(directory, recursive, extensions)
        self.detect_images_and_save_txts(image_paths)

    def detect_directory_and_save_json(
        self,
        directory: str,
        output_path: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None
    ) -> None:
        """检测目录中所有图片的水印，并将结果保存到JSON文件
        
        Args:
            directory: 图片目录路径
            output_path: 输出JSON文件路径
            recursive: 是否递归处理子目录
            extensions: 要处理的图片扩展名列表,默认为['.jpg', '.jpeg', '.png', '.webp']
        """
        image_paths = image_utils.get_image_paths(directory, recursive, extensions)
        self.detect_images_and_save_json(image_paths, output_path)

    def detect_one_and_show(
        self, 
        image_path: str,
        save_path: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 0),  # 默认使用绿色
        thickness: int = 2,
        show_window: bool = True
    ) -> np.ndarray:
        """检测水印并在图像上绘制边界框
        
        Args:
            image_path: 输入图像路径
            save_path: 保存结果图像的路径,如果为None则不保存
            color: BGR格式的边界框颜色,默认为绿色
            thickness: 边界框线条粗细
            show_window: 是否显示结果窗口
            
        Returns:
            绘制了检测结果的图像数组
        """

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 检测水印
        boxes = self.detect(image_path)
        
        # 在图像上绘制边界框
        result_image = image.copy()
        for xmin, ymin, xmax, ymax in boxes:
            # 绘制矩形
            cv2.rectangle(
                result_image, 
                (xmin, ymin), 
                (xmax, ymax), 
                color, 
                thickness
            )
        
        # 保存结果图像
        if save_path:
            cv2.imwrite(save_path, result_image)
        
        # 显示结果
        if show_window:
            window_name = "检测结果"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_image 