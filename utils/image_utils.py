import numpy as np
import cv2
from PIL import Image
from typing import Union
from pathlib import Path
from typing import List, Optional

def get_image_paths(
    directory: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """获取目录中的所有图片路径
    
    Args:
        directory: 图片目录路径
        recursive: 是否递归处理子目录
        extensions: 要处理的图片扩展名列表
        
    Returns:
        图片路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
    extensions = {ext.lower() for ext in extensions}
    
    def is_image(path: str) -> bool:
        return Path(path).suffix.lower() in extensions
        
    image_paths = []
    base_path = Path(directory)
    
    if recursive:
        for path in base_path.rglob('*'):
            if path.is_file() and is_image(str(path)):
                image_paths.append(str(path))
    else:
        for path in base_path.iterdir():
            if path.is_file() and is_image(str(path)):
                image_paths.append(str(path))
                
    return image_paths

def convert_to_numpy(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """将PIL Image或numpy数组转换为标准格式的numpy数组
    
    Args:
        image: 输入图像,可以是PIL Image或numpy数组
        
    Returns:
        标准化后的numpy数组,shape为(C,H,W),值范围[0,1]
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise ValueError("输入图像必须是PIL Image或numpy数组格式!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # CHW格式
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3
    return img.astype(np.float32) / 255

def ceil_modulo(x: int, mod: int) -> int:
    """计算大于x且能被mod整除的最小数"""
    return x if x % mod == 0 else (x // mod + 1) * mod

def scale_image(img: np.ndarray, factor: float, interpolation=cv2.INTER_AREA) -> np.ndarray:
    """缩放图像
    
    Args:
        img: 输入图像,shape为(C,H,W)
        factor: 缩放因子
        interpolation: 插值方法
        
    Returns:
        缩放后的图像
    """
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img

def pad_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    """将图像填充到能被mod整除的尺寸"""
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    ) 