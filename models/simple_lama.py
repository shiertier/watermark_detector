import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple
from ..utils import image_utils, download_utils

class SimpleLama:
    """简单的图像修复模型"""
    
    DEFAULT_MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"

    def __init__(self, device: torch.device = None):
        """初始化模型
        
        Args:
            device: 运行设备,默认使用GPU(如果可用)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = self._load_model()
        self.model.eval()
        self.model.to(device)

    def _load_model(self) -> torch.jit.ScriptModule:
        """加载模型文件"""
        model_path = os.environ.get("LAMA_MODEL")
        if model_path and os.path.exists(model_path):
            return torch.jit.load(model_path, map_location=self.device)
            
        model_url = os.environ.get("LAMA_MODEL_URL", self.DEFAULT_MODEL_URL)
        model_path = download_utils.download_model(model_url)
        return torch.jit.load(model_path, map_location=self.device)

    def _prepare_inputs(
        self, 
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        scale_factor: float = None,
        pad_modulo: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备模型输入数据"""
        image = image_utils.convert_to_numpy(image)
        mask = image_utils.convert_to_numpy(mask)

        if scale_factor:
            image = image_utils.scale_image(image, scale_factor)
            mask = image_utils.scale_image(mask, scale_factor, cv2.INTER_NEAREST)

        if pad_modulo > 1:
            image = image_utils.pad_to_modulo(image, pad_modulo)
            mask = image_utils.pad_to_modulo(mask, pad_modulo)

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        mask = (mask > 0) * 1

        return image, mask

    def __call__(
        self, 
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray]
    ) -> Image.Image:
        """执行图像修复
        
        Args:
            image: 输入图像
            mask: 修复掩码
            
        Returns:
            修复后的图像
        """
        image, mask = self._prepare_inputs(image, mask)

        with torch.inference_mode():
            output = self.model(image, mask)
            result = output[0].permute(1, 2, 0).detach().cpu().numpy()
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(result) 