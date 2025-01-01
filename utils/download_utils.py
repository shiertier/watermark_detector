import os
import sys
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir

def get_cache_path(url: str, filename: str = None) -> str:
    """获取模型文件的缓存路径"""
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    if filename is None:
        filename = os.path.basename(parts.path)
    return os.path.join(model_dir, filename)

def download_model(url: str, filename: str = None) -> str:
    """下载模型文件
    
    Args:
        url: 模型文件URL
        filename: 保存的文件名,如果为None则使用URL中的文件名
        
    Returns:
        本地缓存文件路径
    """
    cached_file = get_cache_path(url, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('正在下载: "{}" 到 {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, hash_prefix=None, progress=True)
    return cached_file 