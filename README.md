# Watermark Detector & Remover

一个用于检测图片水印并筛选/删除带水印图片的Python工具。

## 重要说明

本工具的主要目的是**检测**图片中的水印,用于筛选和删除带有水印的图片。虽然工具中包含了水印移除功能,但这只是一个非常基础的实现,效果有限,仅作为辅助功能提供。

**请注意:** 不建议也不希望该工具被用于任何侵犯版权的行为。水印通常用于保护创作者的知识产权,请尊重他人的劳动成果。

## 功能特点

- 基于YOLO的水印检测
- 支持批量处理图片目录
- 多种输出模式:
  - 保存检测结果为YOLO格式标注文件
  - 保存检测结果为JSON文件
  - 可视化检测结果
- 支持多种保存模式:
  - 原位置覆盖
  - 保存到指定输出目录
  - 保存所有处理结果(包括无水印图片)
- 简单的水印移除功能(基于LaMa模型)

## 使用方法

### 1. 检测单张图片中的水印

```python
from watermark_detector.model.watermark_detector import WatermarkDetector

detector = WatermarkDetector()

# 返回检测到的水印边界框
boxes = detector.detect("image.jpg")

# 保存可视化结果
detector.detect_one_and_show(
    "image.jpg",
    save_path="result.jpg",
    show_window=True
)
```

### 2. 批量检测并保存结果

```python
# 处理整个目录
detector.detect(
    "input_dir",
    output_type="json",
    save_path="detect_results.json",
    recursive=True  # 递归处理子目录
)

# 保存为YOLO格式的标注文件
detector.detect(
    "input_dir",
    output_type="yolo_txt",
    recursive=True
)
```

### 3. 使用水印移除器(实验性功能)

```python
from watermark_detector.model.watermark_remover import WatermarkRemover

remover = WatermarkRemover()

# 处理单张图片
remover.run_one(
    "image.jpg",
    output_dir="output",
    save_mode="output_dir",
    detect_mode="json"  # 同时保存检测结果
)

# 处理整个目录
remover.run_directory(
    "input_dir",
    output_dir="output_dir",
    recursive=True,
    save_mode="output_dir_all",  # 保存所有处理结果
    detect_mode="yolo_txt"  # 保存检测结果为YOLO格式
)
```

## 保存模式说明

- `overwrite`: 直接覆盖原文件
- `output_dir`: 仅保存处理后的图片到输出目录
- `output_dir_all`: 将所有图片(包括无水印图片)复制到输出目录

## 检测结果保存模式

- `none`: 不保存检测结果
- `yolo_txt`: 保存为YOLO格式的标注文件
- `json`: 保存为JSON文件,包含所有图片的检测结果

## 环境要求

- Python >= 3.7
- PyTorch >= 2.0.0
- 其他依赖见requirements.txt

## 注意事项

1. 首次运行时会自动下载预训练模型
2. 水印检测模型针对常见水印类型训练,可能无法检测特殊水印
3. 水印移除功能仅作为实验性功能提供,效果有限
4. 建议在使用前先在小批量数据上测试效果

## 许可证

MIT License

## 免责声明

本工具仅供学习和研究使用。使用者应当遵守相关法律法规,尊重知识产权,不得将本工具用于任何侵犯他人权益的行为。作者对使用者的任何行为不承担责任。
