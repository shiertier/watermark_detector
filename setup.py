from setuptools import setup, find_packages

setup(
    name='watermark_detector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'Pillow',
        'torch>=2.0.0',
        'torchvision',
        'tqdm',
        'ultralytics',
    ],
    author='shiertier',
    author_email='junjie.text@gmail.com',
    description='一个用于检测和移除图片水印的工具',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shiertier/watermark_detector',  # 替换为您的项目链接
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)