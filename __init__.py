"""
Python Constellation Optimizer 包

一个基于深度学习和Sionna框架的星座图优化工具库
"""

# 包版本
__version__ = "1.0.0"

# 作者信息
__author__ = "Cao Jianhua"
__email__ = "caojh2005@163.com"

# 导入主要模块
try:
    from .sionna_basis import *
    from .channel_polarization import *
    from .differentiable_Communication_System import *
    from .fixed_constellation_optimizer import *
    from .fixed_advanced_constellation import *
    from .constellation_analysis import *
    from .High_level_simulation import *
    from .upload_images import *
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("某些功能可能不可用。请确保所有依赖项已正确安装。")

# 包描述
__description__ = "一个基于深度学习和Sionna框架的星座图优化工具库"
__url__ = "https://github.com/KingtaShawn/python-constellation-optimizer"

# 导出的公共API
__all__ = [
    # 主要模块
    "sionna_basis",
    "channel_polarization",
    "differentiable_Communication_System",
    "fixed_constellation_optimizer",
    "fixed_advanced_constellation",
    "constellation_analysis",
    "High_level_simulation",
    "upload_images",
    
    # 版本信息
    "__version__",
    "__author__",
    "__description__",
    "__url__"
]