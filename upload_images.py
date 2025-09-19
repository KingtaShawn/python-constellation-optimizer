import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import shutil
import argparse
from datetime import datetime

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class ImageUploader:
    """图像上传和管理工具类"""
    def __init__(self, source_dir=None, dest_dir='images', supported_formats=['.png', '.jpg', '.jpeg', '.gif', '.svg']):
        # 如果未指定源目录，则使用当前目录
        self.source_dir = source_dir if source_dir else os.getcwd()
        self.dest_dir = dest_dir
        self.supported_formats = supported_formats
        
        # 创建目标目录（如果不存在）
        os.makedirs(self.dest_dir, exist_ok=True)
        
        print(f"图像上传器初始化成功")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.dest_dir}")
        print(f"支持的图像格式: {', '.join(self.supported_formats)}")
    
    def find_images(self):
        """查找源目录中的所有支持格式的图像文件"""
        image_files = []
        
        for fmt in self.supported_formats:
            # 在源目录及其子目录中查找图像文件
            pattern = os.path.join(self.source_dir, '**', f'*{fmt}')
            files = glob.glob(pattern, recursive=True)
            image_files.extend(files)
        
        # 去除重复的文件路径
        image_files = list(set(image_files))
        
        print(f"找到 {len(image_files)} 个图像文件")
        return image_files
    
    def copy_images(self, image_files):
        """将找到的图像文件复制到目标目录"""
        copied_count = 0
        skipped_count = 0
        
        for img_path in image_files:
            # 获取文件名
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(self.dest_dir, img_name)
            
            # 检查目标文件是否已存在
            if os.path.exists(dest_path):
                # 比较文件大小，如果不同则覆盖
                if os.path.getsize(img_path) != os.path.getsize(dest_path):
                    # 为避免覆盖，添加时间戳
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    name, ext = os.path.splitext(img_name)
                    new_name = f"{name}_{timestamp}{ext}"
                    dest_path = os.path.join(self.dest_dir, new_name)
                    shutil.copy2(img_path, dest_path)
                    copied_count += 1
                    print(f"已复制（重命名）: {img_name} -> {new_name}")
                else:
                    skipped_count += 1
                    print(f"已跳过（文件相同）: {img_name}")
            else:
                # 复制文件到目标目录
                shutil.copy2(img_path, dest_path)
                copied_count += 1
                print(f"已复制: {img_name}")
        
        print(f"\n复制完成: 成功复制 {copied_count} 个文件, 跳过 {skipped_count} 个文件")
        return copied_count, skipped_count
    
    def generate_image_report(self, image_files):
        """生成图像文件报告"""
        report_path = os.path.join(self.dest_dir, "image_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("图像文件报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源目录: {self.source_dir}\n")
            f.write(f"目标目录: {self.dest_dir}\n")
            f.write(f"找到的图像文件总数: {len(image_files)}\n")
            f.write("\n文件列表:\n")
            f.write("-" * 50 + "\n")
            
            # 按文件大小排序
            image_files.sort(key=os.path.getsize, reverse=True)
            
            # 写入每个文件的信息
            for i, img_path in enumerate(image_files, 1):
                img_name = os.path.basename(img_path)
                size_kb = os.path.getsize(img_path) / 1024
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                except Exception as e:
                    width, height = "N/A", "N/A"
                    mode = "N/A"
                    print(f"无法获取图像信息: {img_name}, 错误: {e}")
                
                f.write(f"{i}. {img_name}\n")
                f.write(f"   大小: {size_kb:.2f} KB\n")
                f.write(f"   尺寸: {width}x{height}\n")
                f.write(f"   模式: {mode}\n")
                f.write(f"   源路径: {img_path}\n")
                f.write(f"   目标路径: {os.path.join(self.dest_dir, img_name)}\n\n")
        
        print(f"图像报告已生成: {report_path}")
        return report_path
    
    def generate_image_gallery(self, thumbnail_size=(200, 200)):
        """生成图像缩略图画廊"""
        # 收集目标目录中的所有图像
        dest_images = []
        for fmt in self.supported_formats:
            pattern = os.path.join(self.dest_dir, f'*{fmt}')
            files = glob.glob(pattern)
            dest_images.extend(files)
        
        if not dest_images:
            print("目标目录中没有找到图像文件，无法生成画廊")
            return None
        
        # 计算画廊布局
        num_images = len(dest_images)
        cols = min(4, num_images)  # 最多4列
        rows = (num_images + cols - 1) // cols  # 向上取整计算行数
        
        # 创建大图
        fig = plt.figure(figsize=(cols * 4, rows * 4))
        fig.suptitle('图像画廊', fontsize=16)
        
        for i, img_path in enumerate(dest_images):
            try:
                # 打开图像并创建缩略图
                with Image.open(img_path) as img:
                    img.thumbnail(thumbnail_size)
                    
                    # 将PIL图像转换为numpy数组
                    img_array = np.array(img)
                    
                    # 添加子图
                    ax = fig.add_subplot(rows, cols, i + 1)
                    ax.imshow(img_array)
                    ax.axis('off')
                    
                    # 添加文件名作为标题
                    img_name = os.path.basename(img_path)
                    ax.set_title(img_name, fontsize=8, wrap=True)
            except Exception as e:
                print(f"无法处理图像 {os.path.basename(img_path)}: {e}")
                # 添加一个空的子图
                ax = fig.add_subplot(rows, cols, i + 1)
                ax.axis('off')
                ax.set_title(f"无法加载: {os.path.basename(img_path)}", fontsize=8, wrap=True)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为标题留出空间
        
        # 保存画廊
        gallery_path = os.path.join(self.dest_dir, "image_gallery.png")
        plt.savefig(gallery_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图像画廊已生成: {gallery_path}")
        return gallery_path
    
    def optimize_images(self, quality=90):
        """优化目标目录中的图像文件"""
        optimized_count = 0
        
        # 获取目标目录中的所有图像
        dest_images = []
        for fmt in self.supported_formats:
            pattern = os.path.join(self.dest_dir, f'*{fmt}')
            files = glob.glob(pattern)
            dest_images.extend(files)
        
        if not dest_images:
            print("目标目录中没有找到图像文件，无法进行优化")
            return 0
        
        print(f"开始优化 {len(dest_images)} 个图像文件")
        
        for img_path in dest_images:
            img_name = os.path.basename(img_path)
            try:
                # 打开图像
                with Image.open(img_path) as img:
                    # 检查是否为JPEG或PNG格式
                    if img.format in ['JPEG', 'PNG']:
                        # 保存优化后的图像
                        if img.format == 'JPEG':
                            # 对于JPEG，使用指定的质量
                            img.save(img_path, quality=quality, optimize=True)
                        else:
                            # 对于PNG，使用优化参数
                            img.save(img_path, optimize=True)
                        optimized_count += 1
                        print(f"已优化: {img_name}")
                    else:
                        print(f"跳过（不支持优化的格式）: {img_name}")
            except Exception as e:
                print(f"优化失败: {img_name}, 错误: {e}")
        
        print(f"图像优化完成: 成功优化 {optimized_count} 个文件")
        return optimized_count

# 批量处理图像的工具函数
def batch_process_images(source_dirs=None, dest_dir='images', optimize=True, generate_gallery=True):
    """批量处理多个源目录中的图像"""
    if source_dirs is None:
        source_dirs = [os.getcwd()]
    
    total_images = 0
    total_copied = 0
    
    # 为每个源目录创建单独的上传器实例
    for source_dir in source_dirs:
        print(f"\n处理源目录: {source_dir}")
        
        # 创建图像上传器
        uploader = ImageUploader(source_dir=source_dir, dest_dir=dest_dir)
        
        # 查找图像文件
        image_files = uploader.find_images()
        total_images += len(image_files)
        
        # 复制图像文件
        copied, _ = uploader.copy_images(image_files)
        total_copied += copied
    
    # 创建最终的上传器，用于生成报告和画廊
    final_uploader = ImageUploader(source_dir=dest_dir, dest_dir=dest_dir)  # 使用目标目录作为源目录
    
    # 生成图像报告
    final_uploader.generate_image_report(final_uploader.find_images())
    
    # 优化图像（如果需要）
    if optimize:
        final_uploader.optimize_images()
    
    # 生成图像画廊（如果需要）
    if generate_gallery:
        final_uploader.generate_image_gallery()
    
    print(f"\n批量处理完成！")
    print(f"总处理图像数量: {total_images}")
    print(f"成功复制的图像数量: {total_copied}")
    print(f"所有图像已保存到: {os.path.abspath(dest_dir)}")

# 主函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='图像上传和管理工具')
    parser.add_argument('-s', '--source', type=str, help='源目录路径')
    parser.add_argument('-d', '--destination', type=str, default='images', help='目标目录路径')
    parser.add_argument('-o', '--optimize', action='store_true', help='是否优化图像')
    parser.add_argument('-g', '--gallery', action='store_true', help='是否生成图像画廊')
    parser.add_argument('-r', '--recursive', action='store_true', help='是否递归搜索子目录')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了源目录，则使用指定的源目录
    source_dir = args.source if args.source else os.getcwd()
    
    print("图像上传工具")
    print("=" * 50)
    
    # 创建图像上传器
    uploader = ImageUploader(source_dir=source_dir, dest_dir=args.destination)
    
    # 查找图像文件
    image_files = uploader.find_images()
    
    if not image_files:
        print("未找到任何图像文件，程序退出")
        return
    
    # 复制图像文件
    copied, skipped = uploader.copy_images(image_files)
    
    if copied > 0:
        # 生成图像报告
        uploader.generate_image_report(image_files)
        
        # 优化图像（如果需要）
        if args.optimize:
            uploader.optimize_images()
        
        # 生成图像画廊（如果需要）
        if args.gallery:
            uploader.generate_image_gallery()
    
    print("\n图像上传任务已完成！")
    print(f"详细信息请查看: {os.path.join(args.destination, 'image_report.txt')}")

# 当作为主程序运行时执行main函数
if __name__ == "__main__":
    main()