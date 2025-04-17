import os
import json
from shutil import copy2


def find_files(directory, extension):
    """遍历目录，返回指定扩展名的文件列表"""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension) and not filename.startswith('.'):
                files.append(os.path.join(root, filename))
    return files


def copy_file(src, dst_dir):
    """复制文件到目标目录"""
    try:
        dst_path = os.path.join(dst_dir, os.path.basename(src))
        copy2(src, dst_path)
        print(f"文件 {src} 已复制到 {dst_path}")
    except Exception as e:
        print(f"无法复制文件 {src}：{e}")


def main():
    current_dir = r"D:\Code\Python\PFI-YOLOv8\datasets\T1、T2 PFI AI 辅助\8 Junior2+AI-PFI-T2"  # 获取当前目录
    json_files = find_files(current_dir, '.json')  # 找到所有json文件

    for json_file in json_files:
        if os.path.basename(json_file).startswith('.'):
            print(f"跳过以点开头的文件：{json_file}")
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                image_path = data.get('imagePath')
                if not image_path:
                    print(f"无法读取 {json_file} 中的 imagePath")
                    continue

                # 从JSON文件名中提取基础文件名
                json_basename = os.path.splitext(os.path.basename(json_file))[0]
                image_basename = os.path.splitext(image_path)[0]

                # 检查JSON文件名是否与JPG文件名一致
                if json_basename != image_basename:
                    print(f"JSON文件名 {json_basename} 与图像文件名 {image_basename} 不一致")
                    continue

                image_file = os.path.join(os.path.dirname(json_file), image_path)
                if not os.path.isfile(image_file):
                    print(f"找不到图像文件 {image_file}")
                    continue

                # 确保目标文件夹存在
                json_dir = os.path.join(current_dir,  'json')
                image_dir = os.path.join(current_dir, 'images')
                os.makedirs(json_dir, exist_ok=True)
                os.makedirs(image_dir, exist_ok=True)

                # 复制文件
                copy_file(json_file, json_dir)
                copy_file(image_file, image_dir)

        except json.JSONDecodeError:
            print(f"无法解析 {json_file} 为JSON格式")


if __name__ == "__main__":
    main()
