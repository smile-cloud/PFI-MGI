import os
import random
import shutil


def ensure_directory_exists(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_files(directory, extension):
    """遍历目录，返回指定扩展名的文件列表"""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.startswith('.'):
                files.append(os.path.join(root, filename))
    return files


def move_files(files, src_dir, dst_dir):
    """移动文件到目标目录"""
    for file in files:
        filename = os.path.basename(file)
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        shutil.move(src_file, dst_file)
        print(f"文件 {src_file} 已移动到 {dst_file}")


def split_dataset(image_files, txt_files, train_ratio=0.8):
    """将数据集按比例划分为训练集和验证集"""
    data = list(zip(image_files, txt_files))
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_images, train_labels = zip(*train_data)
    val_images, val_labels = zip(*val_data)

    return train_images, train_labels, val_images, val_labels


def main():
    current_dir = os.getcwd() + '\\MGI-1014'
    images_dir = os.path.join(current_dir, 'images')
    txt_dir = os.path.join(current_dir, 'txt')

    # 找到所有图像和标签文件
    image_files = find_files(images_dir, '.JPG')
    txt_files = find_files(txt_dir, '.txt')

    # 确保图像和标签文件数量相同
    assert len(image_files) == len(txt_files), "图像和标签文件数量不匹配"

    # 确保目标文件夹存在
    train_images_dir = os.path.join(current_dir, 'train', 'images')
    train_txt_dir = os.path.join(current_dir, 'train', 'labels')
    val_images_dir = os.path.join(current_dir, 'val', 'images')
    val_txt_dir = os.path.join(current_dir, 'val', 'labels')

    ensure_directory_exists(train_images_dir)
    ensure_directory_exists(train_txt_dir)
    ensure_directory_exists(val_images_dir)
    ensure_directory_exists(val_txt_dir)

    # 划分数据集
    train_images, train_labels, val_images, val_labels = split_dataset(image_files, txt_files)

    # 移动文件到相应目录
    move_files(train_images, images_dir, train_images_dir)
    move_files(train_labels, txt_dir, train_txt_dir)
    move_files(val_images, images_dir, val_images_dir)
    move_files(val_labels, txt_dir, val_txt_dir)

    print("数据集划分完成。")


if __name__ == "__main__":
    main()
