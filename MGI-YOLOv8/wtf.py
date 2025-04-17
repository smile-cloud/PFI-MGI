import os
import cv2
import numpy as np

# 定义类别和对应颜色（十六进制）
classes = ['G0', 'G1', 'G2', 'G3', 'G4']
hex_colors = ["191970", "778899", "556B2F", "000000", "FF0000"]


# 将十六进制颜色转换为BGR格式
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


colors = [hex_to_bgr(color) for color in hex_colors]

# 文件夹路径
image_folder = 'datasets/val3/images'
label_folder = 'datasets/val3/labels'

# 创建输出文件夹
output_folder = 'draw'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 参数设置
font_scale = 2.5  # 标签字体大小
thickness = 10  # 边界框和文字的厚度

# 透明度设置
alpha_mask = 0.5  # 掩码透明度
alpha_bbox = 1.0  # 边框透明度
alpha_text_bg = 1.0  # 字体背景透明度
alpha_text = 1.0  # 字体透明度

# 遍历标签文件
for label_file in os.listdir(label_folder):
    label_path = os.path.join(label_folder, label_file)

    # 对应的图片文件
    image_file = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(image_folder, image_file)

    if not os.path.exists(image_path):
        continue

    # 读取图片
    image = cv2.imread(image_path)
    overlay = image.copy()
    height, width, _ = image.shape

    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        points = np.array(parts[1:], dtype=float).reshape(-1, 2)

        # 反归一化坐标
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(int)

        # 画三角形掩膜
        cv2.fillPoly(overlay, [points], color=colors[cls_id])

        # 计算边界框
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        # 画边界框
        bbox_overlay = overlay.copy()
        cv2.rectangle(bbox_overlay, (x_min, y_min), (x_max, y_max), colors[cls_id], thickness)
        cv2.addWeighted(bbox_overlay, alpha_bbox, overlay, 1 - alpha_bbox, 0, overlay)

        # 计算类别背景框的大小
        (text_width, text_height), baseline = cv2.getTextSize(classes[cls_id], cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              thickness)
        y_min_text = max(y_min - text_height - baseline, 0)

        # 画类别背景框
        text_bg_overlay = overlay.copy()
        cv2.rectangle(text_bg_overlay, (x_min, y_min_text), (x_max, y_min), colors[cls_id], thickness=-1)
        cv2.addWeighted(text_bg_overlay, alpha_text_bg, overlay, 1 - alpha_text_bg, 0, overlay)

        # 写类别名称，加粗显示，白色字体
        label = classes[cls_id]
        text_overlay = overlay.copy()
        cv2.putText(text_overlay, label, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)
        cv2.addWeighted(text_overlay, alpha_text, overlay, 1 - alpha_text, 0, overlay)

    # 将半透明的掩膜叠加到原始图像上
    mask_overlay = image.copy()
    cv2.addWeighted(overlay, alpha_mask, image, 1 - alpha_mask, 0, mask_overlay)

    # 保存结果图片
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, mask_overlay)

print("处理完成。")
