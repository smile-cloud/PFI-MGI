from ultralytics import YOLO
import os

model_path = './runs/segment/train10/weights/best.pt'
model = YOLO(model_path)
if __name__ == '__main__':
    def get_all_file_paths(directory):
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths


    # 指定文件夹路径
    # directory = r'../datasets/0908'
    # directory = r'../datasets/val5/images'
    directory = r'../datasets/MGI-1030/val/images'

    # 获取所有文件的路径名
    all_file_paths = get_all_file_paths(directory)

    # 获取文件名列表
    all_file_paths = get_all_file_paths(directory)

    for file in all_file_paths:
        a = model.predict(file, save=True, batch=1, workers=1, device='cpu', conf=0, iou=0)
