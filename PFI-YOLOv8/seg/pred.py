from ultralytics import YOLO
import os
import time

model_path = './runs/segment/train13/weights/best.pt'
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
    directory = r'../datasets/PFI混合'
    # directory = r'../datasets/0908'

    # 获取所有文件的路径名
    all_file_paths = get_all_file_paths(directory)

    for file in all_file_paths:
        start = time.time()
        a = model.predict(file, save=False, batch=1, workers=1, device='cuda', conf=0.3, iou=0)
        end = time.time()
        print("time: ", end - start)
