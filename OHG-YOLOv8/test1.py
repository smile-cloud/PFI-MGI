from ultralytics import YOLO

model_path = './runs/classify/train7/weights/best.pt'
model = YOLO(model_path)
if __name__ == '__main__':
    model.val(data=r'D:\Work\paper\projects\huaxi\teeth\datasets\T2', conf=0.3, device='cpu')
