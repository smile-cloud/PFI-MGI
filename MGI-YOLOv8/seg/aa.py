from ultralytics import YOLO
import os
model_path = './runs/segment/train4/weights/best.pt'
model = YOLO(model_path)
if __name__ == '__main__':

    a = model.predict('1.JPG', save=True, batch=1, workers=1, device='cpu',conf=0.1,iou=0.1)
