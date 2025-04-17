from ultralytics import YOLO

model_path = './runs/segment/train10/weights/best.pt'
model = YOLO(model_path)
if __name__ == '__main__':
    metric = model.val(data=r"D:\Code\Python\MGI-YOLOv8\seg\ultralytics\cfg\datasets\coco128-seg.yaml",
                       device='cuda', conf=0.355, iou=0)





