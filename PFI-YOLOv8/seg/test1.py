from ultralytics import YOLO

model_path = './runs/segment/train13/weights/best.pt'
model = YOLO(model_path)
if __name__ == '__main__':
    model.val(data=r"D:\Code\Python\PFI-YOLOv8\seg\ultralytics\cfg\datasets\coco128-seg.yaml",
              device='cuda', conf=0.381, iou=0)
