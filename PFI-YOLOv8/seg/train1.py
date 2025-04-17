from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\Code\Python\PFI-YOLOv8\seg\ultralytics\cfg\models\v8\yolov8-seg.yaml")  # build a new model
    # from YAML
    model = YOLO("yolov8s-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model

    results = model.train(data=r"D:\Code\Python\PFI-YOLOv8\seg\ultralytics\cfg\datasets\coco128-seg.yaml",
                          epochs=200, imgsz=640, batch=8, workers=1, device='cuda')
    # Train the model
    # results = model.train(data='T1', epochs=100, imgsz=64,batch=64,workers=2)
