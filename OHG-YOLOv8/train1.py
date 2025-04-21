from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8s-cls.pt')  # 加载预训练模型

    # Train the model
    results = model.train(data=r'D:\Work\paper\projects\huaxi\teeth\datasets\T1', epochs=200, imgsz=64,batch=64,workers=2)