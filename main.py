from ultralytics import YOLO
import torch
if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = YOLO("yolov8n.yaml") #build a new model
    results = model.train(data="config.yaml",epochs=30,workers=1,device=[0],batch=1)