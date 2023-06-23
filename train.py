from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/data/sungmin/yolo/ultralytics/yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/data/sungmin/yolo/ultralytics/ultralytics/datasets/coco128.yaml", epochs=100, imgsz=640, batch=16)
