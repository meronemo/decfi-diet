model = YOLO("yolov8n.pt")
model.train(
    data=data_yaml,     #path to data.yaml
    epochs=100,         #number of epochs
    imgsz=640,          #image size
    batch=16,           #batch size
    device=0            #GPU device
)
