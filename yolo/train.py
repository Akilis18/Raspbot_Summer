from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    # --------------------------
    # Model Structure
    # --------------------------
    data="./dataset/data.yaml",
    imgsz=640,
    batch=4,
    model="yolov8n.yaml",

    # --------------------------
    # Training Behavior
    # --------------------------
    epochs=100,
    patience=20,
    device=0,  # GPU 0
    save=True,
    save_period=5,  # save checkpoint every 5 epochs
    project="./runs_road_signs",
    exist_ok=True,

    # --------------------------
    # Data Preprocessing / Augmentation
    # --------------------------
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    flipud=0.1,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

# model.export(format="onnx")
# model.export(format="onnx", imgsz=640, opset=11)
