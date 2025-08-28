import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

def preprocess_image(image, img_size=640):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binarization
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    # Remove pepper noise (morphological opening)
    kernel = np.ones((5, 5), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Convert single channel back to 3 channels for model compatibility
    image_processed = cv2.cvtColor(binary_cleaned, cv2.COLOR_GRAY2BGR)
    # Resize to model input size
    image_resized = cv2.resize(image_processed, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return image_resized

def detect_objects(image_path, model_path, class_names=None, conf_thres=0.5, image=None):
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Read and preprocess image
    if image is None:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return [], None
    orig_h, orig_w = image.shape[:2]
    img = preprocess_image(image, img_size=640)
    
    # Run inference
    results = model(img, conf=conf_thres)
    detected_objects = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            print("No detections")
            continue  # No detections in this result
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # Scale bounding box coordinates back to original image size
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            label = class_names[cls_id] if class_names else model.names[cls_id]
            detected_objects.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })
    return detected_objects, image

if __name__ == "__main__":
    # Example usage similar to inference.py
    model_path = './best.pt'
    image_path = './front_20250828_003838_427314.jpg'  # Change as needed

    # Load class names from model
    model = YOLO(model_path)
    class_names = model.names if hasattr(model, 'names') else None

    # Detect objects
    detections, image = detect_objects(image_path, model_path, class_names)

    # Draw bounding boxes on the image
    for obj in detections:
        x1, y1, x2, y2 = obj['bbox']
        label = f"{obj['label']} {obj['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow('YOLOv8 Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()