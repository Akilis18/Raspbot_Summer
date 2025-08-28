# Import necessary libraries
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('./runs_road_signs/train/weights/best.pt')

# Path to the image you want to run inference on
image_path = 'images_road_signs_after/front_20250828_003838_427314_processed.jpg'  # Change as needed

# Read the image
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Draw bounding boxes on the image

# Get class names from model
class_names = model.names if hasattr(model, 'names') else None

for result in results:
	boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) array
	class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes, 'cls') else []
	for i, box in enumerate(boxes):
		x1, y1, x2, y2 = map(int, box)
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		# Draw class label
		if class_names is not None and i < len(class_ids):
			label = class_names[class_ids[i]]
			cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow('YOLOv8 Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
