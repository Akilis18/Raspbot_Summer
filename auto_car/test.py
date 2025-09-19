import cv2
import os
import sys
import numpy as np

from perception.lane_detection.lane_detector import LaneDetector

# video_path = "./images/front/record_20250919_152643_108088.avi"
# output_path = "./images/front/lane_result_20250919_152643_108088.avi"
image_path = "./images/front/front_20250908_223743_669597.jpg"

def lane_detection_on_image(image_path, show=True, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot open image: {image_path}")
        return
    frame_height, frame_width = img.shape[:2]
    detector = LaneDetector(frame_width, frame_height, plot_enabled=False)
    result_frame, success, lane_info = detector.process_frame(img, show_real_time=True)
    if show:
        cv2.imshow("Lane Detection Image", result_frame if result_frame is not None else img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_path:
        cv2.imwrite(save_path, result_frame if result_frame is not None else img)
        print(f"Saved lane detection result to {save_path}")

# # Lane detection on video
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print(f"Cannot open video: {video_path}")
#     exit(1)

# ret, frame = cap.read()
# if not ret:
#     print("Cannot read first frame from video.")
#     cap.release()
#     exit(1)

# frame_height, frame_width = frame.shape[:2]
# detector = LaneDetector(frame_width, frame_height, plot_enabled=False)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

# while ret:
#     result_frame, success, lane_info = detector.process_frame(frame, show_real_time=True)
#     out.write(result_frame if result_frame is not None else frame)
#     cv2.imshow("Lane Detection", result_frame if result_frame is not None else frame)
#     key = cv2.waitKey(30) & 0xFF
#     if key == ord('q'):
#         break
#     ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# Lane detection on image
lane_detection_on_image(image_path, show=True, save_path="./images/front/lane_result_image.jpg")