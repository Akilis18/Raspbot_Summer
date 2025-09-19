# main.py
"""
This script serves as the entry point for the application.
It orchestrates the entire system.
"""

import threading
import cv2
import sys
from datetime import datetime
import pathlib

import time

from perception.sensors.camera_node import CameraNode
from perception.lane_detection.lane_detector import LaneDetector
from control.control import CarController
# from path_planning import planner  # To be implemented
# from perception.object_detection import object_detection
# from path_planning import decision_making
from perception.object_detection import sign_detection  # Use your CV-based sign detection

class CarApp:
    def __init__(self):
        self.stop_event = threading.Event()
        self.base_dir = pathlib.Path("images")
        self.front_dir = self.base_dir / "front"
        self.front_dir.mkdir(parents=True, exist_ok=True)
        self.controller = CarController()
        self.plan_stack = ["turnR", "forward"]  # Temporary stack for planned commands

    def suppress_libpng_warnings(self):
        sys.stderr = open('/dev/null', 'w')

    def run_cameras(self):
        cam_front = CameraNode(camera_index=0, resolution=(640, 480), flip_front=True)
        cam_front.start()
        try:
            while not self.stop_event.is_set():
                frame_front = cam_front.get_frame()
                cv2.imshow("Cam", frame_front)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break
                elif key == ord('g'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    front_path = self.front_dir / f"front_{timestamp}.jpg"
                    cv2.imwrite(str(front_path), frame_front)
                    print(f"Saved: {front_path}")
        finally:
            cam_front.stop()
            cv2.destroyAllWindows()

    def auto_mode(self):
        print("Auto mode starting...")

        self.controller.stop()
        self.controller.set_servo(1, 90)
        self.controller.set_servo(2, 120)

        cam = CameraNode(camera_index=0, resolution=(640, 480), flip_front=True)
        cam.start()

        try:
            last_sign_present = False
            sign_left_view = False

            while not self.stop_event.is_set():
                # --- Perception ---
                frame = cam.get_frame()
                if frame is None:
                    continue

                # Lane detection
                detector = LaneDetector(frame.shape[1], frame.shape[0], plot_enabled=False)
                result_frame, success, lane_info = detector.process_frame(frame, show_real_time=True)

                # Sign detection using OpenCV method
                sign_results = sign_detection.detect_signs(frame)
                detected_signs = []
                for result in sign_results:
                    detected_signs.append({
                        "label": result.get("label", "sign"),
                        "bbox": result.get("bbox", (0, 0, 0, 0))
                    })

                # --- Path Planning / Decision ---
                decision = None
                # Assign decision only when multiple objects are detected
                if len(detected_signs) >= 1 and self.plan_stack:
                    decision = self.plan_stack[0]  # Peek, don't pop yet

                # Track sign presence for "just left view"
                sign_present = bool(detected_signs)
                if last_sign_present and not sign_present:
                    sign_left_view = True
                else:
                    sign_left_view = False
                last_sign_present = sign_present

                # --- Control ---
                if decision in ["turnL", "turnR"]:
                    # Execute the planned command immediately when multiple objects are detected
                    decision = self.plan_stack.pop(0)
                    print(f"Executing planned command: {decision}")

                    if decision == "turnL":
                        # # Check if sign is left of ROI before turning
                        # if detected_signs:
                        #     sign_x = (detected_signs[0]['bbox'][0] + detected_signs[0]['bbox'][2]) // 2
                        #     roi_left = lane_info.get('roi_left', 0)
                        #     if sign_x < roi_left:
                        #         print("Sign is left of ROI, switching left before turnL")
                        #         self.controller.switchL()
                        self.controller.turnL()
                    elif decision == "turnR":
                        self.controller.turnR()
                    else:
                        self.controller.set_speed(80, 80)
                else:
                    if decision == "forward":
                        # Pop and execute forward
                        self.plan_stack.pop(0)
                    
                    # Default: lane following
                    if success and lane_info:
                        steer_deg = lane_info.get('steer_deg', 0.0)
                        base_speed = 50  # Slower base speed
                        steering_gain = 1.5
                        speed_diff = steering_gain * steer_deg
                        left_speed = max(0, min(70, base_speed - speed_diff))   # Reduced max speed
                        right_speed = max(0, min(70, base_speed + speed_diff))  # Reduced max speed
                    else:
                        left_speed = 0
                        right_speed = 0
                    self.controller.set_speed(left_speed, right_speed)

                # --- Visualization (unchanged, but draw sign bboxes) ---
                for obj in detected_signs:
                    x1, y1, x2, y2 = obj['bbox']
                    label = obj['label']
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Auto Mode - Lane & Object Detection", result_frame if result_frame is not None else frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break

                # time.sleep(1)
                # self.controller.stop()
                # time.sleep(0.5)

        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            self.controller.stop()
            self.controller.stop()
            cam.stop()
            cv2.destroyAllWindows()
            self.stop_event.set()
            print("Auto mode finished.")

    def manual_mode(self):
        print("Manual mode starting...")
        # Load object detection model once
        script_dir = pathlib.Path(__file__).parent.resolve()
        # from perception.object_detection import object_detection
        # model_path = str(script_dir / "perception" / "object_detection" / "best.pt")
        # model = object_detection.YOLO(model_path)
        # class_names = model.names if hasattr(model, 'names') else None

        def camera_and_detection():
            cam_front = CameraNode(camera_index=0, resolution=(640, 480), flip_front=True)
            cam_front.start()
            recording = False
            video_writer = None
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_path = None
            try:
                while not self.stop_event.is_set():
                    frame_front = cam_front.get_frame()
                    if frame_front is None:
                        continue

                    # Object detection (YOLO) commented out
                    # detections, _ = object_detection.detect_objects(
                    #     image_path=None,
                    #     model_path=model_path,
                    #     class_names=class_names,
                    #     conf_thres=0.5,
                    #     image=frame_front
                    # )

                    # Draw bounding boxes on the image
                    # for obj in detections:
                    #     x1, y1, x2, y2 = obj['bbox']
                    #     label = f"{obj['label']} {obj.get('confidence', 0):.2f}"
                    #     cv2.rectangle(frame_front, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #     cv2.putText(frame_front, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Handle recording
                    if recording:
                        if video_writer is None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            out_path = self.front_dir / f"record_{timestamp}.avi"
                            video_writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (frame_front.shape[1], frame_front.shape[0]))
                            print(f"Recording started: {out_path}")
                        video_writer.write(frame_front)
                    else:
                        if video_writer is not None:
                            video_writer.release()
                            print(f"Recording stopped: {out_path}")
                            video_writer = None

                    cv2.imshow("Cam + Detection", frame_front)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break
                    elif key == ord('g'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        front_path = self.front_dir / f"front_{timestamp}.jpg"
                        cv2.imwrite(str(front_path), frame_front)
                        print(f"Saved: {front_path}")
                    elif key == ord('r'):
                        recording = not recording  # Toggle recording

            finally:
                if video_writer is not None:
                    video_writer.release()
                cam_front.stop()
                cv2.destroyAllWindows()

        cam_thread = threading.Thread(target=camera_and_detection)
        cam_thread.start()
        from control.manual_control import RemoteControl
        remote = RemoteControl()
        remote.start()
        self.stop_event.set()
        cam_thread.join()
        print("Manual mode finished.")

    def start(self, mode="manual"):
        self.suppress_libpng_warnings()
        if mode == "manual":
            self.manual_mode()
        elif mode == "auto":
            self.auto_mode()

if __name__ == "__main__":
    mode = input("Enter mode (manual/auto): ").strip().lower()
    app = CarApp()
    app.start(mode)
