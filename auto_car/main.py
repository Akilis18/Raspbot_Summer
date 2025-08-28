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

from perception.sensors.camera_node import CameraNode
from perception.lane_detection import lane_origin
from control.control import CarController
# from path_planning import planner  # To be implemented
from perception.object_detection import object_detection
# from path_planning import decision_making

class CarApp:
    def __init__(self):
        self.stop_event = threading.Event()
        self.base_dir = pathlib.Path("images")
        self.front_dir = self.base_dir / "front"
        self.front_dir.mkdir(parents=True, exist_ok=True)
        self.controller = CarController()
        self.plan_stack = ["turnL", "forward"]  # Temporary stack for planned commands

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
        self.controller.set_servo(1, 85)
        self.controller.set_servo(2, 110)

        cam = CameraNode(camera_index=0, resolution=(640, 480), flip_front=True)
        cam.start()

        # Load object detection model once
        script_dir = pathlib.Path(__file__).parent.resolve()
        model_path = str(script_dir / "perception" / "object_detection" / "best.pt")
        model = object_detection.YOLO(model_path)
        class_names = model.names if hasattr(model, 'names') else None

        try:
            last_sign_present = False
            sign_left_view = False

            while not self.stop_event.is_set():
                # --- Perception ---
                frame = cam.get_frame()
                if frame is None:
                    continue

                # Lane detection
                result_frame, success, lane_info = lane_origin.process_one_frame(frame, plot=False, show_real_time=True)

                # Object detection
                detections, _ = object_detection.detect_objects(
                    image_path=None,
                    model_path=model_path,
                    class_names=class_names,
                    conf_thres=0.5,
                    image=frame
                )

                # Gather detected sign names and positions
                detected_signs = []
                for obj in detections:
                    detected_signs.append({
                        "label": obj["label"],
                        "bbox": obj["bbox"]
                    })

                # --- Path Planning / Decision ---
                decision = None
                if self.plan_stack:
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
                    # 1. Road sign detected (in current or previous frame)
                    # 2. Sign just left camera view
                    # 3. Only one lane present
                    one_lane = lane_info.get('lane_count', 2) == 1 if lane_info else False
                    if sign_left_view and one_lane:
                        # Pop the decision now
                        decision = self.plan_stack.pop(0)
                        print(f"Executing planned command: {decision}")

                        if decision == "turnL":
                            # Check if sign is left of ROI before turning
                            if detected_signs:
                                sign_x = (detected_signs[0]['bbox'][0] + detected_signs[0]['bbox'][2]) // 2
                                roi_left = lane_info.get('roi_left', 0)
                                if sign_x < roi_left:
                                    print("Sign is left of ROI, switching left before turnL")
                                    self.controller.switchL()
                            self.controller.turnL()
                        elif decision == "turnR":
                            self.controller.turnR()
                        else:
                            self.controller.set_speed(80, 80)
                    else:
                        # Keep going forward or stop
                        self.controller.set_speed(80, 80)
                elif decision == "forward":
                    # Pop and execute forward
                    self.plan_stack.pop(0)
                    self.controller.set_speed(80, 80)
                else:
                    # Default: lane following
                    if success and lane_info:
                        steer_deg = lane_info.get('steer_deg', 0.0)
                        base_speed = 80
                        steering_gain = 1.5
                        speed_diff = steering_gain * steer_deg
                        left_speed = max(0, min(255, base_speed - speed_diff))
                        right_speed = max(0, min(255, base_speed + speed_diff))
                    else:
                        left_speed = 0
                        right_speed = 0
                    self.controller.set_speed(left_speed, right_speed)

                # --- Visualization (unchanged) ---
                for obj in detections:
                    x1, y1, x2, y2 = obj['bbox']
                    label = f"{obj['label']} {obj.get('confidence', 0):.2f}"
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Auto Mode - Lane & Object Detection", result_frame if result_frame is not None else frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break

        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            self.controller.stop()
            cam.stop()
            cv2.destroyAllWindows()
            print("Auto mode finished.")

    def manual_mode(self):
        print("Manual mode starting...")
        cam_thread = threading.Thread(target=self.run_cameras)
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
