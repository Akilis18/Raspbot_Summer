# main.py
"""
This script serves as the entry point for the application.
It orchestrates the entire system.
"""

import threading
import cv2
import sys
import curses
from datetime import datetime
import pathlib

import control.manual_control as manual_control
from perception.sensors.camera_node import CameraNode


class CarApp:
    def __init__(self):
        self.stop_event = threading.Event()
        self.base_dir = pathlib.Path("images")
        self.front_dir = self.base_dir / "front"
        self.front_dir.mkdir(parents=True, exist_ok=True)

    def suppress_libpng_warnings(self):
        sys.stderr = open('/dev/null', 'w')

    def run_remote_control(self, stdscr):
        remote = manual_control.RemoteControl()

        def run_remote(stdscr):
            remote.car.Car_Stop()
            remote.car.Ctrl_Servo(1, 85)
            remote.car.Ctrl_Servo(2, 110)
            
            stdscr.nodelay(True)
            stdscr.clear()
            stdscr.addstr("Use W/A/S/D to move, Space to stop, Q to quit\n")
            stdscr.refresh()

            while not self.stop_event.is_set():
                key = stdscr.getch()
                stdscr.addstr(2, 0, f"Key pressed: {chr(key) if key != -1 else 'None'}      ")
                stdscr.refresh()

                if key == ord('w'):
                    remote.car.Car_Run(150, 150)
                elif key == ord('s'):
                    remote.car.Car_Back(150, 150)
                elif key == ord('a'):
                    remote.car.Car_Left(0, 150)
                elif key == ord('d'):
                    remote.car.Car_Right(150, 0)
                elif key == ord('x'):
                    remote.car.Car_Stop()
                elif key == ord('q'):
                    self.stop_event.set()
                    break

        curses.wrapper(run_remote)
        remote.close_connection()

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
        # Placeholder for autonomous driving logic (lane detection, object detection, etc.)
        self.run_cameras()
        print("Auto mode finished.")

    def manual_mode(self):
        print("Manual mode starting...")
        cam_thread = threading.Thread(target=self.run_cameras)
        cam_thread.start()

        curses.wrapper(self.run_remote_control)

        self.stop_event.set()  # Ensure both threads are stopped
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
