import curses
import time
import sys
import os

# Ensure the firmware directory is always on sys.path, regardless of current working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIRMWARE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../firmware"))
if FIRMWARE_DIR not in sys.path:
    sys.path.insert(0, FIRMWARE_DIR)

from YB_car import YB_Pcb_Car

class RemoteControl:
    def __init__(self):
        self.car = YB_Pcb_Car()  # Instantiate the car object
        self.car.Car_Stop()
        self.car.Ctrl_Servo(1, 90)  # Set initial servo positions
        self.car.Ctrl_Servo(2, 120)

    def start(self):
        curses.wrapper(self.run)

    def run(self, stdscr):
        stdscr.clear()
        stdscr.addstr("Use W/A/S/D to move, Space to stop, Q to quit\n")
        stdscr.refresh()

        while True:
            key = stdscr.getch()
            stdscr.addstr(2, 0, f"Key pressed: {chr(key) if key != -1 else 'None'}      ")
            stdscr.refresh()

            if key == ord('w'):
                self.car.Car_Run(100, 100)
            elif key == ord('s'):
                self.car.Car_Back(100, 100)
            elif key == ord('a'):
                self.car.Car_Left(0, 100)
            elif key == ord('d'):
                self.car.Car_Right(100, 0)
            elif key == ord('x'):
                self.car.Car_Stop()
            elif key == ord('j'):
                self.car.Ctrl_Servo(1, 150)
                self.car.Ctrl_Servo(2, 140)
            elif key == ord('k'):
                self.car.Ctrl_Servo(1, 85)
                self.car.Ctrl_Servo(2, 100)
            elif key == ord('l'):
                self.car.Ctrl_Servo(1, 30)
                self.car.Ctrl_Servo(2, 140)
            elif key == ord('q'):
                break

        self.close_connection()

    def close_connection(self):
        self.car.Car_Stop()

if __name__ == "__main__":
    remote_control = RemoteControl()
    remote_control.start()
