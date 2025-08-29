import sys
import os

import time

# Dynamically add the firmware directory to sys.path for absolute import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIRMWARE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../firmware"))
if FIRMWARE_DIR not in sys.path:
    sys.path.insert(0, FIRMWARE_DIR)

from YB_car import YB_Pcb_Car

class CarController:
    def __init__(self):
        self.car = YB_Pcb_Car()
        self.car.Car_Stop()
        self.car.Ctrl_Servo(1, 85)
        self.car.Ctrl_Servo(2, 110)

    def set_speed(self, left_speed, right_speed):
        if left_speed > 0 or right_speed > 0:
            self.car.Car_Run(int(left_speed), int(right_speed))
        else:
            self.car.Car_Stop()

    def stop(self):
        self.car.Car_Stop()

    def set_servo(self, servo_id, angle):
        self.car.Ctrl_Servo(servo_id, angle)


    def turnL(self):
        # soft big L turn
        # self.car.Control_Car(60, 105)

        # sharp big L turn
        self.car.Control_Car(50, 110)
        time.sleep(1)
        self.car.Car_Stop()

    def turnR(self):
        # sharp big R turn
        self.car.Control_Car(100, 30)
        time.sleep(1.3)
        self.car.Car_Stop()

    def switchL(self):
        self.car.Control_Car(40, 100)
        time.sleep(0.3)
        self.car.Car_Run(70, 70)
        time.sleep(0.5)
        self.car.Control_Car(100, 40)
        time.sleep(0.3)
        self.car.Car_Stop()

    def switchR(self):
        self.car.Control_Car(100, 40)
        time.sleep(0.3)
        self.car.Car_Run(70, 70)
        time.sleep(0.5)
        self.car.Control_Car(40, 100)
        time.sleep(0.3)
        self.car.Car_Stop()
<<<<<<< Updated upstream

import sys
import os

import time

# Dynamically add the firmware directory to sys.path for absolute import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FIRMWARE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../firmware"))
if FIRMWARE_DIR not in sys.path:
    sys.path.insert(0, FIRMWARE_DIR)

from YB_car import YB_Pcb_Car

class CarController:
    def __init__(self):
        self.car = YB_Pcb_Car()
        self.car.Car_Stop()
        self.car.Ctrl_Servo(1, 85)
        self.car.Ctrl_Servo(2, 110)

    def set_speed(self, left_speed, right_speed):
        if left_speed > 0 or right_speed > 0:
            self.car.Car_Run(int(left_speed), int(right_speed))
        else:
            self.car.Car_Stop()

    def stop(self):
        self.car.Car_Stop()

    def set_servo(self, servo_id, angle):
        self.car.Ctrl_Servo(servo_id, angle)


    def turnL(self):
        # soft big L turn
        # self.car.Control_Car(60, 105)

        # sharp big L turn
        self.car.Control_Car(50, 110)
        time.sleep(0.1)
        self.car.Car_Stop()

    def turnR(self):
        # sharp big R turn
        self.car.Control_Car(100, 30)
        time.sleep(0.1)
        self.car.Car_Stop()

    def switchL(self):
        self.car.Control_Car(40, 100)
        time.sleep(0.1)
        self.car.Car_Run(70, 70)
        time.sleep(0.2)
        self.car.Control_Car(100, 40)
        time.sleep(0.1)
        self.car.Car_Stop()

    def switchR(self):
        self.car.Control_Car(100, 40)
        time.sleep(0.1)
        self.car.Car_Run(70, 70)
        time.sleep(0.2)
        self.car.Control_Car(40, 100)
        time.sleep(0.1)
        self.car.Car_Stop()
=======
>>>>>>> Stashed changes
