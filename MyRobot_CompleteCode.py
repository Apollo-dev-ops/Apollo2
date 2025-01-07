
import wpilib
import wpilib.drive
from ctre import WPI_VictorSPX, NeutralMode
from navx import AHRS
from networktables import NetworkTables
import math
import numpy as np
import cv2
from threading import Thread
import heapq
import queue
from enum import Enum, auto
import logging

# Constants
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
GRID_RESOLUTION = 0.2
OBSTACLE_MARGIN = 0.5
JOY_DEADBAND = 0.05

# Configure logging
logging.basicConfig(level=logging.INFO)

# Enums for game pieces and autonomous routines
class GamePiece(Enum):
    CUBE = auto()
    CONE = auto()
    NONE = auto()

class AutoRoutine(Enum):
    SCORE_TWO_HIGH = auto()
    SCORE_THREE_LOW = auto()
    MOBILITY_AND_DOCK = auto()
    SCORE_AND_BALANCE = auto()

# PathPlanner class
class PathPlanner:
    def __init__(self, field_width: float, field_height: float, grid_resolution: float = GRID_RESOLUTION, obstacle_margin: float = OBSTACLE_MARGIN):
        self.field_width = field_width
        self.field_height = field_height
        self.grid_resolution = grid_resolution
        self.obstacle_margin = obstacle_margin
        self.grid = np.zeros((int(field_width / grid_resolution), int(field_height / grid_resolution)))

    def update_obstacles(self, obstacles: list[tuple[float, float, float]]):
        self.grid.fill(0)  # Reset grid
        for x, y, radius in obstacles:
            self.add_obstacle(x, y, radius)

    def add_obstacle(self, x: float, y: float, radius: float):
        radius_cells = int((radius + self.obstacle_margin) / self.grid_resolution)
        center_x = int(x / self.grid_resolution)
        center_y = int(y / self.grid_resolution)
        y_indices, x_indices = np.ogrid[:self.grid.shape[0], :self.grid.shape[1]]
        dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        self.grid[dist_from_center <= radius_cells] = 1

    def a_star(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        def heuristic(cell1, cell2):
            return math.sqrt((cell1[0] - cell2[0])**2 + (cell1[1] - cell2[1])**2)

        def neighbors(cell):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in directions:
                next_cell = (cell[0] + dx, cell[1] + dy)
                if (0 <= next_cell[0] < self.grid.shape[0] and
                        0 <= next_cell[1] < self.grid.shape[1] and
                        self.grid[next_cell] != 1):
                    yield next_cell

        start_cell = (int(start[0] / self.grid_resolution), int(start[1] / self.grid_resolution))
        goal_cell = (int(goal[0] / self.grid_resolution), int(goal[1] / self.grid_resolution))
        frontier = [(0, start_cell)]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            for next_cell in neighbors(current):
                new_cost = cost_so_far[current] + heuristic(current, next_cell)
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(goal_cell, next_cell)
                    heapq.heappush(frontier, (priority, next_cell))
                    came_from[next_cell] = current

        path = []
        current = goal_cell
        while current:
            path.append((current[0] * self.grid_resolution, current[1] * self.grid_resolution))
            current = came_from.get(current)
        return path[::-1]

# VisionProcessor class
class VisionProcessor:
    def __init__(self, yolo_cfg_path: str, yolo_weights_path: str, camera_id: int = 0):
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.net = cv2.dnn.readNetFromDarknet(yolo_cfg_path, yolo_weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.detected_objects = queue.Queue()
        self._running = True
        self.vision_thread = Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()

    def _vision_loop(self):
        try:
            while self._running:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)
                objects = self._process_detections(outputs, frame)
                while not self.detected_objects.empty():
                    self.detected_objects.get()
                self.detected_objects.put(objects)
        finally:
            self.camera.release()

    def _process_detections(self, outputs, frame):
        height, width = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        detections = [{'bbox': boxes[i], 'class': class_ids[i], 'confidence': confidences[i]} for i in indices.flatten()]
        return detections

    def shutdown(self):
        self._running = False
        self.vision_thread.join()
        self.camera.release()

# Main robot class
class MyRobot(wpilib.TimedRobot):
    def robotInit(self):
        self.drive_motors = [WPI_VictorSPX(i) for i in range(1, 5)]
        self.drive = wpilib.drive.DifferentialDrive(self.drive_motors[0], self.drive_motors[1])
        self.joystick = wpilib.Joystick(0)
        self.vision = VisionProcessor("yolov4-tiny.cfg", "yolov4-tiny.weights")
        self.navx = AHRS.create_spi()

    def autonomousInit(self):
        pass

    def autonomousPeriodic(self):
        pass

    def teleopPeriodic(self):
        # Joystick-based arcade drive
        forward = self._apply_deadband(self.joystick.getRawAxis(1))  # Forward/Backward
        rotation = self._apply_deadband(self.joystick.getRawAxis(2))  # Left/Right
        self.drive.arcadeDrive(forward, rotation)

        # Trigger vision detection
        if self.joystick.getRawButton(1):  # Assume button 1 triggers vision
            if not self.vision.detected_objects.empty():
                objects = self.vision.detected_objects.get()
                logging.info(f"Detected objects: {objects}")

    @staticmethod
    def _apply_deadband(value: float) -> float:
        if abs(value) > JOY_DEADBAND:
            return value
        return 0.0

    def testPeriodic(self):
        # Test motors and sensors
        logging.info("Testing motors and sensors...")
        for i, motor in enumerate(self.drive_motors, start=1):
            motor.setNeutralMode(NeutralMode.Coast)
            motor.set(0.2)  # Run at low speed
            wpilib.Timer.delay(0.5)
            motor.set(0)
            logging.info(f"Motor {i} tested.")

        # Check NavX status
        if not self.navx.isConnected():
            logging.warning("NavX is not connected!")

if __name__ == "__main__":
    wpilib.run(MyRobot)
