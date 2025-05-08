#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys, math, time, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
from PyQt5.QtWidgets import QApplication, QGraphicsLineItem, QGraphicsScene, QGraphicsView
from sklearn.preprocessing import StandardScaler, RobustScaler
from dwa_window import Ui_DWA_Simulator
from src.preprocessing.point_cloud import process_point_cloud
import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from natsort import natsorted
from src.preprocessing.map import coupled_map
from itertools import cycle
from pathlib import Path
from src.preprocessing.transform import transform_coordinates


os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)
os.environ["OPEN3D_QT_BACKEND"] = "PyQt5"

# Global parameters
goal_from_picture_x = 8    # Target X
goal_from_picture_y = 10    # Target Y
sampling_interval = 0.1

def get_file_paths(directory):
    filenames = os.listdir(directory)
    full_paths = [os.path.join(directory, filename) for filename in filenames if os.path.isfile(os.path.join(directory, filename))]
    sorted_paths = natsorted(full_paths)
    return sorted_paths

def standardize_scores(data):
    # Normalize scores assigned to every path using StandardScaler
    data = np.array(data).reshape(-1, 1)  # Reshape for StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data.flatten()  # Flatten back to 1D array

def min_max_normalize(data):
# normalizing scores assigned to every path
    data = np.array(data)

    max_data = max(data)
    min_data = min(data)

    if max_data - min_data == 0:
        data = [0.0 for i in range(len(data))]
    else:
        data = (data - min_data) / (max_data - min_data)

    return data

def angle_range_corrector(angle):
    if angle > math.pi:
        while angle > math.pi:
            angle -= 2 * math.pi
    elif angle < -math.pi:
        while angle < -math.pi:
            angle += 2 * math.pi
    return angle

class Path:
    def __init__(self, u_th, u_v):
        self.x = None
        self.y = None
        self.th = None
        self.u_v = u_v
        self.u_th = u_th

class Obstacle:
    def __init__(self, x, y, size=0.25):
        self.x = x
        self.y = y
        self.size = size

class Two_wheeled_robot:
    def __init__(self, init_x, init_y, init_th):
        self.x = init_x  
        self.y = init_y
        self.th = init_th
        self.u_v = 0.0
        self.u_th = 0.0
        self.traj_x = [init_x]
        self.traj_y = [init_y]
        self.traj_th = [init_th]

    def update_state(self, u_th, u_v, dt):
        self.u_th = u_th
        self.u_v = u_v
        self.x += self.u_v * np.cos(self.th) * dt
        self.y += self.u_v * np.sin(self.th) * dt
        self.th += self.u_th * dt
        self.th = angle_range_corrector(self.th)
        self.traj_x.append(self.x)
        self.traj_y.append(self.y)
        self.traj_th.append(self.th)
        return self.x, self.y, self.th

class Const_goal:
    def __init__(self):
        self.traj_g_x = []
        self.traj_g_y = []

    def calc_goal(self, time_step):
        g_x = goal_from_picture_x 
        g_y = goal_from_picture_y
        self.traj_g_x.append(g_x)
        self.traj_g_y.append(g_y)
        return g_x, g_y

class Simulator_DWA_robot:
    def __init__(self):
        self.max_accelation = 1
        self.max_ang_accelation = 100 * math.pi / 180
        self.lim_max_velo = 0.6
        self.lim_min_velo = 0.0
        self.lim_max_ang_velo = 0.2  # Increased for better turning
        self.lim_min_ang_velo = -0.2  # Increased for better turning

    def predict_state(self, ang_velo, velo, x, y, th, dt, pre_step):
        next_xs = []
        next_ys = []
        next_ths = []
        for i in range(pre_step):
            temp_x = velo * math.cos(th) * dt + x 
            temp_y = velo * math.sin(th) * dt + y
            temp_th = ang_velo * dt + th
            temp_th = angle_range_corrector(temp_th)
            next_xs.append(temp_x)
            next_ys.append(temp_y)
            next_ths.append(temp_th)
            x = temp_x
            y = temp_y
            th = temp_th
        return next_xs, next_ys, next_ths

class DWA:
    def __init__(self, coupled_cost_map):
        global sampling_interval
        self.pre_step = 30
        self.pre_time = sampling_interval * self.pre_step
        self.delta_velo = 0.02
        self.delta_ang_velo = 0.02  # Increased for wider angular sampling
        self.weight_end_dist = 2.5  # Increased to prioritize goal
        self.weight_surr = 0.2    # Reduced to deprioritize surroundings
        self.weight_obs = 1.5      # Increased to avoid obstacles more
        self.area_dis_to_obs = 2
        self.traj_paths = []
        self.traj_opt = []
        self.obstacles = []
        self.simu_robot = Simulator_DWA_robot()
        self.coupled_cost_map = coupled_cost_map

    def calc_input(self, g_x, g_y, state, obstacles):
        paths = self._make_path(state) 
        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles) 
        self.traj_opt.append(opt_path) 
        return paths, opt_path

    def _make_path(self, state): 
        min_ang_velo, max_ang_velo, min_velo, max_velo = self._calc_range_velos(state) 
        paths = []
        for ang_velo in np.arange(min_ang_velo, max_ang_velo, self.delta_ang_velo):
            for velo in np.arange(min_velo, max_velo, self.delta_velo):
                path = Path(ang_velo, velo)
                next_x, next_y, next_th = self.simu_robot.predict_state(ang_velo, velo, state.x, state.y, state.th, sampling_interval, self.pre_step)
                path.x = next_x 
                path.y = next_y
                path.th = next_th
                paths.append(path)
        self.traj_paths.append(paths) 
        return paths

    def _calc_range_velos(self, state):
        range_ang_velo = sampling_interval * self.simu_robot.max_ang_accelation 
        min_ang_velo = max(self.simu_robot.lim_min_ang_velo, state.u_th - range_ang_velo)
        max_ang_velo = min(self.simu_robot.lim_max_ang_velo, state.u_th + range_ang_velo)
        range_velo = sampling_interval * self.simu_robot.max_accelation 
        min_velo = max(self.simu_robot.lim_min_velo, state.u_v - range_velo)
        max_velo = min(self.simu_robot.lim_max_velo, state.u_v + range_velo)
        return min_ang_velo, max_ang_velo, min_velo, max_velo

    def _eval_path(self, paths, g_x, g_y, state, obstacles):
        nearest_obs = self._calc_nearest_obs(state, obstacles)
        score_dist_end = []
        score_surr = []
        score_obstacles = []

        for path in paths:
            end_dist = self._goal_distance(path, g_x, g_y)
            score_dist_end.append(end_dist)
            score_surr.append(self._calc_surr_score(path))
            score_obstacles.append(self._obstacle(path, nearest_obs))

        score_dist_end  = standardize_scores(score_dist_end)
        score_surr  = standardize_scores(score_surr)
        score_obstacles  = standardize_scores(score_obstacles)
        # Normalize scores
        # for scores in [score_dist_end, score_surr, score_obstacles]:
        #     scores = min_max_normalize(scores)

        score = float('inf')
        opt_path = None

        for k in range(len(paths)):
            # Invert end_dist score (lower distance = better)
            total_score = (self.weight_end_dist * score_dist_end[k] + 
                          self.weight_surr * score_surr[k] + 
                          self.weight_obs * score_obstacles[k])
            if total_score < score:
                score = total_score
                opt_path = paths[k]

        if opt_path is None:
            print("No optimal path found!")
            return paths[0] if paths else None
        return opt_path

    def _goal_distance(self, path, g_x, g_y):
        last_x = path.x[-1] 
        last_y = path.y[-1]
        return np.sqrt((g_y - last_y)**2 + (g_x - last_x)**2)
    
    def _calc_surr_score(self, path):
        if self.coupled_cost_map is None:
            return 0
        projected_path_scaled = self._project_path(path)
        cardinality = len(projected_path_scaled)
        cost = 0
        for (y, x) in projected_path_scaled:
            if 0 <= x < self.coupled_cost_map.shape[1] and 0 <= y < self.coupled_cost_map.shape[0]:
                cost += float(self.coupled_cost_map[int(y), int(x)])
        return cost / cardinality if cardinality > 0 else 0
    
    def _project_path(self, path):
        
        x_coords = path.x
        y_coords = path.y
        
        projected_path = transform_coordinates(x_coords, y_coords)
        return projected_path
    

    def _calc_nearest_obs(self, state, obstacles):
        nearest_obs = []
        for obs in obstacles:
            temp_dis_to_obs = math.sqrt((state.x - obs.x) ** 2 + (state.y - obs.y) ** 2)
            if temp_dis_to_obs < self.area_dis_to_obs:
                nearest_obs.append(obs)
        return nearest_obs

    def _obstacle(self, path, nearest_obs):

        if not nearest_obs:
            return 0  # No obstacles detected, assign score_obstacle as 0
        
        # obstacle avoidance
        score_obstacle = 5
        temp_dis_to_obs = 0.0

        for i in range(len(path.x)):
            for obs in nearest_obs: 
                temp_dis_to_obs = math.sqrt((path.x[i] - obs.x) * (path.x[i] - obs.x) + (path.y[i] - obs.y) *  (path.y[i] - obs.y))

                if temp_dis_to_obs < score_obstacle:
                    score_obstacle = temp_dis_to_obs #the nearest obstacle

                # collision with obstacle
                if temp_dis_to_obs < obs.size + 0.05: #0.75 is the margin
                    # score_obstacle = -float('inf')
                    score_obstacle = -float(1e6)
                    break
            else:
                continue
            
            break

        # print(f"obs:{score_obstacle}")    
        return 1/score_obstacle

class DynamicObstacleManager:
    def __init__(self, point_cloud_paths, mask_paths, image_paths):
        self.point_cloud_paths = cycle(point_cloud_paths)
        self.mask_paths = cycle(mask_paths)
        self.image_paths = cycle(image_paths)
        self.active_obstacles = []  # Accumulate all obstacles
        self.current_map = None
        self.load_initial_data()
    
    def load_initial_data(self):
        file_path = next(self.point_cloud_paths)
        mask_path = next(self.mask_paths)
        image_path = next(self.image_paths)
        obstacle_tuples = process_point_cloud(file_path)
        self.current_map = coupled_map(image_path, mask_path)
        self.active_obstacles = [Obstacle(x, y) for x, y, _ in obstacle_tuples]
    
    def update_obstacles_and_cost_map(self, robot_x, robot_y):
        if all(robot_x > obs.x and robot_y > obs.y for obs in self.active_obstacles):
            file_path = next(self.point_cloud_paths)
            mask_path = next(self.mask_paths)
            image_path = next(self.image_paths)
            obstacle_tuples = process_point_cloud(file_path)
            self.current_map = coupled_map(image_path, mask_path)
            # Add new obstacles without clearing old ones
            new_obstacles = [Obstacle(robot_x + x, robot_y + y) for x, y, _ in obstacle_tuples]
            self.active_obstacles.extend(new_obstacles)
        
    def get_obstacles(self):
        return self.active_obstacles
    
    def get_cost_map(self):
        return self.current_map

class Main_controller:
    def __init__(self, point_cloud_files, mask_files, image_files):
        self.robot = Two_wheeled_robot(0.0, 0.0, 0.0)  # Start at (0,0)
        self.goal_maker = Const_goal()
        self.obstacle_manager = DynamicObstacleManager(point_cloud_files, mask_files, image_files)
        self.obstacles = self.obstacle_manager.get_obstacles()
        self.controller = DWA(self.obstacle_manager.get_cost_map())
        self.new_goal_flag = False

    def run_to_goal(self, time_step, goal_flag):
        global sampling_interval
        
        if not goal_flag:
            g_x, g_y = self.goal_maker.calc_goal(time_step)
            paths, opt_path = self.controller.calc_input(g_x, g_y, self.robot, self.obstacles)
            
            if opt_path:
                u_th, u_v = opt_path.u_th, opt_path.u_v
            else:
                u_th, u_v = 0, 0
                print("No path available, stopping")
                return True

            self.robot.update_state(u_th, u_v, sampling_interval)
            self.obstacle_manager.update_obstacles_and_cost_map(self.robot.x, self.robot.y)
            self.obstacles = self.obstacle_manager.get_obstacles()
            self.controller = DWA(self.obstacle_manager.get_cost_map())
            
            dis_to_goal = np.sqrt((g_x - self.robot.x)**2 + (g_y - self.robot.y)**2)
            if dis_to_goal < 0.5:
                self.new_goal_flag = True
                print(f"Goal reached at ({self.robot.x}, {self.robot.y})!")

            window.draw(self.robot.traj_x, self.robot.traj_y, self.robot.traj_th, 
                       self.goal_maker.traj_g_x, self.goal_maker.traj_g_y, 
                       self.controller.traj_paths, self.controller.traj_opt, 
                       self.obstacles)
        return self.new_goal_flag

class Simulation_Window(QDialog):
    time_step = 0
    goal_flag = False
    pcd_directory = "/home/srivabhi22/Desktop/AMCO/data/point_cloud"
    masks_directory = "/home/srivabhi22/Desktop/AMCO/data/masks"
    images_directory = "/home/srivabhi22/Desktop/AMCO/data/image"
    
    controller = Main_controller(get_file_paths(pcd_directory), 
                               get_file_paths(masks_directory), 
                               get_file_paths(images_directory))
    timer = QTimer()

    def __init__(self, parent=None):
        super(Simulation_Window, self).__init__(parent)
        self.ui = Ui_DWA_Simulator()
        self.ui.setupUi(self)

        self.ui.pre_time_spinBox.setValue(3)
        self.ui.pre_step_spinBox.setValue(30)
        self.ui.vel_delta_SpinBox.setValue(0.02)
        self.ui.ang_vel_delta_SpinBox.setValue(0.04)
        self.ui.sampling_interval_SpinBox.setValue(0.1)
        self.ui.goal_weight_SpinBox.setValue(2.0)
        self.ui.surr_weight_SpinBox.setValue(0.001)
        self.ui.obstacle_weight_SpinBox.setValue(5.0)
        self.ui.area_dis_to_obs_SpinBox.setValue(2)
        self.ui.Max_Acc_SpinBox.setValue(1)
        self.ui.Max_Ang_Acc_SpinBox.setValue(100*math.pi/180)
        self.ui.Max_Vel_SpinBox.setValue(0.6)
        self.ui.Min_Vel_SpinBox.setValue(0.0)
        self.ui.Max_Ang_Vel_SpinBox.setValue(0.4)
        self.ui.Min_Ang_Vel_SpinBox.setValue(-0.4)
        self.ui.Possible_passes_spinBox.setValue(8)

    def do_calculations(self):
        global sampling_interval

        if not self.goal_flag:
            self.goal_flag = self.controller.run_to_goal(self.time_step, self.goal_flag)
            self.time_step += 1

        if self.goal_flag:
            self.timer.stop()
            print("Simulation stopped: Goal reached or no path available")

        self.controller.controller.pre_time = self.ui.pre_time_spinBox.value()
        self.controller.controller.pre_step = self.ui.pre_step_spinBox.value()
        self.controller.controller.delta_velo = self.ui.vel_delta_SpinBox.value()
        self.controller.controller.delta_ang_velo = self.ui.ang_vel_delta_SpinBox.value()
        sampling_interval = self.ui.sampling_interval_SpinBox.value()
        self.controller.controller.weight_end_dist = self.ui.goal_weight_SpinBox.value()
        self.controller.controller.weight_surr = self.ui.surr_weight_SpinBox.value()
        self.controller.controller.weight_obs = self.ui.obstacle_weight_SpinBox.value()
        self.controller.controller.area_dis_to_obs = self.ui.area_dis_to_obs_SpinBox.value()
        self.controller.controller.simu_robot.max_accelation = self.ui.Max_Acc_SpinBox.value()
        self.controller.controller.simu_robot.max_ang_accelation = self.ui.Max_Ang_Acc_SpinBox.value()
        self.controller.controller.simu_robot.lim_max_velo = self.ui.Max_Vel_SpinBox.value()
        self.controller.controller.simu_robot.lim_min_velo = self.ui.Min_Vel_SpinBox.value()
        self.controller.controller.simu_robot.lim_max_ang_velo = self.ui.Max_Ang_Vel_SpinBox.value()
        self.controller.controller.simu_robot.lim_min_ang_velo = self.ui.Min_Ang_Vel_SpinBox.value()

    def pause(self):
        self.timer.stop()

    def reset(self):
        self.controller.robot.traj_x.clear()
        self.controller.robot.traj_y.clear()
        self.controller.robot.traj_th.clear()
        self.controller.goal_maker.traj_g_x.clear()
        self.controller.goal_maker.traj_g_y.clear()
        self.controller.controller.traj_paths.clear()
        self.controller.controller.traj_opt.clear()
        self.controller.new_goal_flag = False
        self.controller.goal_flag = False
        self.goal_flag = False
        self.time_step = 0

        self.controller.robot.x = 0.0  # Reset to (0,0)
        self.controller.robot.y = 0.0
        self.controller.robot.th = 0.0
        self.controller.robot.u_v = 0.0
        self.controller.robot.u_th = 0.0
        self.controller.robot.traj_x = [0.0]
        self.controller.robot.traj_y = [0.0]
        self.controller.robot.traj_th = [0.0]

        self.controller.obstacle_manager.load_initial_data()
        self.controller.obstacles = self.controller.obstacle_manager.get_obstacles()
        self.controller.controller = DWA(self.controller.obstacle_manager.get_cost_map())

    def Start_simulation(self):
        self.timer.timeout.connect(self.do_calculations)
        self.timer.start(50)

    def draw(self, traj_x, traj_y, traj_th, goal_x, goal_y, traj_paths, traj_opt, obstacles):
        scale = 0.025
        C = 1 / scale
        robot_x = traj_x[-1]
        robot_y = traj_y[-1]
        X_offset = 300
        Y_offset = 300
        scene_min_x = robot_x - X_offset * scale
        scene_max_x = robot_x + X_offset * scale
        scene_min_y = robot_y - Y_offset * scale
        scene_max_y = robot_y + Y_offset * scale

        self.scene = GraphicsScene()
        self.scene.setSceneRect(scene_min_x * C, -scene_max_y * C, 
                              (scene_max_x - scene_min_x) * C, 
                              (scene_max_y - scene_min_y) * C)
        self.ui.graphicsView.setScene(self.scene)

        pen_axis = QPen(Qt.black)
        pen_axis.setWidth(1)
        pen_axis.setStyle(Qt.DashLine)
        grid_step = 10
        start_x = int(scene_min_x // grid_step) * grid_step
        end_x = int(scene_max_x // grid_step) * grid_step
        start_y = int(scene_min_y // grid_step) * grid_step
        end_y = int(scene_max_y // grid_step) * grid_step

        for x in range(start_x, end_x + 1, grid_step):
            self.scene.addLine(QLineF(x * C, -scene_min_y * C, x * C, -scene_max_y * C), pen_axis)
        for y in range(start_y, end_y + 1, grid_step):
            self.scene.addLine(QLineF(scene_min_x * C, -y * C, scene_max_x * C, -y * C), pen_axis)

        pen_traj = QPen(Qt.blue)
        pen_traj.setStyle(Qt.DashLine)
        for i in range(len(traj_x) - 1):
            self.scene.addLine(QLineF(C * traj_x[i], -C * traj_y[i], 
                                    C * traj_x[i + 1], -C * traj_y[i + 1]), pen_traj)

        pen_obstacle = QPen(QColor(200, 0, 150))
        for obs in obstacles:
            obstacle_diameter = obs.size * C
            self.scene.addEllipse(C * obs.x - obstacle_diameter / 2, 
                                -C * obs.y - obstacle_diameter / 2, 
                                obstacle_diameter, obstacle_diameter, 
                                pen_obstacle, QBrush(QColor(200, 0, 150)))

        X = traj_x[-1]
        Y = traj_y[-1]
        th = traj_th[-1] - math.pi / 2
        robot_vertices = [[C * (X + (-0.25 * math.sin(th))), -C * (Y + (0.25 * math.cos(th)))], 
                         [C * (X + (-0.2 * math.cos(th) - 0.15 * math.sin(th))), -C * (Y + (-0.2 * math.sin(th) + 0.15 * math.cos(th)))], 
                         [C * (X + (-0.2 * math.cos(th) - (-0.15) * math.sin(th))), -C * (Y + (-0.2 * math.sin(th) - 0.15 * math.cos(th)))],
                         [C * (X + (0.2 * math.cos(th) - (-0.15) * math.sin(th))), -C * (Y + (0.2 * math.sin(th) - 0.15 * math.cos(th)))], 
                         [C * (X + (0.2 * math.cos(th) - 0.15 * math.sin(th))), -C * (Y + (0.2 * math.sin(th) + 0.15 * math.cos(th)))]]
        qpoly_robot = QPolygonF([QPointF(p[0], p[1]) for p in robot_vertices])
        pen_robot = QPen(Qt.red)
        pen_robot.setWidth(2)
        self.scene.addPolygon(qpoly_robot, pen_robot)

        pen_goal = QPen(Qt.blue)
        diameter = 16
        self.scene.addEllipse(C * goal_x[-1] - diameter / 2, 
                            -C * goal_y[-1] - diameter / 2, 
                            diameter, diameter, pen_goal, QBrush(Qt.green))

class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, 0, 0, 600, 600, parent=None)
        self.opt = ""
        
    def mousePressEvent(self, event):
        global goal_from_picture_x, goal_from_picture_y
        goal_from_picture_x = (event.scenePos().x() - 100)*0.025
        goal_from_picture_y = -(event.scenePos().y() - 500)*0.025

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Simulation_Window()
    window.show()
    sys.exit(app.exec_())
