#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from nav_msgs.msg import Odometry, Path as navPath
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu, Image, Joy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from utils.rl_policy import RLModel
from utils.waypoints import Waypoints
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import euler_from_quaternion

import os
from pathlib import Path
import yaml
import time
import torch
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from utils.generate_elevation_map import crop_heightmap


class Hound_RLHL_Control:
    def __init__(self, policy, data_collection):
        with open(f"/root/catkin_ws/src/hound_core/config/policies/{policy}.yaml") as f:
            config_data = yaml.safe_load(f)

        self.throttle_to_wheelspeed = config_data["throttle_to_wheelspeed"]
        self.steering_max = config_data["steering_max"]
        self.rate = config_data["rate"]
        self.obs_type = config_data["obs_type"]
        hidden_shape = config_data["hidden_shape"]
        model_path = config_data["model_path"]
        model_type = config_data["model_type"]

        ## state variables
        self.state_init = False
        self.imu = None
        self.odom_update = False
        self.pose = torch.zeros(6)
        self.twists = torch.zeros(6)
        self.start_action = False
        self.pad_latch = True

        if self.obs_type == "relative":
            self.state = np.zeros(12, dtype=np.float32)
            self.model = RLModel(model_path, 
                                 type=model_type,
                                 acargs=(12,12,2),
                                 ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = False
        elif self.obs_type == "blind":
            self.state = np.zeros(14, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(14,14,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 12
        elif self.obs_type == "elevation":
            self.state = np.zeros(972, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(972,972,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 9
            self.heightmap = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap2.npy")
            self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)
        elif self.obs_type == "goal_based_elevation":
            self.state = np.zeros(975, dtype=np.float32)
            self.model = RLModel(model_path,
                                type=model_type,
                                acargs=(975,975,2),
                                ackwargs={
                                   "actor_hidden_dims": hidden_shape,
                                   "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.last_action_offset = 12
            self.heightmap = np.load("/root/catkin_ws/src/hound_core/config/elevation/heightmap2.npy")
            self.heightmap_sub = rospy.Subscriber("/heightmap", Float32MultiArray, self.heightmap_callback)
            self.goal = np.array(config_data["goal"], dtype=np.float32)
        elif self.obs_type == 'rgb':
            self.state = np.zeros(40 * 80 + 8, dtype=np.float32)
            self.model = RLModel(model_path, 
                                type=model_type,
                                acargs = (40 * 80 + 8, 40 * 80 + 8, 2),
                                ackwargs={
                                      "actor_hidden_dims": hidden_shape,
                                      "critic_hidden_dims": hidden_shape
                                })
            self.include_last_action = True
            self.cv_bridge = CvBridge()
            self.image_shape = (40, 80)
            self.resize_shape = (80, 60)
            self.image = np.zeros(self.image_shape[0]*self.image_shape[1])
            self.last_action_offset = 40 * 80 + 6
            self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, callback_args={"resize_shape": self.resize_shape})
            self.threshold = config_data["threshold"]
        else:
            ValueError("must choose valid obs type")

        self.collect_data = data_collection

        if data_collection:
            self.value_pub = rospy.Publisher("value", Float32MultiArray, queue_size=1)

        waypoints = Waypoints()
        waypoints.generate_waypoints()
        print("\n1\n")
        # initialize the odometry and imu subscribers with callbacks
        self.odom_sub = rospy.Subscriber(
            "/car/vesc/odom", Odometry, self.odom_callback
        )

        self.rc_sub = rospy.Subscriber('/car/teleop/joy', Joy, self.rcin_callback)

        self.imu_sub = rospy.Subscriber("/camera/gyro/sample", Imu, self.imu_callback)
        # self.grid_map_sub = rospy.Subscriber(
        #     "/grid_map_occlusion_inpainting/all_grid_map",
        #     GridMap,
        #     self.grid_map_callback,
        # )
        # self.path_sub = rospy.Subscriber(
        #     "path",
        #     navPath,
        #     self.path_callback,
        #     queue_size=10,
        # )
        self.ctrl_limits_sub = rospy.Subscriber(
            "/control_limits",
            AckermannDriveStamped,
            self.limits_callback,
            queue_size=10,
        )
        self.hard_limit = 0.3
        self.large_dt = False

        ## set up publishers:
        self.control_pub = rospy.Publisher(
            "/car/mux/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1
        )
        self.state_pub = rospy.Publisher(
            "hl_controller/state", Float32MultiArray, queue_size=1
        )

        self.marker_pub = rospy.Publisher("marker", MarkerArray, queue_size=1)
        self.reset_pub = rospy.Publisher(
            "/simulation_reset", AckermannDriveStamped, queue_size=2
        )
        time.sleep(1)
        reset_msg = AckermannDriveStamped()
        self.reset_pub.publish(reset_msg)
        time.sleep(1)
        ## initialize controller:
        print("\n2\n")
        self.main_loop()

    def limits_callback(self, msg):
        self.hard_limit = msg.drive.speed

    def rcin_callback(self, data):
        try:
            self.start_action = data.buttons[5] == 1
        except Exception as e:
            pass

    def main_loop(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if (self.state_init and self.odom_update):
                ctrl = self.model.inference(self.state)
                msg = Float32MultiArray()
                msg.data = self.state.tolist()
                self.state_pub.publish(msg)
                if self.collect_data and self.start_action:
                    msg = Float32MultiArray()
                    data = np.zeros(15, dtype=np.float32)
                    data[14] = self.model.get_value(self.state).tolist()[0]
                    data[0:6] = self.pose.numpy()
                    data[6:12] = self.twists.numpy()
                    data[12:14] = ctrl
                    msg.data = data.tolist()
                    self.value_pub.publish(msg)
                    self.pad_latch = True
                elif self.collect_data and self.pad_latch:
                    msg = Float32MultiArray()
                    msg.data = np.zeros(15, dtype=np.float32).tolist()
                    self.value_pub.publish(msg)
                    self.pad_latch = False
                    
                self.send_ctrl(ctrl)
                self.odom_update = False
            rate.sleep()

    def send_ctrl(self, ctrl):
        control_msg = AckermannDriveStamped()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.header.frame_id = "base_link"
        control_msg.drive.steering_angle = -(ctrl[1] * self.steering_max)
        control_msg.drive.speed = 0.5 #if (ctrl[0] * self.throttle_to_wheelspeed) > 0 else 0
        if not self.start_action:
            control_msg.drive.speed = 0
        if self.include_last_action:
            if self.start_action:
                self.state[self.last_action_offset] = ctrl[0]
                self.state[self.last_action_offset + 1] = ctrl[1]
            else:
                self.state[self.last_action_offset] = 0.0
                self.state[self.last_action_offset + 1] = 0.0
        self.control_pub.publish(control_msg)

    def obtain_state(self, odom):
        ## obtain the state from the odometry and imu messages:
        new_pose = torch.zeros(6)
        quaternion = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        )

        rpy = euler_from_quaternion(quaternion)
        new_pose[0] = odom.pose.pose.position.x
        new_pose[1] = odom.pose.pose.position.y
        new_pose[2] = odom.pose.pose.position.z

        new_pose[3] = (rpy[0] + 2*np.pi) % (2*np.pi)
        new_pose[4] = (rpy[1] + 2*np.pi) % (2*np.pi)
        new_pose[5] = (rpy[2] + 2*np.pi) % (2*np.pi)

        self.pose = new_pose

        self.twists[0] = odom.twist.twist.linear.x
        self.twists[1] = odom.twist.twist.linear.y
        self.twists[2] = odom.twist.twist.linear.z
        # lazy fix for wierd camera reference frame
        self.twists[3] = self.imu.angular_velocity.z
        self.twists[4] = - self.imu.angular_velocity.x
        self.twists[5] = - self.imu.angular_velocity.y

        if self.obs_type == "relative":
            self.obtain_relative_state(odom)
        elif self.obs_type == "blind":
            self.obtain_blind_state(odom)
        elif self.obs_type == "elevation":
            self.obtain_elevation_state(odom)
        elif self.obs_type == "goal_based_elevation":
            self.obtain_goal_based_elevation_state(odom)
        elif self.obs_type == "rgb":
            self.obtain_rgb_state(odom)
        else:
            ValueError("must choose valid obs type")

    def obtain_blind_state(self, odom):
        self.state[:6] = self.pose.numpy()
        self.state[6:12] = self.twists.numpy()

    def obtain_relative_state(self, odom):
        self.state[:6] = self.pos_angle(self.pose).numpy()
        self.state[6:12] = self.twists.numpy()

    def get_local_elevation_map(self, x, y, yaw, width=20):
        elevation_map = crop_heightmap(self.heightmap, x, y, yaw, width=width)
        return elevation_map.flatten()
    
    def obtain_elevation_state(self, odom):
        #Obtain state for elevation policy
        x=self.pose[0].numpy()
        y=self.pose[1].numpy()
        yaw = self.pose[5].numpy()
        self.state[:3] = self.pose[3:6].numpy() #obtain the orientation
        self.state[3:9] = self.twists.numpy()
        #TODO: in the observation term I also have the last action term, how do I include it here?  
        self.state[11:972] = self.get_local_elevation_map(x, y, yaw, width=31) # this should be an array of shape (N,) where N = size*size, here it will be 400(taking 20 as the size)

    def obtain_goal_based_elevation_state(self, odom):
        x=self.pose[0].numpy()
        y=self.pose[1].numpy()
        yaw = self.pose[5].numpy()
        self.state[:3] = self.goal - self.pose[:3].numpy()
        self.state[3:6] = self.pose[3:6].numpy() #obtain the orientation
        self.state[6:12] = self.twists.numpy()
        #TODO: in the observation term I also have the last action term, how do I include it here?  
        self.state[14:975] = self.get_local_elevation_map(x, y, yaw, width=31)

    def obtain_rgb_state(self, odom):
        image_offset = self.image_shape[0] * self.image_shape[1]
        self.state[:image_offset] = self.image
        self.state[image_offset:image_offset+6] = self.twists.numpy()

    def image_callback(self, msg, callback_args):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            raise RuntimeError(e)
        # h x w x c
        resize_shape = callback_args["resize_shape"]
        resized_image = cv2.resize(image, resize_shape)
        H = resized_image.shape[0]

        # remove upper 1/3
        resized_image = resized_image[H//3:, ...]

        norm = np.linalg.norm(resized_image - np.array([255, 255, 255]), ord=1, axis=-1)
        gray_image = np.zeros_like(resized_image[...])
        gray_image[norm < 100] = resized_image[norm < 100]
        gray_image[norm >= 100] = np.random.uniform(0, 30, size=gray_image[norm >= 100].shape)

        # convert to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) / 255.

        # make it unit gaussian assuming mean std of 0.5 0.5

        flattened_image = gray_image.reshape(-1)
        if self.threshold > 0:
           flattened_image = flattened_image > self.threshold
        else:
           flattened_image = self.image = flattened_image

        self.image = (flattened_image - 0.5) / 0.5

    def odom_callback(self, odom):
        if self.imu is None:
            return
        self.obtain_state(odom)
        if not self.state_init:
            self.state_init = True
        self.odom_update = True  ## indicate that a new reading is available

    def imu_callback(self, imu):
        self.imu = imu

    def heightmap_callback(self, msg):
        self.heightmap = np.array(msg.data).reshape(self.heightmap.shape)

    def pos_angle(self, pos):
        waypoints = Waypoints().waypoints
        waypoints = waypoints.to(pos.device)
        distances = torch.norm(waypoints - pos[:3], dim=-1)
        closest_index = torch.argmin(distances)
        goal_index = (closest_index+1)%len(waypoints)
        goal_waypoint = waypoints[goal_index.unsqueeze(0)].squeeze(0)
        goal_angle = torch.atan2(goal_waypoint[1] - pos[1], goal_waypoint[0] - pos[0])
        goal_euler = torch.cat([torch.zeros(2, device=pos.device), goal_angle.unsqueeze(0)])
        return torch.cat([goal_waypoint, goal_euler]) - pos


if __name__ == "__main__":
    rospy.init_node("hl_controller")
    policy = rospy.get_param("~policy")
    data_collection = rospy.get_param("~data_collection")
    planner = Hound_RLHL_Control(policy, data_collection)
    rospy.spin()
