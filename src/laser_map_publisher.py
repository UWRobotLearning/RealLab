#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from utils.scan_utils import generate_local_obstacle_map

class ObstacleMapNode(object):
    def __init__(self):
        rospy.init_node('obstacle_map_node')

        # Parameters: resolution in meters per cell and map size in meters (square map)
        self.resolution = rospy.get_param("~resolution", 0.1)
        self.size = rospy.get_param("~size", 3.1)

        self.bridge = CvBridge()

        # Publisher for the obstacle map as an image message
        self.image_pub = rospy.Publisher("obstacles/image", Image, queue_size=1)
        self.map_pub = rospy.Publisher("obstacles/map", Float32MultiArray, queue_size=1)

        # Subscriber for LaserScan messages
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)

    def scan_callback(self, scan_msg):
        # Convert the LaserScan message into a dict that our generator expects.
        scan_data = {
            "ranges": np.array(list(scan_msg.ranges)),
            "angle_min": scan_msg.angle_min,
            "angle_increment": scan_msg.angle_increment,
            "range_min": scan_msg.range_min,
            "range_max": scan_msg.range_max
        }
        
        # Generate the obstacle map as a numpy array.
        obstacle_map = generate_local_obstacle_map(scan_data, self.resolution, self.size)

        map_msg = Float32MultiArray()
        map_msg.data = obstacle_map.flatten().tolist()
        self.map_pub.publish(map_msg)

        # Optionally, you can apply image processing or colormaps here.
        # For simplicity, we assume a single-channel (grayscale) image where
        # obstacles are white (255) and free space is black (0).
        try:
            obstacle_map = np.where(obstacle_map>0.1, 255, 0).astype(np.uint8)
            obstacle_map = np.flipud(obstacle_map)
            image_msg = self.bridge.cv2_to_imgmsg(obstacle_map, encoding="mono8")
            self.image_pub.publish(image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: {}".format(e))    

if __name__ == '__main__':
    node = ObstacleMapNode()
    rospy.loginfo("Obstacle map node started. Listening to /scan...")
    rospy.spin()
