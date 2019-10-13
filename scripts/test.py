#!/usr/bin/env python

from environments.ddpg_environment import Environment
from options.option_ddpg import Options
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D

import numpy as np
import os
import rospkg
import rospy
from std_srvs.srv import Empty

from utils.common import *
from utils.map import *

import time

def reset_env():
    rospy.wait_for_service('/reset_positions')
    try:
        rospy.ServiceProxy('/reset_positions', Empty)
        print('reset success')
        # print(val)
    except rospy.ServiceException:
        print('Service call failed')

def main():
    args = Options().parse()

    # get obstacle and free map
    map_size = args.map_size
    map_resolution = args.map_resolution
    obstacles_map = args.obstacles_map
    obstacle_padding = args.obstacle_padding

    obstacle_positions = get_obstacle_positions(map_size, obstacles_map)
    obstacles_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, obstacle_padding)

    n_states = 38
    # create environment
    
    environment = Environment(args, n_states)
    environment.set_obstacles_map(obstacles_map, map_resolution)
    num = 0
    while True:
        # next_state = environment.get_network_state()
        # print(next_state)
        next_state = environment.get_network_state()
        print(next_state)
        time.sleep(1)



if __name__ == '__main__':
    main()
    





