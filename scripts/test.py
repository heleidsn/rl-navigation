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
import math


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

    reset_env()
    
    while True:
        # next_state = environment.get_network_state()
        # print(next_state)
        next_state = environment.get_network_state()
        print(next_state)
        time.sleep(1)

def test():
    list_a = [1, 2, 3]
    list_b = [2, 3, 4]
    for index, (a, b) in enumerate(zip(list_a, list_b)):
        print(index, a, b)

    list_aa = []

    list_aa.append([1, 2])
    list_aa.append([3, 4])

    print(list_aa[0], list_aa[1])

    print(max(list_aa[1]))

    goal = [[-1,9,0], [0,0,0]]
    print(goal[0][1])


if __name__ == '__main__':
    #  main()
    test()
    





