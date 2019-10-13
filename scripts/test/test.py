#!/usr/bin/env python

from environments.base_environment import Environment
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D

import numpy as np
import os
import rospkg
import rospy
from std_srvs.srv import Empty


def reset_env():
    rospy.wait_for_service('/reset_positions')
    try:
        rospy.ServiceProxy('/reset_positions', Empty)
        print('reset success')
        # print(val)
    except rospy.ServiceException:
        print('Service call failed')

def main():
    a = True
    b = 0
    b += a
    b += a
    c = False
    b += c

    str = 'test: {:.2f}'.format(10.1111)

    print(str)


if __name__ == '__main__':
    main()
    





