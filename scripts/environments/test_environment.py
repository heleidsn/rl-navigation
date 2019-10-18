import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class TestEnv():
    def __init__(self):

        self.topic_num = 0
        self.pose_data = Pose()
        self.stalled = False

        rospy.loginfo("Initializing Robot...")
        rospy.init_node('ros_node')
        rospy.loginfo("Node Created")

        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 1) ##Set queue size
        rospy.loginfo("Publisher Created: /cmd_vel")
        ##Added for single laser(stage)
        rospy.loginfo("Setting Subscribers...")

        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.update_robot_state_data)
        rospy.Subscriber("/stalled", Int8, self.update_robot_stall_data)

        rospy.loginfo("Topic Subscribed: /base_pose_ground_truth")

        rate = rospy.Rate(1) # 10hz
        while not rospy.is_shutdown():
            print(rospy.Time.now(), rospy.Duration(0.2))
            rate.sleep()


    def update_robot_state_data(self, msg):
        all_data = msg
        self.pose_data = all_data.pose.pose
        self.topic_num += 1
        # print(self.topic_num, self.pose_data)
    
    def update_robot_stall_data(self, msg):
        self.stalled = msg.data
        # print(rospy.time)
        # print(rospy.Time.now(), self.stalled)

def main():
    env = TestEnv()

if __name__ == '__main__':
    main()
    