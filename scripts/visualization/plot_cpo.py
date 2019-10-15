#!/usr/bin/env python
"""
For every point and rotation in map
Get the state, action and Q-value 
"""

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import cPickle as pickle

import os.path

import time

import tensorflow as tf
import numpy as np
import rospy
import rospkg
import math

from datetime import datetime

from environments.base_environment import Environment
from utils.memory import ExperienceBuffer
from algo.cpo import Actor, Critic, SafetyBaseline
from utils.common import *
from utils.map import *
from options.option_cpo import Options
from std_srvs.srv import Empty

def main():
    args = Options().parse()

    tf.reset_default_graph()

    sess = tf.Session()

    base_path = rospkg.RosPack().get_path("reinforcement_learning_navigation")
    summary_filename = args.output_name

    # Set up folder structure
    date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    storage_path = os.path.join(base_path, "logs", date_str + "_" + summary_filename)
    weights_path = os.path.join(storage_path, "weights")
    tensorboard_path = os.path.join(storage_path, "tensorboard")
    diagnostics_path = os.path.join(storage_path, "diagnostics")
    os.mkdir(storage_path)
    os.mkdir(weights_path)
    os.mkdir(tensorboard_path)
    os.mkdir(diagnostics_path)

    rew_disc_factor = args.rew_disc_factor
    saf_disc_factor = args.saf_disc_factor
    lamda = args.lamda
    safety_lamda = args.safety_lamda
    safety_desired_threshold = args.safety_desired_threshold
    center_advantages = args.center_advantages
    use_safety_baseline = args.use_safety_baseline

    experience_batch_size = args.timesteps_per_epoch
    n_epochs = args.n_epochs

    map_size = args.map_size
    map_resolution = args.map_resolution
    map_strategy = args.map_strategy
    obstacles_map = args.obstacles_map
    obstacle_padding = args.obstacle_padding
    free_padding = args.free_padding

    obstacle_positions = get_obstacle_positions(map_size, obstacles_map)
    obstacles_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, obstacle_padding) # 得到一个201×*201的矩阵，用于表示地图上每个位置是否有障碍物
    free_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, free_padding)

    arch = args.architecture
    if(arch == 'asl'):
        n_states = 38
    elif(arch == 'tai'):
        n_states = 12

    action_dim = 2 #Translational and rotational velocity
    trans_vel_limits = [args.trans_vel_low, args.trans_vel_high]
    rot_vel_limits = [args.rot_vel_low, args.rot_vel_high]

    std_trans_init = 0
    std_rot_init =0

    environment = Environment(args, n_states)
    environment.set_obstacles_map(obstacles_map, map_resolution)

    actor_filename = None

    # for test
    args.jump_start = 1
    args.model_init = '/logs/2019-10-14_15-54-16_tmp_model/weights/weights_actor700.p'

    if args.jump_start:
        print("Jump starting the model.")
        # actor_filename = os.path.join(rospkg.RosPack().get_path("reinforcement_learning_navigation"), args.model_init)
        actor_filename = '/home/heleidsn/catkin_ws/src/rl-navigation/logs/2019-10-14_15-54-16_tmp_model/weights/weights_actor700.p'
        critic_filename = '/home/heleidsn/catkin_ws/src/rl-navigation/logs/2019-10-14_15-54-16_tmp_model/weights/weights_critic700.p'

    print("Training setup:")
    print("Initializing the model from: {}".format(actor_filename))
    print("Translational velocity limits: {}".format(trans_vel_limits))
    print("Rotational velocity limits: {}".format(rot_vel_limits))
    print("The universal output file name is: {}".format(args.output_name))
    print("Timesteps per epoch: {}".format(experience_batch_size))
    print("Number of epochs: {}".format(n_epochs))

    actor_desired_kl = args.actor_desired_kl
    critic_desired_kl = args.critic_desired_kl
    safety_baseline_desired_kl = args.safety_baseline_desired_kl

    policy_estimator = Actor(n_states, action_dim, [trans_vel_limits, rot_vel_limits],
                                        [np.log(std_trans_init), np.log(std_rot_init)], actor_desired_kl, sess, arch, actor_filename)

    value_estimator = Critic(n_states, critic_desired_kl, sess, arch, filename=critic_filename)

    if(use_safety_baseline == True):
        value_estimator_safety = SafetyBaseline(n_states, safety_baseline_desired_kl, sess, arch, filename=None)
    experience_buffer = ExperienceBuffer()

    total_experiences = 0
    episode_number = 0
    episodes_this_epoch = 0

    summary_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    summary_dict = {'episodes_in_epoch':np.array(()), 'average_reward_sum_per_episode':np.array(()), 'average_returns':np.array(()), 'average_reward_per_experience':np.array(()), 'average_experience_per_episode':np.array(()), 'average_safety_constraint':np.array(()), 'success_rate':np.array(()), 'crash_rate':np.array(())}
    tf_ep_epoch = tf.placeholder(shape=[], dtype = tf.float32)
    tf_ep_epoch_avg_rew = tf.placeholder(shape=[], dtype = tf.float32)
    tf_ep_epoch_avg_disc_rew = tf.placeholder(shape=[], dtype = tf.float32)
    tf_ex_epoch_avg_rew = tf.placeholder(shape = [], dtype = tf.float32)
    tf_ex_per_ep_epoch_avg = tf.placeholder(shape = [], dtype = tf.float32)
    tf_ep_epoch_avg_disc_safe_cost = tf.placeholder(shape = [], dtype = tf.float32)
    tf_frac_goal_reached = tf.placeholder(shape = [], dtype = tf.float32)
    tf_frac_crashed = tf.placeholder(shape = [], dtype = tf.float32)

    with tf.variable_scope("Diagnostics"):
        ep_epoch_summary = tf.summary.scalar('episodes_in_epoch', tf_ep_epoch)
        ep_epoch_avg_rew_summary = tf.summary.scalar('average_reward_sum_per_episode', tf_ep_epoch_avg_rew)
        ep_epoch_avg_disc_rew_summary = tf.summary.scalar('average_discounted_reward_sum_per_episode', tf_ep_epoch_avg_disc_rew)
        ex_epoch_avg_rew_summary = tf.summary.scalar('average_reward_per_experience', tf_ex_epoch_avg_rew)
        ex_per_ep_epoch_avg_summary = tf.summary.scalar('average_experience_per_episode', tf_ex_per_ep_epoch_avg)
        ep_epoch_avg_disc_safe_cost_summary = tf.summary.scalar('average_discounted_safety_cost', tf_ep_epoch_avg_disc_safe_cost)
        tf_frac_goal_reached_summary = tf.summary.scalar('fraction_goal_reached', tf_frac_goal_reached)
        tf_frac_crashed_summary = tf.summary.scalar('fraction_crashed', tf_frac_crashed)
        summary_op = tf.summary.merge([ep_epoch_summary, ep_epoch_avg_rew_summary, ep_epoch_avg_disc_rew_summary, ex_epoch_avg_rew_summary, ex_per_ep_epoch_avg_summary, ep_epoch_avg_disc_safe_cost_summary, tf_frac_goal_reached_summary, tf_frac_crashed_summary])

    sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()
    graph.finalize()

    # init robot position
    rospy.wait_for_service('reset_positions')
    reset_pose = rospy.ServiceProxy('reset_positions', Empty)
    try:
        reset_pose()
    except rospy.ServiceException:
        print ("reset_positions service call failed")


    goal = [[-1,9,0], [0,0,0]]
    environment.set_goal(goal)



    # Search for all position
    action1 = np.zeros((9, 9, 10))
    action2 = np.zeros((9, 9, 10))
    q_values = np.zeros((9, 9, 10))
    for i in range(9):
        for j in range(9):
            for r in range(10):
                # set robot position
                x = i - 9
                y = j + 1
                yaw = 0.2 * math.pi * r
                environment.set_robot_pose([x, y, 0], [0, 0, yaw])
                # get state, action and q-value
                state = environment.reset()
                action = policy_estimator.predict_action(np.reshape(state, (-1, n_states)))
                q_value = value_estimator.predict_value(np.reshape(state, (-1, n_states)))
                action1[i, j, r] = action[0]
                action2[i, j, r] = action[1]
                q_values[i, j, r] = q_value

    np.save('scripts/visualization/action1.npy', action1)
    np.save('scripts/visualization/action2.npy', action2)
    np.save('scripts/visualization/q_values.npy', q_values)

    print('record finish...')


def test():
    for i, j in range(2, 10):
        print(i, j)

if __name__ == "__main__":
    main()
    # test()