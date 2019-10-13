# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import cPickle as pickle
import os.path
import time
import tensorflow as tf
import numpy as np
import rospy
import rospkg

import datetime

from environments.ddpg_environment import Environment
from utils.memory import ExperienceBuffer

from utils.common import *
from utils.map import *
from options.option_ddpg import Options
from std_srvs.srv import Empty

from algo.ddpg import DDPG
from utils.tensorboardlog import TensorboardLog
from tqdm import tqdm

def get_dir():
    base_path = rospkg.RosPack().get_path("reinforcement_learning_navigation")
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        base_path,
        'ddpg',
        "logs",
        "ddpg_stage",
        date_time,
    )
    model_dir = os.path.join(
        base_path,
        'ddpg'
        "models",
        "ddpg_stage",
        date_time,
    )
    os.makedirs(model_dir)
    return log_dir, model_dir

def main():
    args = Options().parse()

    # get obstacle and free map
    map_size = args.map_size
    map_resolution = args.map_resolution
    map_strategy = args.map_strategy
    obstacles_map = args.obstacles_map
    obstacle_padding = args.obstacle_padding
    free_padding = args.free_padding

    obstacle_positions = get_obstacle_positions(map_size, obstacles_map)
    obstacles_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, obstacle_padding)
    free_map = get_obstacles_map(map_size, obstacle_positions, map_resolution, free_padding)

    n_states = 38

    # create environment
    environment = Environment(args, n_states)
    environment.set_obstacles_map(obstacles_map, map_resolution)

    # Create ddpg model
    model = DDPG(input_dim=n_states, output_dim=2, steer_range=args.rot_vel_high, velocity_min=args.trans_vel_low, velocity_max=args.trans_vel_high)

    # init logger
    log_dir, model_dir = get_dir()
    logger= TensorboardLog(model, log_dir)

    # Some training setting
    episode = 10000
    batch=64
    time_every_episode=40

    # map_choice
    map_choice = get_map_choice(map_strategy)

    # Init tqdm
    tqdm_e = tqdm(range(episode), desc="Reward: {:.2f} | loss: {:.2f}".format(0, 0), \
        leave=True, unit=" episodes")

    # Start training
    print("Start training")
    for episode in tqdm_e:
        # start a new episode
        reward_sum = 0
        losses = [0]
        q_values = []
        done = 0

        # init start point and target
        # init robot position
        rospy.wait_for_service('reset_positions')
        reset_pose = rospy.ServiceProxy('reset_positions', Empty)
        try:
            reset_pose()
        except rospy.ServiceException:
            print ("reset_positions service call failed")

        goal_position = get_free_position(free_map, map_resolution, map_size/2, map_choice)
        goal_orientation = (2*np.random.rand() - 1) * np.pi
        goal = [[goal_position[0],goal_position[1],0], [0,0,goal_orientation]]
        environment.set_goal(goal)

        # environment reset and get the first observation
        state = environment.reset()

        while not done:
            # chocie action from ε-greedy.
            x = state.reshape(-1, n_states)

            # actor action
            action, _ = model.get_action(x)
            action1 = [action[0][1], action[0][0]]

            # get Q-value
            q_value = model.critic.predict([x, action])
            q_values.append(q_value[0])

            # observation, reward, done, _ = env.step(action)
            observation, reward, safety_cost, simulator_flag, _ = environment.execute_action(action1)

            if environment.crashed:
                done = 1
            elif environment.goal_reached:
                done = 3
            elif not environment.is_running:
                done = 2
            
            # add data to experience replay.
            reward_sum += reward
            model.remember(x[0], action[0], reward, observation, done)

            # start training once buffer > 2000
            if len(model.memory_buffer) > 2000:
                X1, X2, y = model.process_batch(batch)

                # update DDPG model
                loss = model.update_model(X1, X2, y)
                # update target model
                model.update_target_model()
                losses.append(loss)
                # client.simPrintLogMessage("Loss", loss, 0)

        # reduce epsilon per episode
        model.update_epsilon()

        loss = np.mean(losses)
        q_value_log = np.mean(q_values)

        # tensorboard update
        logger.update(loss, reward_sum, done, q_value_log, model.epsilon, episode)

        # 暂存模型
        if episode % 20 == 0: 
            model.actor.save_weights(model_dir + '/ddpg_actor_{}.h5'.format(episode))
            model.critic.save_weights(model_dir + '/ddpg_critic_{}.h5'.format(episode))

        # print('Episode: {}/{} | reward: {} | loss: {:.3f}'.format(i, episode, reward_sum, loss))
        tqdm_e.set_description("Reward: {:.2f} | loss: {:.2f}".format(reward_sum, loss))
        tqdm_e.refresh()

    # end of training
    print("Training Finished")


if __name__ == "__main__":
    main()