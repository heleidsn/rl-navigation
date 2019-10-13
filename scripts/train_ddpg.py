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
    model = DDPG(input_dim=n_states, output_dim=2, steer_range=1, velocity_min=0.1, velocity_max=1, epsilon_min=0.2, epsilon_decay=0.98)

    # init logger
    log_dir, model_dir = get_dir()
    logger= TensorboardLog(model, log_dir)

    # map_choice
    map_choice = get_map_choice(map_strategy)

    # Training setting
    n_epochs = 1000
    episode_every_epoch = 20
    epoch = 1 
    batch_size = 1000
    goal_reach_counter = 0
    crash_counter = 0

    total_experiences = 0
    episode_number = 0
    episodes_this_epoch = 0

    epoch_start_time = time.time()

    # Start training
    print("Start training")
    while epoch <= n_epochs:
        # start a new episode
        reward_sum = 0
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

        episode_number += 1
        episodes_this_epoch += 1

        experience_counter = 0

        while(environment.is_running):
            # chocie action from ε-greedy.
            x = state.reshape(-1, n_states)

            # actor action
            action, _ = model.get_action(x)
            action1 = [action[0][1], action[0][0]]

            # get Q-value
            # q_value = model.critic.predict([x, action])

            # step
            next_state, reward, _, simulator_flag = environment.execute_action(action1)

            if(simulator_flag == True): #Workaround for stage simulator crashing and restarting
                episode_number -= 1
                episodes_this_epoch -= 1
                break

            if environment.crashed:
                done = 1
            elif environment.goal_reached:
                done = 3
            elif not environment.is_running:
                done = 2

            experience_counter += 1
            state = next_state
            
            # add data to experience replay.
            reward_sum += reward
            model.remember(x[0], action[0], reward, next_state, done)
            # print('Exp: {:d}  speed: {:.2f}  steer: {:.2f}  reward: {:.2f}  done: {:d}'.format(experience_counter, action1[0], action1[1], reward, done))
        
        print('episode_number: {:d} target_pos: {:.2f} {:.2f} total_reward: {:.2f}'  \
            .format(episode_number, goal_position[0], goal_position[1], reward_sum))
            
        if simulator_flag == False:
            #Compute metrics
            goal_reach_counter += environment.goal_reached
            crash_counter += environment.crashed
            total_experiences += experience_counter
            
            # train every epoch or every 20 episode
            if len(model.memory_buffer) > batch_size and episode_number % episode_every_epoch == 0:
                # print epoch summary
                print('epoch {}, Updating after {} episodes, Time for epoch {}'.format(epoch, episodes_this_epoch, (time.time() - epoch_start_time)))
                crash_rate = float(crash_counter)/episode_every_epoch
                success_rate = float(goal_reach_counter)/episode_every_epoch
                print('Crash Rate: ', crash_rate)
                print('Success Rate: ', success_rate)
                # get batch
                X1, X2, y = model.process_batch(batch_size)
                # update DDPG model
                loss = model.update_model(X1, X2, y)
                print('Loss: ', loss)
                # update target model
                model.update_target_model()

                # tensorboard update
                # replace done and q_value_log with crash_rate and success_rate
                logger.update(loss, reward_sum, crash_rate, success_rate, model.epsilon, epoch)

                # 暂存模型
                if epoch % 20 == 0: 
                    model.actor.save_weights(model_dir + '/ddpg_actor_{}.h5'.format(epoch))
                    model.critic.save_weights(model_dir + '/ddpg_critic_{}.h5'.format(epoch))

                epoch += 1
                epoch_start_time = time.time()
                goal_reach_counter = 0
                crash_counter = 0
                episodes_this_epoch = 0

                # reduce epsilon per epoch
                model.update_epsilon()

    # end of training
    print("Training Finished")


if __name__ == "__main__":
    main()