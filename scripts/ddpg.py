import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json

from ddpg.ReplayBuffer import ReplayBuffer
from ddpg.ActorNetwork import ActorNetwork
from ddpg.CriticNetwork import CriticNetwork
from ddpg.OU import OU
import timeit

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
from environments.ddpg_environment import Environment
from utils.common import *
from utils.map import *
from options.option_ddpg import Options
from std_srvs.srv import Empty
import rospy
import rospkg

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 2  #Steering/Acceleration/Brake
    state_dim = 38  #of sensors input

    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    done = False
    step = 0
    epsilon = 1

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
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

    # map_choice
    map_choice = get_map_choice(map_strategy)


    print("TORCS Experiment Start.")
    for i in range(episode_count):
        # start a new episode
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        total_reward = 0

        # init robot position
        rospy.wait_for_service('reset_positions')
        reset_pose = rospy.ServiceProxy('reset_positions', Empty)
        try:
            reset_pose()
        except rospy.ServiceException:
            print ("reset_positions service call failed")

        # set goal position 
        # goal_position = get_free_position(free_map, map_resolution, map_size/2, map_choice)
        # goal_orientation = (2*np.random.rand() - 1) * np.pi
        # goal = [[goal_position[0],goal_position[1],0], [0,0,goal_orientation]]
        goal = [[-1, 9, 0], [0,0,0]]
        environment.set_goal(goal)

        # environment reset and get the first observation
        state = environment.reset()

        for j in range(max_steps):
            loss = 0 

            # get action with OU noise
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            a_t_original = actor.model.predict(state.reshape(1, n_states))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)

            a_t[0][0] = np.clip(a_t_original[0][0] + noise_t[0][0], -1, 1)
            a_t[0][1] = np.clip(a_t_original[0][1] + noise_t[0][1], 0, 1)

            # step
            next_state, r_t, done, simulator_flag = environment.execute_action(a_t[0])

            if(simulator_flag == True): #Workaround for stage simulator crashing and restarting
                j -= 1 
                break

            buff.add(state, a_t[0], r_t, next_state, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            state = next_state
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    print("Finish.")

if __name__ == "__main__":
    playGame()
