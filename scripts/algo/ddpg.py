# -*- coding: utf-8 -*- 
import os
import random

import math
import numpy as np
from collections import deque

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, concatenate, Flatten
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import plot_model


class DDPG():
    """Deep Deterministic Policy Gradient Algorithms.
    """
    def __init__(self, input_dim=12, output_dim=2, steer_range=math.radians(30), memory_size=60000, \
                 TAU=0.001, gamma=0.99, epsilon=1, epsilon_decay=0.998, epsilon_min=0.2, \
                 a_lr=0.0001, c_lr=0.001, velocity_min=0, velocity_max=1):

        self.sess = K.get_session()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.steer_range = steer_range
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.velocity_range = self.velocity_max - self.velocity_min
        
        # update rate for target model.
        self.TAU = TAU
        # experience replay.
        self.memory_buffer = deque(maxlen=memory_size)
        # discount rate for q value.
        self.gamma = gamma
        # epsilon of action selection
        self.epsilon = epsilon
        # discount rate for epsilon.
        self.epsilon_decay = epsilon_decay
        # min epsilon of ε-greedy.
        self.epsilon_min = epsilon_min

        # OU-Noise
        self.ou_noise1 = OrnsteinUhlenbeckActionNoise(mu = np.array([0]), sigma=0.1) # noise for angle （-0.5~0.5）
        self.ou_noise2 = OrnsteinUhlenbeckActionNoise(mu = np.array([0]), theta=0.05, sigma=0.4) # noise for speed （-4~4）
 
        # actor learning rate
        self.a_lr = a_lr
        # critic learining rate
        self.c_lr = c_lr

        # ddpg model
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # 打印模型
        # plot_model(self.actor, to_file='lgmd_ddpg_actor.png', show_shapes=True)
        # plot_model(self.critic, to_file='lgmd_ddpg_critic.png', show_shapes=True)

        # target model
        self.target_actor = self._build_actor()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic = self._build_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        # gradient function
        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

    def load(self, date_time, episode_num):
        """ Load model
        """
        path_actor = 'models/lgmd_ddpg/{}/ddpg_actor_{}.h5'.format(date_time, episode_num)
        path_critic = 'models/lgmd_ddpg/{}/ddpg_critic_{}.h5'.format(date_time, episode_num)
        if os.path.exists(path_actor) and os.path.exists(path_critic):
            self.actor.load_weights(path_actor)
            self.critic.load_weights(path_critic)
            return 1
        else:
            print('Model {} does not exist... Loading fail...'.format(episode_num))

    def _build_actor(self):
        """Actor model.
        """
        inputs = Input(shape=(self.input_dim,), name='state_input')
        x = Dense(1000, activation='tanh')(inputs)
        x = Dense(300, activation='tanh')(x)
        x = Dense(100, activation='tanh')(x)
        steering = Dense(1, activation='tanh')(x)
        velocity = Dense(1, activation='sigmoid')(x)
        output = concatenate([steering, velocity])

        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.a_lr))

        return model

    def _build_critic(self):
        """Critic model.
        """
        sinput = Input(shape=(self.input_dim,), name='state_input')
        ainput = Input(shape=(self.output_dim,), name='action_input')
        s = Dense(1000, activation='tanh')(sinput)
        a = Dense(300, activation='tanh')(ainput)
        x = concatenate([s, a])
        x = Dense(100, activation='tanh')(x)
        output = Dense(1, activation='linear')(x)

        model = Model(inputs=[sinput, ainput], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.c_lr))

        return model

    def actor_optimizer(self):
        """actor_optimizer.

        Returns:
            function, opt function for actor.
        """
        self.ainput = self.actor.input
        aoutput = self.actor.output
        trainable_weights = self.actor.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, self.output_dim))

        # tf.gradients will calculate dy/dx with a initial gradients for y
        # action_gradient is dq / da, so this is dq/da * da/dparams
        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        self.opt = tf.train.AdamOptimizer(self.a_lr).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def critic_gradient(self):
        """get critic gradient function.

        Returns:
            function, gradient function for critic.
        """
        cinput = self.critic.input
        coutput = self.critic.output

        # compute the gradient of the action with q value, dq/da.
        action_grads = K.gradients(coutput, cinput[1])

        return K.function([cinput[0], cinput[1]], action_grads)

    def get_action(self, X):
        """get actor action with ou noise.

        Arguments:
            X: state value.
            输出：偏角（弧度制） 速度（m/s）
        """
        action_original = self.actor.predict(X)
        action = np.zeros([1, self.output_dim])
        # add randomness to action selection for exploration
        action[0][0] = np.clip(action_original[0][0]*self.steer_range + self.ou_noise1() * max(self.epsilon, 0) * 0.5, -self.steer_range, self.steer_range)
        action[0][1] = np.clip(action_original[0][1]*self.velocity_range + self.ou_noise2() * max(self.epsilon, 0), self.velocity_min, self.velocity_max)
      
        return action, action_original

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.

        Arguments:
            state: observation.
            action: action.
            reward: reward.
            next_state: next_observation.
            done: if game done.
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """update epsilon.
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self, batch):
        """process batch data.

        Arguments:
            batch: batch size.

        Returns:
            states: states.
            actions: actions.
            y: Q_value.
        """
        y = []
         # ranchom choice batch data from experience replay.
        data = random.sample(self.memory_buffer, batch)
        states = np.array([d[0] for d in data])
        actions = np.array([d[1] for d in data])
        next_states = np.array([d[3] for d in data])

        # Q_target。
        next_actions = self.target_actor.predict(next_states)
        q = self.target_critic.predict([next_states, next_actions])

        # update Q value
        for i, (_, _, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * q[i][0]
            y.append(target)

        return states, actions, y

    def update_model(self, X1, X2, y):
        """update ddpg model.

        Arguments:
            states: states.
            actions: actions.
            y: Q_value.

        Returns:
            loss: critic loss.
        """
        # loss = self.critic.train_on_batch([X1, X2], y)
        loss = self.critic.fit([X1, X2], y, verbose=0)
        loss = np.mean(loss.history['loss'])

        X3 = self.actor.predict(X1)
        a_grads = np.array(self.get_critic_grad([X1, X3]))[0]
        self.sess.run(self.opt, feed_dict={
            self.ainput: X1,
            self.action_gradient: a_grads
        })

        return loss

    def update_target_model(self):
        """soft update target model.
        formula：θ​​t ← τ * θ + (1−τ) * θt, τ << 1. 
        """
        critic_weights = self.critic.get_weights()
        actor_weights = self.actor.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]

        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]

        self.target_critic.set_weights(critic_target_weights)
        self.target_actor.set_weights(actor_target_weights)

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)