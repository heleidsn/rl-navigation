import tensorflow as tf
import keras.backend as K
from keras.models import Model


class TensorboardLog():
    def __init__(self, agent, log_dir):
        self.agent = agent

        self.loss_tensor = tf.Variable(initial_value=0, dtype=tf.float32)
        self.reward_tensor = tf.Variable(initial_value=0, dtype=tf.float32)
        self.state_tensor = tf.Variable(initial_value=0, dtype=tf.float32)
        self.mean_q_tensor = tf.Variable(initial_value=0, dtype=tf.float32)
        self.epsilon_tensor = tf.Variable(initial_value=0, dtype=tf.float32)
        
        createVar = locals()

        # init weight and biase histogram for all dense layers
        for layers in self.agent.actor.layers:
            if 'dense' in layers.name:
                # get data
                createVar['w_' + layers.name] = agent.actor.get_layer(layers.name).get_weights()[0].reshape(1, -1)
                createVar['b_' + layers.name] = agent.actor.get_layer(layers.name).get_weights()[1]

                # set data
                setattr(self, 'w_actor_' + layers.name + '_tensor', tf.Variable(initial_value=createVar['w_' + layers.name], dtype=tf.float32))
                setattr(self, 'b_actor_' + layers.name + '_tensor', tf.Variable(initial_value=createVar['b_' + layers.name], dtype=tf.float32))

                # create summary
                tf.summary.histogram('w_' + layers.name, getattr(self, 'w_actor_' + layers.name + '_tensor'))
                tf.summary.histogram('b_' + layers.name, getattr(self, 'b_actor_' + layers.name + '_tensor'))

        for layers in self.agent.critic.layers:
            if 'dense' in layers.name:
                # get data
                createVar['w_' + layers.name] = agent.critic.get_layer(layers.name).get_weights()[0].reshape(1, -1)
                createVar['b_' + layers.name] = agent.critic.get_layer(layers.name).get_weights()[1]

                # set data
                setattr(self, 'w_critic_' + layers.name + '_tensor', tf.Variable(initial_value=createVar['w_' + layers.name], dtype=tf.float32))
                setattr(self, 'b_critic_' + layers.name + '_tensor', tf.Variable(initial_value=createVar['b_' + layers.name], dtype=tf.float32))

                # create summary
                tf.summary.histogram('w_' + layers.name, getattr(self, 'w_critic_' + layers.name + '_tensor'))
                tf.summary.histogram('b_' + layers.name, getattr(self, 'b_critic_' + layers.name + '_tensor'))


        self.init = tf.global_variables_initializer()
        
        self.agent.sess.run(self.init)

        tf.summary.scalar("Ave Loss", self.loss_tensor)
        tf.summary.scalar("Reward", self.reward_tensor)
        tf.summary.scalar("Finish State", self.state_tensor)
        tf.summary.scalar("Mean Q", self.mean_q_tensor)
        tf.summary.scalar("Epsilon", self.epsilon_tensor)

        

        self.graph = self.agent.sess.graph

        self.writer = tf.summary.FileWriter(logdir=log_dir)
        self.writer.add_graph(self.graph)

    def update(self, loss, reward_sum, done, q_value_log, epsilon, episode):
        K.get_session().run(tf.assign(self.loss_tensor, loss))
        K.get_session().run(tf.assign(self.reward_tensor, reward_sum))
        K.get_session().run(tf.assign(self.state_tensor, done))
        K.get_session().run(tf.assign(self.mean_q_tensor, q_value_log))
        K.get_session().run(tf.assign(self.epsilon_tensor, epsilon))

        if episode % 20 == 0:
            createVar = locals()
            for layers in self.agent.actor.layers:
                if 'dense' in layers.name:
                    # get value
                    createVar['w_' + layers.name] = self.agent.actor.get_layer(layers.name).get_weights()[0].reshape(1, -1)
                    createVar['b_' + layers.name] = self.agent.actor.get_layer(layers.name).get_weights()[1]

                    # set value
                    K.get_session().run(tf.assign(getattr(self, 'w_actor_' + layers.name + '_tensor'), createVar['w_' + layers.name]))
                    K.get_session().run(tf.assign(getattr(self, 'b_actor_' + layers.name + '_tensor'), createVar['b_' + layers.name]))

            for layers in self.agent.critic.layers:
                if 'dense' in layers.name:
                    # get value
                    createVar['w_' + layers.name] = self.agent.critic.get_layer(layers.name).get_weights()[0].reshape(1, -1)
                    createVar['b_' + layers.name] = self.agent.critic.get_layer(layers.name).get_weights()[1]

                    # set value
                    K.get_session().run(tf.assign(getattr(self, 'w_critic_' + layers.name + '_tensor'), createVar['w_' + layers.name]))
                    K.get_session().run(tf.assign(getattr(self, 'b_critic_' + layers.name + '_tensor'), createVar['b_' + layers.name]))


        result = K.get_session().run(tf.compat.v1.summary.merge_all())
        self.writer.add_summary(result, episode)