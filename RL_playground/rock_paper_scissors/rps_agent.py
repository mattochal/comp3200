import numpy as np
import Core.functions as cf
import tensorflow as tf


action_space = 3
observation_space = 1


class PolicyEstimator:
    """
    Policy Function approximator
    """

    def __init__(self, learning_rate=0.5):
        self.state = tf.placeholder(tf.int32, [3, 1], name="state")
        self.action = tf.placeholder(dtype=tf.int32, name="action")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        self.weights = tf.Variable(tf.random_normal(action_space, stddev=0.1))
        self.action_probs = tf.nn.softmax(tf.matmul(self.state, self.weights))

        # Get probability of the picked action
        self.picked_action_prob = tf.gather(self.action_probs, self.action)

        # Loss
        self.loss = -tf.log(self.picked_action_prob) * self.target
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state):
        sess = tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action):
        sess = tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.weights = tf.Variable(tf.random_normal(action_space, stddev=0.1))
            self.action_probs = tf.squeeze(tf.nn.softmax(tf.matmul(self.state, self.weights)))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator:
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.int32, name="state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just table lookup estimator
            state_one_hot = tf.one_hot(self.state, int(observation_space))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class RPSAgentLearner:

    def __init__(self):
        self.state_value = [0]
        self.policy_estimator = PolicyEstimator()
        self.value_estimator = ValueEstimator()

    def choose_action(self, state):
        action_probs = self.policy_estimator.predict(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update_policy_estimator(self, state, advantage, action):
        self.policy_estimator.update(state, advantage, action)

    def estimate_value(self, state):
        return self.value_estimator.predict(state)

    def update_value_estimator(self, state, target_value):
        self.value_estimator.update(state, target_value)


class RPSAgentNaive:

    def __init__(self):
        self.state_value = [0]
        self.policy = [cf.softmax(np.array([0, 0, 0]))]

    def choose_action(self, state):
        action_probs = self.policy[state]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action
