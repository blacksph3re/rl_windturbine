import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

def placeholder(dim=None):
    return tf1.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf1.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hparams):
    act_dim = a.shape.as_list()[-1]
    with tf1.variable_scope('pi'):
        pi = hparams.act_limit * mlp(x, list(hparams.ac_hidden_sizes)+[hparams.act_dim], hparams.ac_activation, hparams.ac_output_activation)
    with tf1.variable_scope('q'):
        q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hparams.ac_hidden_sizes)+[1], hparams.ac_activation, None), axis=1)
    with tf1.variable_scope('q', reuse=True):
        q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hparams.ac_hidden_sizes)+[1], hparams.ac_activation, None), axis=1)
    return pi, q, q_pi