# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import random

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier


def getActivations(sess, s, layer, stimuli, filters):
  #print "stim shape: %s" % stimuli.shape
  units = layer.eval(session=sess, feed_dict=({s: [stimuli]}))
  plotNNFilter(units, filters)


def plotNNFilter(units, filters):
  filters = units.shape[3]
  test = units.shape[1]
  print test
  #plt.figure(1, figsize=(20,20))

  fig, axes = plt.subplots(1, filters, figsize=(30, 6),
             subplot_kw={'xticks': [], 'yticks': []})

  print filters

  for ax,i in zip(axes.flat, range(1*filters)):
    inch = i//filters
    outch = i%filters
    img = units[0,:,:,i]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))
  plt.show()

def visualize(experiment_name,
              rmsp_alpha,
              rmsp_epsilon,
              grad_norm_clip,
              agent_type,
              action_size,
              rand_seed,
              checkpoint_dir):

  # use CPU for weight visualize tool
  device = "/cpu:0"

  if agent_type == 'LSTM':
    global_network = GameACLSTMNetwork(action_size, -1, device)
  else:
    global_network = GameACFFNetwork(action_size, -1, device)

  training_threads = []

  learning_rate_input = tf.placeholder("float")

  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = rmsp_alpha,
                                momentum = 0.0,
                                epsilon = rmsp_epsilon,
                                clip_norm = grad_norm_clip,
                                device = device)

  game = GameState(rand_seed, action_size)
  game.process(0)
  x_t = game.x_t

  plt.imshow(x_t, interpolation="nearest", cmap=plt.cm.gray)

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old checkpoint")
    
  W_conv1 = sess.run(global_network.W_conv1)

  # show graph of W_conv1
  fig, axes = plt.subplots(4, 16, figsize=(12, 6),
               subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  for ax,i in zip(axes.flat, range(4*16)):
    inch = i//16
    outch = i%16
    img = W_conv1[:,:,inch,outch]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))

  plt.show()

  W_conv2 = sess.run(global_network.W_conv2)

  # show graph of W_conv2
  fig, axes = plt.subplots(2, 32, figsize=(27, 6),
               subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  for ax,i in zip(axes.flat, range(2*32)):
    inch = i//32
    outch = i%32
    img = W_conv2[:,:,inch,outch]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))

  plt.show()

  arr = sess.run(global_network.get_vars())

  s = tf.placeholder("float", [None, 84, 84, 4])

  b_conv1 = sess.run(global_network.b_conv1)
  b_conv2 = sess.run(global_network.b_conv2)

  inp_1 = tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "VALID")
  h_conv1 = tf.nn.relu(inp_1 + b_conv1)

  inp_2 = tf.nn.conv2d(h_conv1, W_conv2, strides = [1, 2, 2, 1], padding = "VALID")
  h_conv2 = tf.nn.relu(inp_2 + b_conv2)

  s_t = game.s_t

  getActivations(sess, s, h_conv1, s_t, 16)
  getActivations(sess, s, h_conv2, s_t, 32)