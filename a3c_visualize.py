# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import random

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
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
#  for ax,i in zip(axes.flat, range(filters)):
#    print i
#    inch = i//16
#    outch = i%16
    #plt.subplot(7,6,i+1)
 #   plt.title('Filter ' + str(i))
 #   ax.imshow(units[0,:,:,i], interpolation="nearest", cmap=plt.cm.gray)
 #
#  plt.show()
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
    global_network = GameACFFNetwork(action_size, device)

  training_threads = []

  learning_rate_input = tf.placeholder("float")

  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = rmsp_alpha,
                                momentum = 0.0,
                                epsilon = rmsp_epsilon,
                                clip_norm = grad_norm_clip,
                                device = device)

  # for i in range(PARALLEL_SIZE):
  #   training_thread = A3CTrainingThread(i, global_network, 1.0,
  #                                       learning_rate_input,
  #                                       grad_applier, MAX_TIME_STEP,
  #                                       device = device)
  #   training_threads.append(training_thread)





  #x_t = np.zeros((80,80))
  #x_s = random.randint(0,79)
  #y_s = random.randint(0,79)
  #x_g = random.randint(0,79)
  #y_g = random.randint(0,79)

  #x_t[x_s, y_s] = 0.5
  #x_t[x_g, y_g] = 1.0

  game = GameState(rand_seed, action_size)
  game.reset()
  game.process(0)
  x_t = game.x_t

  #print x_t.shape


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

  #print len(W_conv2)
  for ax,i in zip(axes.flat, range(2*32)):
    inch = i//32
    outch = i%32
    img = W_conv2[:,:,inch,outch]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))

  plt.show()

  arr = sess.run(global_network.get_vars())
  #print arr

  s = tf.placeholder("float", [None, 84, 84, 4])
  #s = sess.run(global_network.s)
  b_conv1 = sess.run(global_network.b_conv1)
  b_conv2 = sess.run(global_network.b_conv2)

  inp_1 = tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "VALID")
  h_conv1 = tf.nn.relu(inp_1 + b_conv1)

  inp_2 = tf.nn.conv2d(h_conv1, W_conv2, strides = [1, 2, 2, 1], padding = "VALID")
  h_conv2 = tf.nn.relu(inp_2 + b_conv2)

  #x_t = np.reshape(x_t, (80, 80,- 1))
  s_t = game.s_t

  #h_conv1 = tf.nn.relu(sess.run(global_network._conv2d(s, W_conv1, 4)) + b_conv1)
  #h_conv2 = tf.nn.relu(sess.run(global_network._conv2d(h_conv1, W_conv2, 2)) + b_conv2)
  #print image_input.shape
  getActivations(sess, s, h_conv1, s_t, 16)
  getActivations(sess, s, h_conv2, s_t, 32)