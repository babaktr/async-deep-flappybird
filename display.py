# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

import random


def choose_action(pi_values):
  return np.random.choice(range(len(pi_values)), p=pi_values)

def display(experiment_name,
            rmsp_alpha,
            rmsp_epsilon,
            grad_norm_clip,
            agent_type,
            action_size,
            rand_seed,
            checkpoint_dir,
            display_time_sleep,
            display_episodes,
            display_log_level,
            display_save_log,
            show_max):

  # use CPU for display tool
  device = "/cpu:0"

  LOG_FILE = 'log_{}-{}.txt'.format(experiment_name, agent_type)

  if agent_type == 'LSTM':
    global_network = GameACLSTMNetwork(action_size, -1, device)
  else:
    global_network = GameACFFNetwork(action_size, -1, device)

  learning_rate_input = tf.placeholder("float")

  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = rmsp_alpha,
                                momentum = 0.0,
                                epsilon = rmsp_epsilon,
                                clip_norm = grad_norm_clip,
                                device = device)

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

  episode = 0
  terminal = False

  episode_rewards = []
  episode_steps = []
  episode_passed_obsts = []
  print ' '
  print 'DISPLAYING {} EPISODES'.format(display_episodes)
  print '--------------------------------------------------- '

  while not episode == display_episodes:
    episode_reward = 0
    episode_passed_obst = 0

    game_state = GameState(rand_seed, action_size, show_score=True)

    if display_log_level == 'FULL':
      print 'EPISODE {}'.format(episode)

    full_frame = None
    while True:
      pi_values, value = global_network.run_policy_and_value(sess, game_state.s_t)
      action = choose_action(pi_values)
      game_state.process(action)
      terminal = game_state.terminal
      episode_step = game_state.steps
      reward = game_state.reward
      passed_obst = game_state.passed_obst
      if len(episode_passed_obsts) == 0 and show_max:
        if passed_obst > 0:
          full_frame = game_state.full_frame
      elif episode_passed_obst > np.max(episode_passed_obsts) and show_max:
        full_frame = game_state.full_frame


      episode_reward += reward
      episode_passed_obst += passed_obst

      if display_log_level == 'FULL':
        print 'step  /  pi_values: {}  /  value: {}  /  action: {}  /  reward: {}  /  passed_obst: {}'.format(pi_values, value, action, reward, passed_obst)

      time.sleep(display_time_sleep)

      if not terminal:
        game_state.update()
      else:
        break

    episode_rewards.append(episode_reward)
    episode_steps.append(episode_step)
    episode_passed_obsts.append(episode_passed_obst)

    if not display_log_level == 'NONE':
      reward_steps = format(float(episode_reward)/float(episode_step), '.4f')
      print "EPISODE: {}  /  STEPS: {}  /  PASSED OBST: {}  /  REWARD: {}  /  REWARD/STEP: {}".format(episode, episode_step, passed_obst, episode_reward, reward_steps)
    
    if display_save_log:
      with open(LOG_FILE, "a") as text_file:
        text_file.write('{},{},{},{},{}\n'.format(episode, episode_step, passed_obst, episode_reward, reward_steps))

    episode += 1

  print '--------------------------------------------------- '
  print 'DISPLAY SESSION FINISHED'
  print 'TOTAL EPISODES: {}'.format(display_episodes)
  print ' '
  print 'MIN'
  print 'REWARD: {}  /  STEPS: {}  /  PASSED OBST: {}'.format(np.min(episode_rewards), np.min(episode_steps), np.min(episode_passed_obsts))
  print ' '
  print 'AVERAGE'
  print  'REWARD: {}  /  STEPS: {}  /  PASSED OBST: {}'.format(np.average(episode_rewards), np.average(episode_steps), np.average(episode_passed_obsts))
  print ' '
  print 'MAX'
  print 'REWARD: {}  /   STEPS: {}  /   PASSED OBST: {}'.format(np.max(episode_rewards), np.max(episode_steps), np.max(episode_passed_obsts))

  if show_max and not full_frame == None:
    plt.imshow(full_frame, origin='lower')
    plt.show()




