# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from statistics import Statistics

class A3CTrainingThread(object):
  def __init__(self, thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               action_size,
               gamma,
               local_t_max,
               entropy_beta,
               agent_type,
               performance_log_interval,
               log_level,
               random_seed):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.action_size = action_size
    self.gamma = gamma
    self.local_t_max = local_t_max
    self.agent_type = agent_type
    self.performance_log_interval = performance_log_interval
    self.log_level = log_level

    if self.agent_type == 'LSTM':
      self.local_network = GameACLSTMNetwork(self.action_size, thread_index, device)
    else:
      self.local_network = GameACFFNetwork(self.action_size, device)

    self.local_network.prepare_loss(entropy_beta)

    with tf.device(device):
      var_refs = []
      variables = self.local_network.get_vars()
      for v in variables:
        var_refs.append(v)
      
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)
    
    np.random.seed(random_seed)
    self.game_state = GameState(random_seed * thread_index, self.action_size)
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate
    self.learn_rate = self.initial_learning_rate


    self.reset_counters()

    self.episode = 0

    # variable controling log output
    self.prev_local_t = 0

  def reset_counters(self):
    self.total_q_max = 0
    self.episode_reward = 0
    self.episode_actions = []
    self.passed_obst = 0

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    self.learn_rate = learning_rate

    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, statistics):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run(self.sync)

    start_local_t = self.local_t

    if self.agent_type == 'LSTM':
      start_lstm_state = self.local_network.lstm_state_out
    
    # t_max times loop
    for i in range(self.local_t_max):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
      action = self.choose_action(pi_)

      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      # process game
      self.game_state.process(action)

      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal
      steps = self.game_state.steps
      passed = self.game_state.passed_obst

      self.episode_reward += reward

      # clip reward
      rewards.append(np.clip(reward, -1, 1))

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()
      
      self.total_q_max += np.max(pi_)
      self.episode_actions.append(action)
      self.passed_obst = self.game_state.passed_obst

      if terminal:
        terminal_end = True
        self.episode += 1

        if self.log_level == 'FULL':
          reward_steps = format(float(self.episode_reward)/float(steps), '.4f')
          print "THREAD: {}  /  EPISODE: {}  /  TOTAL STEPS: {}  /  STEPS: {}  /  PASSED OBST: {}  /  REWARD: {}  /  REWARD/STEP: {}" .format(self.thread_index, self.episode, global_t, steps, self.passed_obst, self.episode_reward, reward_steps)
        
        statistics.update(global_t, self.episode_reward, self.total_q_max, steps, self.episode_actions, self.learn_rate, self.passed_obst)
 
        self.reset_counters()

        self.game_state.reset()
        if self.agent_type == 'LSTM':
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + self.gamma * R
      td = R - Vi
      a = np.zeros([self.action_size])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if self.agent_type == 'LSTM':
      batch_si.reverse()
      batch_a.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.learning_rate_input: cur_learning_rate} )
      
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= self.performance_log_interval) and (self.log_level == 'FULL'):
      self.prev_local_t += self.performance_log_interval
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    
