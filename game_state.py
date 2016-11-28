# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
sys.path.append("game")
import numpy as np
import wrapped_flappy_bird as env
import random
import time
import cv2

class GameState(object):
  def __init__(self, rand_seed, action_size, show_score=False):
    self.rand_seed = rand_seed
    random.seed(self.rand_seed)
    self.action_size = action_size
    self.show_score = show_score

    self.reset()

    self.steps = 1
    
    self.reward = 0
    self.terminal = False

    self.reset()

  def _process_frame(self, action_vector, reshape):
    reward = 0
   
    x_t, reward, terminal = self.game.frame_step(action_vector)


    if reward >= 1:
        self.passed_obst += 1
        if self.show_score:
            self.full_frame = self.game.full_frame

    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)

    self.x_t = x_t # used for visualization

    if reshape:
      x_t = np.reshape(x_t, (84, 84, 1))

    return x_t, reward, terminal
    

  def reset(self):
    self.game = env.GameState(self.rand_seed, self.show_score)
    self.steps = 1
    self.passed_obst = 0

    x_t, _, _ = self._process_frame(self.random_action(), False)

    self.reward = 0
    self.terminal = False
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
  def vectorize_action(self, action):
    action_vector = [0] * self.action_size
    action_vector[action] = 1
    return action_vector

  def process(self, action):
    action_vector = self.vectorize_action(action)
    
    x_t1, r, t = self._process_frame(action_vector, True)

    self.reward = r
    self.terminal = t
    self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    

  def update(self):
    self.s_t = self.s_t1
    self.steps += 1

  def random_action(self):
    action = random.randint(0, self.action_size - 1)
    return self.vectorize_action(action)
