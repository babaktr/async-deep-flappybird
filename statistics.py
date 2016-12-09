import os
import numpy as np
import tensorflow as tf

class Statistics(object):
  def __init__(self, sess, summary_writer, average_summary):
    self.sess = sess

    self.average_summary = average_summary
    self.average_summary_count = 0

    self.writer = summary_writer

    with tf.variable_scope('summary'):
      scalar_summary_tags = [
        # average
        'episode/avg_reward', 
        'episode/avg_reward_per_step', 
        'episode/avg_q_max', 
        'episode/avg_steps', 
        'episode/avg_passed_obstacles',
        # max
        'max/max_episode_reward',
        'max/max_reward_per_step',
        'max/max_steps',
        'max/max_passed_obstacles',
        # min
        'min/min_episode_reward',
        'min/min_reward_per_step',
        'min/min_steps',
        'min/min_passed_obstacles',
        # training
        'training/avg_learning_rate',
        'training/min_learning_rate',
        'training/max_learning_rate'
      ]

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary(tag, self.summary_placeholders[tag])

      histogram_summary_tags = ['episode/episode_rewards', 
                                'episode/episode_actions',
                                'episode/passed_obstacles']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.reset_average_summary()


  def reset_average_summary(self):
    self.episode_rewards = []
    self.episode_rewards_per_step = []
    self.episode_avg_q_max = []
    self.episode_steps = []
    self.episode_passed_obstacles = []
    self.episode_learning_rate = []

    self.episode_actions = []


  def update(self, global_t, episode_reward, total_q_max, 
            episode_steps, episode_actions, learning_rate, passed_obst):
    rewards_per_step = float(episode_reward) / float(episode_steps)
    ep_avg_q_max = total_q_max / episode_steps

    # average
    self.episode_rewards.append(episode_reward)
    self.episode_rewards_per_step.append(rewards_per_step)
    self.episode_avg_q_max.append(ep_avg_q_max)
    self.episode_steps.append(episode_steps)
    self.episode_passed_obstacles.append(passed_obst)
    self.episode_actions = self.episode_actions + episode_actions
    self.episode_learning_rate.append(learning_rate)

    if self.average_summary_count % self.average_summary == 0:
      self.inject_summary({
              # average
              'episode/avg_reward': np.average(self.episode_rewards), 
              'episode/avg_reward_per_step': np.average(self.episode_rewards_per_step), 
              'episode/avg_q_max': np.average(self.episode_avg_q_max), 
              'episode/avg_steps': np.average(self.episode_steps),
              'episode/avg_passed_obstacles': np.average(self.episode_passed_obstacles),
              # max
              'max/max_episode_reward': np.max(self.episode_rewards),
              'max/max_reward_per_step': np.max(self.episode_rewards_per_step),
              'max/max_steps': np.max(self.episode_steps),
              'max/max_passed_obstacles': np.max(self.episode_passed_obstacles),
              # min
              'min/min_episode_reward': np.min(self.episode_rewards),
              'min/min_reward_per_step': np.min(self.episode_rewards_per_step),
              'min/min_steps': np.min(self.episode_steps),
              'min/min_passed_obstacles': np.min(self.episode_passed_obstacles),
              # training
              'training/avg_learning_rate': np.average(self.episode_learning_rate),
              'training/min_learning_rate': np.min(self.episode_learning_rate),
              'training/max_learning_rate': np.max(self.episode_learning_rate),
              # histogram
              'episode/episode_rewards': self.episode_rewards,
              'episode/episode_actions': self.episode_actions,
              'episode/passed_obstacles': self.episode_passed_obstacles
            }, global_t)

      self.reset_average_summary()

    self.average_summary_count += 1

  def inject_summary(self, tag_dict, t):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, t)
      self.writer.flush()

