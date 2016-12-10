# Asynchronous Deep ReinFlappyBird

<p align="center">
  <img src="visuals/play.gif"/>
</p>

This repository contains an implementation of Asynchronous Advantage Actor-Critic (A3C) that teaches an agent to play Flappy Bird.

## Performance
Coming soon!

## Technical Details

For my tests, these are the training speeds when using a **CPU** (Intel Core i7 6700K, 4.0 GHz) or **GPU** (NVIDIA GTX1070).

|         | FF            |LSTM          |
|---------|---------------|--------------|
| **CPU** | TBA steps/s   |TBA steps/s   |
| **GPU** | 400 steps/s		|300 steps/s  |


## Settings
Here are some of the available flags you can set when you train an agent. For the full list, see [```a3c.py```](/a3c.py).

### Agent settings
* ```mode``` / ```[train, display, visualize] ``` - Which mode you want to activate when you start a session.
* ```use_gpu``` / ```[True, False]``` - If you have a/want to use GPU to speed up the training process. 
* ```parallel_agent_size``` - Number of parallel agents to use during training. 
* ```action_size``` - Numbers of available actions.
* ```agent_type``` / ```[FF, LSTM]``` - What type of A3C to train the agent with. 



### Training and Optimizer settings
The current settings are based on or borrowed from the [implemenentation] (https://github.com/miyosuda/async_deep_reinforce) by [@miyosuda](https://github.com/miyosuda).
They have not yet been optimized for Flappy Bird but rather used _as is_ for now. Tell me settings that perform better than the current ones!

* ```max_time_step - 40 000 000``` - Maximum training steps. 
* ```initial_alpha_low - -5``` - LogUniform low limit for learning rate (represents x in 10^x).
* ```initial_alpha_high -  -3``` - LogUniform high limit for learning rate (represents x in 10^x).
* ```gamma - 0.99``` - Discount factor for rewards.
* ```entropy_beta - 0.01``` - Entropy regularization constant.
* ```grad_norm_clip - 40.0```- Gradient norm clipping.
* ```rmsp_alpha - 0.99``` - Decay parameter for RMSProp.
* ```rmsp_epsilon - 0.1``` - Epsilon parameter for RMSProp.
* ```local_t_max - 5```- Repeat step size.


### Logging
* ```log_level``` - Log level ```[NONE, FULL]```
* ```average_summary``` - How many episodes to average summary over.

### Display
* ```display_episodes``` - Numbers of episodes to display. 
* ```average_summary``` - How many episodes to average summary over.
* ```display_log_level``` - Display log level - ```NONE``` prints end summary, ```MID``` prints episode summary and ```FULL``` prints the π-values, state value and reward for every state. ```[NONE, MID, FULL]```



## Getting started
To start a training session with the default parameters, run:

```
$ python a3c.py
```

To check your progress and possibly compare different experiments in real time, navigate to your  ```async-deep-flappybird``` folder and start tensorboard by running:

```
$ tensorboard --logdir summaries/
```

Enjoy!

## Credit
**A3C** - The A3C implementation used is a modified [version](https://github.com/miyosuda/async_deep_reinforce) by [@miyosuda](https://github.com/miyosuda).

**Flappy Bird** - The Flappy Bird implementation is based on a [version](https://github.com/yenchenlin/DeepLearningFlappyBird) by [@yenchenlin](https://github.com/yenchenlin) with som minor adjustments.

—


_2016, Babak Toghiani-Rizi_