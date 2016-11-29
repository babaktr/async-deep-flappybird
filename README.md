# Asynchronous Deep ReinFlappyBird
This repo contains an Asynchronous Advantage Actor-Critic (A3C) Algorithm framework for training an agent to play Flappy Bird.

More info later!

## Flags
Here are some of the available flags you can set when you train an agent. For the full list, see [```a3c.py```](https://github.com/babaktr/async-deep-flappybird/blob/master/a3c.py).

### Agent
* ```mode``` - Which mode you want to activate when you start a session. ```[train, display, visualize] ```
* ```use_gpu``` - If you have a/want to use GPU to speed up the training process. ```[True, False]```
* ```parallel_agent_size``` - Number of parallel agents to use during training. 
* ```action_size``` - Numbers of available actions.
* ```agent_type``` - What type of A3C to train the agent with. ```[FF, LSTM]```



### Training and Optimizer settings
The current settings are based on or borrowed from the [implemenentation] (https://github.com/miyosuda/async_deep_reinforce) by [@miyosuda](https://github.com/miyosuda).
They have not yet been optimized for Flappy Bird but rather used _as is_ for now.

* ```max_time_step - 40 000 000``` - Maximum training steps. 
* ```initial_alpha_log_rate - 0.4226``` - log_uniform interpolate rate for learning rate (around 7*10^-4).
* ```initial_alpha_low - 0.0001``` - log_uniform low limit for learning rate.
* ```initial_alpha_high -  0.01``` - log_uniform high limit for learning rate.
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
To start a training session with the default parameters, run

```
$ python a3c.py
```

Enjoy!

## Credit
**A3C** - A modified version [originally](https://github.com/miyosuda/async_deep_reinforce) by [@miyosuda](https://github.com/miyosuda).

**Flappy Bird** - A slightly modified version [originally](https://github.com/yenchenlin/DeepLearningFlappyBird) by [@yenchenlin](https://github.com/yenchenlin).

—


_2016, Babak Toghiani-Rizi_