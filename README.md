# Pytorch SAC

> Pytorch implementation of Soft Actor-Critic Algorithm

Soft actor-critic (SAC) is an off-policy actor-critic deep RL algorithm that 
optimizes stochastic continuous policies defined in the maximum entropy 
framework. 

Paper: Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018), 
[*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://arxiv.org/abs/1801.01290)


## Installation

1. Clone the repository 
```
git clone https://github.com/domingoesteban/pytorch_sac
```

2. Install required python packages
```
cd pytorch_sac
pip install -r requirements.txt
```

## Usage

- Run (train) the algorithm with a continuous observation and action 
OpenAI-gym environment (E.g. Pendulum-v0)

```
python train.py -e Pendulum-v0 -i 30
```

- Plot the results from the previous training process

```
python eval.py -p PATH_TO_LOG_DIRECTORY
```

- Evaluate the resulted policy

```
python eval.py PATH_TO_LOG_DIRECTORY
```
