# RL101
Ruqirements: 需安装gym和tensorflow

## random_agents:

    -random_agents: 简单实现gym环境下的agent与环境交互
    
## dqn:

    -dqn_dis: 离散动作的dqn，solve的环境是"CartPole-v0"
    
    -dqn_con: 连续动作的dqn，solve的环境是"Pendulum-v0"，将连续动作离散化，8000步以上可以收敛
    
## policy_gradient:

    -reinforce_dis: 离散动作的REINFORCE算法，solve的环境是"CartPole-v0"
    
    -reinforce_con: 连续动作的REINFORCE算法，solve的环境是"Pendulum-v0"，不能收敛，估计是方差太大了
    
    -actor_critic/ppo: 连续动作的PPO算法，solve的环境是"Pendulum-v0"
    
## actor_critic:

    -stochastic_ac_dis: 离散动作的随机策略AC算法，solve的环境是"CartPole-v0"
    
    -stochastic_ac_con: 连续动作的随机策略AC算法，solve的环境是"Pendulum-v0"，收敛效果不好，要800 episode以后才能偶尔平衡，MovanZhou的也是这样
    
    -ddpg: deep deterministic poicy gradient算法实现，solve的环境是"Pendulum-v0"
 
and so on......
