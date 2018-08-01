# RL101
Ruqirements: 需安装gym和tensorflow
random_agents:
    -random_agents: 简单实现gym环境下的agent与环境交互
dqn:
    -dqn_dis: 离散动作的dqn，solve的环境是"CortPole-v0"
    -dqn_con: 连续动作的dqn，solve的环境是"Pendulum-v0"，将连续动作离散化，8000步以上可以收敛
policy_gradient:
    -reinforce_dis: 离散动作的REINFORCE算法，solve的环境是"CortPole-v0"
    -reinforce_con: 连续动作的REINFORCE算法，solve的环境是"Pendulum-v0"，不能收敛，不知为何
actor_critic:
    -ac_con: 连续动作的REINFORCE算法，solve的环境是"Pendulum-v0"，收敛效果不好，要400episode以后才能偶尔平衡，MovanZhou的也是这样
 
and so on......
