# Reinforcement Learning
This repository contains the implementation of Rainbow DQN.
paper: [link](https://arxiv.org/pdf/1710.02298.pdf)

# DQN
## Vanilla DQN
## Double DQN


## Prioritized Experience Replay (PER)
paper: [link](https://arxiv.org/pdf/1511.05952.pdf)

### Main Idea
1. Each transition in the replay memory is not equal, and we should not sample uniformly.
2. New transitions get highest priority. After that, each transition is weighted by the TD error, meaning 'how surprise' the network is.
3. Use segment tree to implement sampling.

### Insights
* **Prioritized Supervised Learning:** Sample non-uniformly from a dataset using a priority based on its last-seen error, similarly to boosting and hard negative mining.
* **Off-policy Replay:** Two standard approaches to off-policy RL are rejection sampling and using importance sampling ratios $\rho$ to correct for how likely a transition would have been on-policy.
* **Feedback for Exploration:** We could generate more transitions that lead to useful learnings.

## Dueling

![Dueling Network Illustration](https://i.ibb.co/HNMyjm1/dueling-network.png)

paper: [link](https://arxiv.org/pdf/1511.06581.pdf)

### Main Idea
1. It's unecessary to estimate each action's value.
2. Papers in 1993 had the idea of maintaining separate value and advantage functions, $V(s), A(s,a)$. Dueling architecture keeps two streams, and combine them to estimate $Q(s, A)$.
3. By looking at $Q(s, A) = A(s, A) + V(s)$, we notice that it is *unidentifiable* because when we train the network, our truth is the reward, which is used to compare with $Q(s, A)$ to calculate the loss. Now $Q(s, A)$ is fixed, we have two streams to estimate $A$, $V$. We can subtract any amount from $A$ and add the same amount to $V$ and $Q$ won't change. This makes it unidentifiable and thus not stable. To fix this problem, we subtract the average of all $A(s, A)$. This is because we want $A$ function estimator to have zero advantage if act greedily. We want to subtract $max$, but average makes it more stable.