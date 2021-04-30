# Deep-Q-Learning---CartPole
Solving the traditional CartPole problem in Reinforcement Learning with Deep NNs in the Deep Q Learning Framework

Let $S$, $A$ denote the set of all possible states and actions respectively. (known)

The reward distribution defined on the set of all possible states and actions is denoted by by
$R : (s, a) \in S \times A \rightarrow R(s, a)$ (known env.step(action))

$\gamma$ is the discount factor. The discount factor weights the importance of the expected future rewards at times $t, t+1, \dots$
(parameter defined by user and may change the optimal policy function $\pi^{*}$ defined below)

We want to find the optimal policy $\pi^{*} :S \rightarrow  A$ that gives the action at any given state that maximises the expexted reward, $\max \sum_{t \geq 0} \gamma^{t}R^{t}$

Problem Statement

Find $\pi^{*} :S \rightarrow  A$ such that $\forall s \in S$ $\pi^{*}(s) = \max \sum_{t \geq 0} \gamma^{t}R^{t}$

Given a state, $s \in S$ the expected cumulative reward achieved by the policy $\pi$ is given by the value function $V^{\pi} : S \rightarrow \mathbb{R} $
(unknown)

The Q-value function $Q^{\pi} : S \times A \rightarrow \mathbb{R} $ takes this one-step further, giving the expected cumulative reward achieved by the policy $\pi$ given a state $s \in S$ and action $a \in A$
(unknown)

We approximate $Q^{\pi}$ by a Deep Q Learning Network (alternatively, we can discretise the system and try to learn the Q-table). Then, by optimisation with the fitness function we can determine the optimal policy $\pi^{*}$ that maximises the Q-value function.

The strategy is as folows. 

1.   Initialise the DNN Q-value function with random weights.
2.   Run an agent on a (mini-batch) game, taking actions that maximise the expected reward of the given DNN Q-function. (Of course, the DNN Q-value function is not the true Q-value function)
3.  We fit (replay method) the DNN Q-value function it acccording to the actual reward achieved in the mini-batch of the game. In this way we train the DNN Q -value function to better approximate the actual Q value. 
4. Train the DNN Q-value function on mini batches so that we get suffcient trainig data to make this approximation as accurate as possible. (i.e repeat steps 2 to 4)
5. Run our agent such that it takes actions that maximises the Q-value function.
