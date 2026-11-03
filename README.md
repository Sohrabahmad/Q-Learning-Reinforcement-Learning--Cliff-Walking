# Q-Learning Reinforcement Learning — Cliff Walking

## 📌 Project Overview

This project implements **Q-Learning**, an off-policy reinforcement learning algorithm, using the Cliff Walking environment from Gymnasium.

The agent learns to navigate a grid-based environment, starting from an initial position and reaching a goal while avoiding dangerous cliff cells.

The project uses a Q-table to store state-action values and an epsilon-greedy strategy to balance exploration and exploitation. Through repeated training episodes, the agent learns an optimal or near-optimal navigation policy.

## 🎯 Objectives

- Understand the fundamentals of Reinforcement Learning.
- Implement Q-Learning using Python and NumPy.
- Explore the Cliff Walking environment using Gymnasium.
- Initialize and update a Q-table.
- Implement an epsilon-greedy action-selection policy.
- Balance exploration and exploitation.
- Train an agent over multiple episodes.
- Evaluate the learned policy.
- Visualize the agent's navigation behavior.

## 🛠️ Technologies Used

- **Python** — Programming language
- **NumPy** — Q-table and numerical calculations
- **Gymnasium** — Reinforcement learning environment
- **Random** — Exploration and action selection
- **Jupyter Notebook** — Interactive development environment

## 📂 Project Structure

```text
Q-Learning/
│
├── Q_Learning.ipynb
└── README.md
```

## 🎮 Environment Description

The project uses the built-in `CliffWalking-v1` environment from Gymnasium.

The environment contains a grid with:

- **Starting position:** Where the agent begins.
- **Goal:** The destination the agent must reach.
- **Cliff:** Dangerous cells that result in a large negative reward.
- **Actions:** Four possible movement directions.

### Environment Configuration

```python
import gymnasium as gym

env = gym.make("CliffWalking-v1")
```

| Property | Value |
|---|---:|
| Grid dimensions | 4 × 12 |
| Total states | 48 |
| Available actions | 4 |
| Training episodes | 500 |
| Algorithm | Q-Learning |

## ⚙️ Project Workflow

### 1. Import Libraries

```python
import gymnasium as gym
import numpy as np
import random
```

### 2. Define Q-Learning Hyperparameters

The notebook uses the following configuration:

```python
gamma = 0.99
epsilon = 0.1
alpha = 0.5
episodes = 500
```

| Parameter | Value | Description |
|---|---:|---|
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Epsilon (ε) | 0.1 | Exploration probability |
| Alpha (α) | 0.5 | Learning rate |
| Episodes | 500 | Total training episodes |

### 3. Initialize the Q-Table

The Q-table stores the estimated value of taking each action in each environment state.

```python
Q = np.zeros((48, 4))
```

- 48 rows represent environment states.
- 4 columns represent available actions.
- All Q-values are initially set to zero.

### 4. Implement Epsilon-Greedy Policy

The epsilon-greedy strategy helps the agent balance exploration and exploitation.

```python
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
```

**Exploration:** The agent selects a random action with probability 0.1.

**Exploitation:** The agent selects the action with the highest estimated Q-value with probability 0.9.

### 5. Implement the Q-Learning Algorithm

The notebook trains the agent for 500 episodes.

The Q-Learning update equation is:

\[
Q(s,a) \leftarrow Q(s,a) +
\alpha [r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\]

Where:

- \(s\) = Current state
- \(a\) = Current action
- \(r\) = Immediate reward
- \(s'\) = Next state
- \(a'\) = Possible next action
- \(\alpha\) = Learning rate
- \(\gamma\) = Discount factor

**Implementation:**

```python
for episode in range(episodes):

    done = False
    state, _ = env.reset()

    while not done:

        action = epsilon_greedy(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        Q[state, action] += alpha * (
            reward +
            gamma * np.max(Q[next_state]) -
            Q[state, action]
        )

        state = next_state

    env.close()
```

The notebook also renders the environment every 50 episodes to visualize the agent's learning behavior.

### 6. Evaluate the Learned Policy

After training, the agent follows a greedy policy by selecting the action with the highest Q-value.

```python
env = gym.make(
    "CliffWalking-v1",
    render_mode="human"
)

state, _ = env.reset()

total_reward = 0
episode_len = 0
done = False

while not done:

    action = np.argmax(Q[state])

    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    episode_len += 1
    total_reward += reward

print(
    f"Total reward = {total_reward} "
    f"& Episode length = {episode_len}"
)

env.close()
```

This evaluation demonstrates how the trained Q-table guides the agent toward the goal.

## 📈 Results and Observations

<badge color="success">Successful goal reached</badge>

The notebook's recorded final evaluation output is:

| Evaluation Metric | Result |
|---|---:|
| Training episodes | 500 |
| Learning rate | 0.5 |
| Discount factor | 0.99 |
| Exploration probability | 0.1 |
| Final total reward | -13 |
| Final episode length | 13 steps |

The agent successfully reaches the goal in 13 steps with a total reward of -13 in the recorded evaluation.

The notebook also displays selected Q-table entries for states 36 and 35.

## 🔍 Key Reinforcement Learning Concepts

| Concept | Description |
|---|---|
| Reinforcement Learning | Learning through interaction with an environment. |
| Agent | The learner that selects actions. |
| Environment | The world in which the agent operates. |
| State | Current situation or position of the agent. |
| Action | A decision taken by the agent. |
| Reward | Feedback received after an action. |
| Q-Table | Stores estimated state-action values. |
| Exploration | Trying different actions to discover better strategies. |
| Exploitation | Selecting the action with the highest estimated value. |
| Q-Learning | An off-policy temporal-difference learning algorithm. |

## 🆚 Q-Learning vs SARSA

| Q-Learning | SARSA |
|---|---|
| Off-policy algorithm | On-policy algorithm |
| Uses the maximum next-state Q-value | Uses the Q-value of the next selected action |
| Learns toward a greedy target policy | Learns using the behavior policy |
| Update uses \(\max_{a'}Q(s',a')\) | Update uses \(Q(s',a')\) |

The main difference is how the two algorithms calculate their next-state value during the Q-table update.

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Q-Learning
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy gymnasium "gymnasium[toy-text]" jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook Q_Learning.ipynb
```

Run the notebook cells sequentially to initialize the environment, train the agent, and visualize the learned policy.

**Note:** The notebook uses `render_mode="human"` for visualization, so a compatible graphical display environment is required.

## 🚀 Applications

Q-Learning can be applied to:

- Robot navigation
- Game-playing agents
- Pathfinding
- Autonomous decision-making
- Resource allocation
- Traffic signal optimization
- Sequential decision-making problems

## 🔮 Future Improvements

- Plot cumulative rewards across training episodes.
- Add a moving average reward graph.
- Compare Q-Learning with SARSA.
- Implement epsilon decay to gradually reduce exploration.
- Experiment with different learning rates and discount factors.
- Evaluate the learned policy across multiple episodes.
- Save and load the trained Q-table.
- Visualize the learned policy using directional arrows.
- Compare different exploration strategies.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute it under a specific open-source license.
