import dqn
from dqn import DQN, LinearQN, MLP, death_gradient
from embeddings import EmbeddingsHelper
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

with open('config.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

embeddings_cfg = cfg['embeddings']

glove_path = embeddings_cfg['glove_path']
voc_path = embeddings_cfg['voc_path']

model_dir = cfg['model_dir']

k = 10
gamma = 0.9

EXPLOIT, EXPLORE = 1, 2


def sample_reward(action, p):
    exploit_r = -10
    explore_r = 10

    coin = np.random.binomial(1, p)
    if coin == 0:
        return 0
    else:
        if action == EXPLOIT:
            return exploit_r
        else:
            return explore_r


def generate_data(size, p):
    states = np.random.random((size, k))
    next_states = np.random.random((size, k))
    # 1 - Exploit
    # 2 - Explore
    actions = np.random.randint(1, 3, size)
    rewards = np.asarray([sample_reward(a, p) for a in actions])

    states = torch.from_numpy(states).float()
    next_states = torch.from_numpy(next_states).float()
    actions = torch.from_numpy(actions).float()
    rewards = torch.from_numpy(rewards).float()
    return states, actions, rewards, next_states


def backprop(minibatch):
    states, actions, rewards, next_states = minibatch
    action_values = network.forward_raw(states)

    with torch.no_grad():
        next_action_values = network.forward_raw(next_states).detach()

    next_actions = ["Exploit" if na == 1 else "Explore" for na in next_action_values.argmax(dim=1)]
    updates = [r if r > 0 else r + gamma * q.max() for r, q in zip(rewards, next_action_values)]

    target_values = action_values.clone().detach()

    for row_ix, action in enumerate(actions):
        col_ix = 0 if action == EXPLORE else 1
        target_values[row_ix, col_ix] += (1.0*(updates[row_ix] - target_values[row_ix, col_ix]))

    loss = F.mse_loss(action_values, target_values)

    return loss, next_actions


def sample(data, size):
    states, actions, rewards, next_states = data

    indices = np.arange(len(states))

    ixs = np.random.choice(indices, size, replace=False)

    return states[ixs, : ], actions[ixs], rewards[ixs], next_states[ixs, :]


if __name__ == "__main__":
    helper = EmbeddingsHelper(glove_path, voc_path)
    network = LinearQN(k, helper, False)
    optimizer = optim.RMSprop(network.parameters())

    # Generate random transitions
    data = generate_data(100000, .05)

    rewards = pd.Series(data[2])
    print(rewards.value_counts())

    minibatch = sample(data, 1000)

    for i in range(1000):
        loss, next_actions = backprop(minibatch)
        next_actions = pd.Series(next_actions)
        print(loss)
        print(next_actions.value_counts())
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in network.parameters():
            param.grad.data.clamp_(-1, 1)

        x = death_gradient(network.parameters())

        print("Dead gradient proportion: %f" % x)
        optimizer.step()


