import itertools as it
from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


# noinspection PyArgumentList
def process_input_data(datum):
    """Converts a state datum into a sequence of features ready to be passed to the network"""
    # First, compute cartesian product of the candidate entities
    candidate_entities = [frozenset(e) for e in datum['candidates']]

    ranks, iteration_introductions, entity_usage = {}, {}, {}
    for ix, e in enumerate(candidate_entities):
        ranks[e] = datum['ranks'][ix]
        iteration_introductions[e] = datum['iterationsOfIntroduction'][ix]
        entity_usage[e] = datum['entityUsage'][ix]

    # Filter out a pair if they are the same entity
    candidate_pairs = [(a, b) for a, b in it.product(candidate_entities, candidate_entities) if a != b]
    features = datum['features']

    inputs = []
    for a, b in candidate_pairs:
        new_features = {
            'log_count_a': entity_usage[a],
            'log_count_b': entity_usage[b],
            'intro_a': iteration_introductions[a],
            'intro_b': iteration_introductions[b],
            'rank_a': ranks[a],
            'rank_b': ranks[b],
        }

        inputs.append({'features': {**features, **new_features}, 'A': a, 'B': b})

    return inputs, candidate_pairs


def zero_init(m):
    """Initializes the parameters of a module to Zero"""

    # Right now it considers only the parameters of a linear module
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)


class BaseApproximator(nn.Module):
    def __init__(self, num_feats, zero_init_params=False):
        super().__init__()

        self.num_feats = num_feats

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.layers = self.build_layers(num_feats)

        if zero_init_params:
            self.layers.apply(zero_init)

    @abstractmethod
    def build_layers(self, num_feats):
        pass

    def backprop(self, data, gamma=0.9):

        # Parse the data
        data = [d for d in data if len(set(frozenset(e) for e in d['new_state']['candidates'])) > 1]
        states, actions, rewards, next_states = zip(*(d.values() for d in data))
        action_values = self(states)
        # The next_action_values computation is tricky, as it involves looking at many possible states
        with torch.no_grad():
            pairs, next_action_values = self.select_action(next_states)

        updates = [r + gamma * q.max() for r, q in zip(rewards, next_action_values.detach())]

        target_values = action_values.clone().detach()

        for row_ix, action in enumerate(actions):
            col_ix = 0 if action == "exploration" else 1
            target_values[row_ix, col_ix] += (updates[row_ix] - target_values[row_ix, col_ix])

        loss = F.mse_loss(action_values, target_values)

        return loss

    def select_action(self, data):

        ret_tensors = list()
        ret_pairs = list()

        all_inputs = list()
        instance_indices = list()
        instance_candidate_pairs = list()

        for ix, d in enumerate(data):
            inputs, candidate_pairs = process_input_data(d)
            instance_candidate_pairs.append(candidate_pairs)
            all_inputs.extend(inputs)
            instance_indices.extend(it.repeat(ix, len(inputs)))

        action_values = self(all_inputs)

        # Create slice views of the tensors
        groups = it.groupby(enumerate(instance_indices), lambda t: t[1])
        for instance_num, g in groups:
            indices = [t[0] for t in g]
            tensor_slice = action_values[indices, :]
            # Compute the row with the highest index
            row_ix = tensor_slice.max(dim=1).values.argmax()
            row = tensor_slice[row_ix, :]
            ret_tensors.append(row)
            ret_pairs.append(instance_candidate_pairs[instance_num][row_ix])

        return ret_pairs, torch.cat(ret_tensors).view((len(ret_tensors), -1))

    @staticmethod
    def raw2json(raw_vals):
        ret = list()

        for row in raw_vals.split(1):
            row = row.squeeze()
            ret.append({"Exploration": row[0].item(), "Exploitation": row[1].item()})

        return ret


class DQN(BaseApproximator):

    def __init__(self, num_feats, embeddings_helper, zero_init_params):
        self.e_helper = embeddings_helper
        super().__init__(num_feats, zero_init_params)
        self.fresh_embeddings = embeddings_helper.fresh_embeddings
        self.pretrained_embeddings = embeddings_helper.pretrained_embeddings

    def build_layers(self, num_feats):
        # Store the helper to use it on the forward method
        # This is necessary for the instance to recognize the embeddings as parameters of the network
        embeddings_helper = self.e_helper

        k = num_feats + embeddings_helper.dimensions() * 2

        # Layers of the network
        return nn.Sequential(
            nn.Linear(k, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

    def forward(self, data):
        # Parse the input data into tensor form
        batch = self.tensor_form(data)

        # Feed the batch  through the network's layers
        values = self.layers(batch)

        return values

    def tensor_form(self, data):
        # Convert the raw data to tensor form
        batch = list()

        sorted_features = list(sorted(data[0]['features']))

        # Create an input vector for each of the elements in data
        for datum in data:
            # Get the raw input
            features = datum['features']
            entity_a = datum['A']
            entity_b = datum['B']

            # Query the helper for the aggregated embeddings of the entities
            ea = self.e_helper.aggregated_embedding(entity_a)
            eb = self.e_helper.aggregated_embedding(entity_b)

            # Build a vector out of the numerical features, sorted by feature name
            f = [features[k] for k in sorted_features]
            f = torch.FloatTensor(f).to(device=self.device)

            # Concatenate them into a single input vector for this instance
            f = torch.cat([f, ea, eb])

            # Store it into a list
            batch.append(f)

        # Create an input matrix from all the elements (the batch)
        input_dim = batch[0].shape[0]
        batch = torch.cat(batch).view((len(batch), input_dim))

        return batch


class LinearQN(BaseApproximator):
    """This is a linear approximator that doesn't use the embeddings at all"""

    def __init__(self, num_feats, helper, zero_init_params):
        super().__init__(num_feats, zero_init_params)

    def build_layers(self, num_feats):
        return nn.Sequential(
            nn.Linear(num_feats, 2),
            nn.Sigmoid(),
        )

    def forward(self, data):
        # Parse the input data into tensor form
        batch = self.tensor_form(data)

        # Feed the batch  through the network's layers
        values = self.layers(batch)

        return values

    def tensor_form(self, data):
        """
            This is an overridden version that ignores the entities (doesn't fetch their embeddings)
        """
        # Convert the raw data to tensor form
        batch = list()

        sorted_features = list(sorted(data[0]['features']))

        # Create an input vector for each of the elements in data
        for datum in data:
            # Get the raw input
            features = datum['features']

            # Build a vector out of the numerical features, sorted by feature name
            f = [features[k] for k in sorted_features]
            f = torch.FloatTensor(f).to(device=self.device)

            # Store it into a list
            batch.append(f)

        # Create an input matrix from all the elements (the batch)
        input_dim = batch[0].shape[0]
        batch = torch.cat(batch).view((len(batch), input_dim))

        return batch


class MLP(LinearQN):
    """This is a linear approximator that doesn't use the embeddings at all"""

    def __init__(self, num_feats, helper, zero_init_params):
        super().__init__(num_feats, helper, zero_init_params)

    def build_layers(self, num_feats):
        return nn.Sequential(
            nn.Linear(num_feats, 100),
            nn.Tanh(),
            nn.Linear(100, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
            # nn.Sigmoid(),
        )
