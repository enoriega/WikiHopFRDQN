import itertools as it
from abc import abstractmethod

import logging
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers import *
import utils
# from bert.bert_orm import entity_origin_to_embeddings
from bert.PrecomputedBert import PrecomputedBert
from cache import Cache

logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
# logging.warning('This will get logged to a file')


def death_gradient(parameters):
    gradient = np.row_stack([p.grad.numpy().reshape(-1, 1) for p in parameters if p.grad is not None])
    return np.isclose(np.zeros(gradient.shape), gradient).astype(int).sum() / float(len(gradient))


# noinspection PyArgumentList
def process_input_data(datum:dict):
    """Converts a state datum into a sequence of features ready to be passed to the network"""
    # First, compute cartesian product of the candidate entities
    candidate_entities = [frozenset(e) for e in datum['candidates']]
    candidate_entity_types = [e for e in datum['candidatesTypes']]
    candidate_entity_origins = [e for e in datum['candidatesOrigins']]

    ranks, iteration_introductions, entity_usage, explore_scores, exploit_scores, types, origins = {}, {}, {}, {}, {}, {}, {}
    for ix, e in enumerate(candidate_entities):
        ranks[e] = datum['ranks'][ix]
        iteration_introductions[e] = datum['iterationsOfIntroduction'][ix]
        entity_usage[e] = datum['entityUsage'][ix]
        types[e] = candidate_entity_types[ix]
        origins[e] = candidate_entity_origins[ix]

    exploit_scores = datum['exploitScores']
    explore_scores = datum['exploreScores']
    same_components = datum['sameComponents']

    # Filter out a pair if they are the same entity
    candidate_pairs = [(a, b) for a, b in it.product(candidate_entities, candidate_entities) if a != b]
    features = datum['features']

    inputs = []
    for ix, (a, b) in enumerate(candidate_pairs):
        new_features = {
            'log_count_a': entity_usage[a],
            'log_count_b': entity_usage[b],
            'intro_a': iteration_introductions[a],
            'intro_b': iteration_introductions[b],
            'rank_a': ranks[a],
            'rank_b': ranks[b],
            'explore_score': explore_scores[ix],
            'exploit_score': exploit_scores[ix],
            'same_component': same_components[ix],
        }

        typeA = 'UNK' if len(types[a]) == 0 else types[a][0]
        typeB = 'UNK' if len(types[b]) == 0 else types[b][0]

        originsA = origins[a]
        originsB = origins[b]

        inputs.append({'features': {**features, **new_features}, 'A': a, 'B': b,
                       'typeA': typeA, 'typeB': typeB,
                       'originsA': originsA, 'originsB': originsB})

    return inputs, candidate_pairs


def zero_init(m):
    """Initializes the parameters of a module to Zero"""

    # Right now it considers only the parameters of a linear module
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)


class BaseApproximator(nn.Module):
    def __init__(self, num_feats, zero_init_params=False, device="cpu"):
        super().__init__()
        self.cache = Cache(size=1000)
        self.num_feats = num_feats

        self.device = device

        self.layers = self.build_layers(num_feats)

        if zero_init_params:
            self.layers.apply(zero_init)

    @abstractmethod
    def build_layers(self, num_feats):
        pass

    def backprop(self, data, gamma=0.9, alpha=1.0):

        # Parse the data
        data = [d for d in data if len(set(frozenset(e) for e in d['new_state']['candidates'])) > 1]
        states, states_ids, actions, rewards, next_states, next_state_ids = zip(*(d.values() for d in data))
        self.eval()
        action_values = self(states, ids=states_ids)
        self.train()
        # The next_action_values computation is tricky, as it involves looking at many possible states
        with torch.no_grad():
            pairs, next_action_values = self.select_action(next_states, ids=next_state_ids)

        # updates = [r + gamma * q.max() for r, q in zip(rewards, next_action_values.detach())]
        # This change shortcuts the update to not account for the next action state when the reward is observed, because
        # this is when the transition was to the final state, which by definition has a value of zero
        updates = [r if r != 0 else r + gamma * q.max() for r, q in zip(rewards, next_action_values.detach())]

        target_values = action_values.clone().detach()

        for row_ix, action in enumerate(actions):
            col_ix = 0 if action == "exploration" else 1
            target_values[row_ix, col_ix] += (alpha * (updates[row_ix] - target_values[row_ix, col_ix]))

        loss = F.mse_loss(action_values, target_values)

        return loss

    def select_action(self, data, ids=list()):

        ret_tensors = list()
        ret_pairs = list()

        all_inputs = list()
        all_inputs_ids = list()
        instance_indices = list()
        instance_candidate_pairs = list()

        for ix, (d, identifier) in enumerate(it.zip_longest(data, ids, fillvalue=None)):
            if identifier is not None and identifier in self.cache:
                inputs, candidate_pairs, inputs_ids = self.cache[identifier]
            else:
                inputs, candidate_pairs = process_input_data(d)
                inputs_ids = list()
                if identifier is not None:
                    inputs_ids = ["%s-%i" % (identifier, input_ix) for input_ix in range(len(inputs))]
                    self.cache[identifier] = (inputs, candidate_pairs, inputs_ids)

            instance_candidate_pairs.append(candidate_pairs)
            all_inputs.extend(inputs)
            all_inputs_ids.extend(inputs_ids)

            instance_indices.extend(it.repeat(ix, len(inputs)))

        action_values = self(all_inputs, ids=all_inputs_ids)

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


class BQN(BaseApproximator):
    """ This approximator user BERT instead of embeddings """

    def build_layers(self, num_feats):
        # Store the helper to use it on the forward method
        # This is necessary for the instance to recognize the embeddings as parameters of the network

        # The features, the aggregated embeddings for both
        k = num_feats + 200

        # Layers of the network
        return nn.Sequential(
            nn.Linear(k, 100),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 20),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(20, 2),
        )

    def __init__(self, num_feats, helper, zero_init_params, device):
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        super().__init__(num_feats, zero_init_params)
        self.bert_helper = helper
        # self.tokenizer = helper.tokenizer

        # embeddings_matrix = list(utils.get_bert_embeddings().parameters())[0]
        # Do PCA on the embeddings to take the top 50 components
        # embeddings_matrix = utils.pca(embeddings_matrix, 50)
        # self.embeddings = nn.EmbeddingBag.from_pretrained(embeddings_matrix, mode='mean')

        self.entity_types = {w: torch.tensor(ix, device=self.device) for ix, w in
                             enumerate(['UNK', 'Person', 'Location', 'Organization', 'CommonNoun'])}
        self.type_embeddings = nn.Embedding(len(self.entity_types), 50)
        self.emb_dropout = nn.Dropout(p=0.2)

        self.combinator = nn.Sequential(
            nn.Linear(self.config.hidden_size*2, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(self.config.hidden_size, 200),
            nn.Tanh(),
            nn.Dropout(p=0.2)
        )

    def forward(self, data, ids=list()):
        # Parse the input data into tensor form

        # Convert the raw data to tensor form
        batch_ids, batch_types, batch_features, batch_embeds = list(), list(), list(), list()

        sorted_features = list(sorted(data[0]['features']))

        # Create an input vector for each of the elements in data
        for datum, identifier in it.zip_longest(data, ids, fillvalue=None):
            if identifier is not None and identifier in self.cache:
                embeds, features = self.cache[identifier]
                batch_embeds.append(embeds)
                batch_features.append(features)
            else:
                with torch.no_grad():
                    # Get the raw input
                    features = datum['features']
                    entity_a = datum['A']
                    entity_b = datum['B']
                    origins_a = datum['originsA']
                    origins_b = datum['originsB']

                    ea_embeds = self.get_embeddings(entity_a, origins_a)
                    ea_embeds = ea_embeds.sum(dim=0)
                    ea_embeds /= ea_embeds.norm().detach()

                    eb_embeds = self.get_embeddings(entity_b, origins_b)
                    eb_embeds = eb_embeds.sum(dim=0)
                    eb_embeds /= eb_embeds.norm().detach()

                    embeds = torch.cat([ea_embeds, eb_embeds]).detach()
                    batch_embeds.append(embeds)

                    # Build a vector out of the numerical features, sorted by feature name
                    f = [float(features[k]) for k in sorted_features]
                    f = torch.tensor(f, device=self.device)
                    batch_features.append(f)

                    # Store in the cache the precomputed results
                    if identifier is not None:
                        self.cache[identifier] = (embeds, f)

        # Use the combination function of the entity embeddings
        comb = self.combinator(torch.stack(batch_embeds))

        f = torch.cat([torch.stack(batch_features), comb], dim=1)

        values = self.layers(f)

        return values

    def get_embeddings(self, entity_tokens, entity_origins) -> torch.Tensor:
        if len(entity_origins) > 0:
            try:
                ea_embeds = self.bert_helper.entity_origin_to_embeddings(entity_origins[0])
                logging.debug("Successful fetch")
            except Exception as e:
                ea_embeds = self.bert_helper.entity_to_embeddings(entity_tokens)
                logging.debug(repr(type(e)) + " on fetch " + str(e))
        else:
            ea_embeds = self.bert_helper.entity_to_embeddings(entity_tokens)
        return ea_embeds


class DQN(BaseApproximator):

    def __init__(self, num_feats, embeddings_helper, zero_init_params, device):
        self.e_helper = embeddings_helper
        super().__init__(num_feats, zero_init_params)
        self.fresh_embeddings = embeddings_helper.fresh_embeddings
        self.pretrained_embeddings = embeddings_helper.pretrained_embeddings
        self.dropout = nn.Dropout(p=0.25)

    def build_layers(self, num_feats):
        # Store the helper to use it on the forward method
        # This is necessary for the instance to recognize the embeddings as parameters of the network
        embeddings_helper = self.e_helper

        k = num_feats + embeddings_helper.dimensions() * 2

        # Layers of the network
        return nn.Sequential(
            nn.Linear(k, 20),
            nn.Tanh(),
            nn.Dropout(p=0.1),
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
            ea = self.dropout(self.e_helper.aggregated_embedding(entity_a))
            eb = self.dropout(self.e_helper.aggregated_embedding(entity_b))

            # Build a vector out of the numerical features, sorted by feature name
            f = [features[k] for k in sorted_features]
            f = torch.FloatTensor(f, device=self.device)

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

    def __init__(self, num_feats, helper, zero_init_params, device):
        super().__init__(num_feats, zero_init_params, device)

    def build_layers(self, num_feats):
        return nn.Sequential(
            nn.Linear(num_feats, 2),
            # nn.Sigmoid(),
        )

    def forward(self, data):
        # Parse the input data into tensor form
        # TODO uncomment this
        batch = self.tensor_form(data)

        # Feed the batch  through the network's layers
        # batch = data
        return self.forward_raw(batch)

    def forward_raw(self, batch):
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
            f = torch.FloatTensor(f, device=self.device)

            # Store it into a list
            batch.append(f)

        # Create an input matrix from all the elements (the batch)
        input_dim = batch[0].shape[0]
        batch = torch.cat(batch).view((len(batch), input_dim))

        return batch


class MLP(LinearQN):
    """This is a linear approximator that doesn't use the embeddings at all"""

    def __init__(self, num_feats, helper, zero_init_params, device):
        super().__init__(num_feats, helper, zero_init_params, device)

    def build_layers(self, num_feats):
        return nn.Sequential(
            nn.Linear(num_feats, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 2),
            # nn.Sigmoid(),
        )
