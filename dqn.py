import itertools as it
from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers import *
import utils

from embeddings import EmbeddingsHelper


def death_gradient(parameters):
    gradient = np.row_stack([p.grad.numpy().reshape(-1, 1) for p in parameters if p.grad is not None])
    return np.isclose(np.zeros(gradient.shape), gradient).astype(int).sum() / float(len(gradient))


# noinspection PyArgumentList
def process_input_data(datum):
    """Converts a state datum into a sequence of features ready to be passed to the network"""
    # First, compute cartesian product of the candidate entities
    candidate_entities = [frozenset(e) for e in datum['candidates']]

    ranks, iteration_introductions, entity_usage, explore_scores, exploit_scores = {}, {}, {}, {}, {}
    for ix, e in enumerate(candidate_entities):
        ranks[e] = datum['ranks'][ix]
        iteration_introductions[e] = datum['iterationsOfIntroduction'][ix]
        entity_usage[e] = datum['entityUsage'][ix]

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

        inputs.append({'features': {**features, **new_features}, 'A': a, 'B': b})

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

        self.num_feats = num_feats

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        states, actions, rewards, next_states = zip(*(d.values() for d in data))
        self.eval()
        action_values = self(states)
        self.train()
        # The next_action_values computation is tricky, as it involves looking at many possible states
        with torch.no_grad():
            pairs, next_action_values = self.select_action(next_states)

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


class FullBQN(BaseApproximator):
    """ This approximator user BERT """

    def build_layers(self, num_feats):
        # Store the helper to use it on the forward method
        # This is necessary for the instance to recognize the embeddings as parameters of the network

        config = BertConfig.from_pretrained('bert-base-uncased')
        k = num_feats + config.hidden_size

        # Layers of the network
        return nn.Sequential(
            nn.Linear(k, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

    def __init__(self, num_feats, helper, zero_init_params, device):
        super().__init__(num_feats, zero_init_params, device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, data):
        # Parse the input data into tensor form
        bert_ids, bert_types, feature_matrix = self.tensor_form(data)

        # Retrieve the un-processed pooler outputs from the last hidden states
        pooled_vectors = list()
        for ids, types in zip(bert_ids, bert_types):
            last_hidden_state, _ = self.bert(ids, token_type_ids=types)
            del _
            cls_h = last_hidden_state[0, 0, :]
            pooled_vectors.append(cls_h)

        pooled_vectors = torch.stack(pooled_vectors)
        matrix = torch.cat([feature_matrix, pooled_vectors], dim=1)
        # Feed the batch  through the network's layers
        values = self.layers(matrix)

        del bert_ids, bert_types, feature_matrix, pooled_vectors, matrix

        return values

    def tensor_form(self, data):
        # Convert the raw data to tensor form
        batch_ids, batch_types, batch_features = list(), list(), list()

        tokenizer = self.tokenizer
        sorted_features = list(sorted(data[0]['features']))

        # Create an input vector for each of the elements in data
        for datum in data:
            # Get the raw input
            features = datum['features']
            entity_a = datum['A']
            entity_b = datum['B']

            # Put together the entity pair as a sequence for BERT's

            ea_tokens = list(it.chain.from_iterable(tokenizer.tokenize(t) for t in sorted(entity_a)))
            eb_tokens = list(it.chain.from_iterable(tokenizer.tokenize(t) for t in sorted(entity_b)))

            bert_tokens = ['[CLS]'] + \
                          ea_tokens + \
                          ['[SEP]'] + \
                          eb_tokens + \
                          ['[SEP]']

            bert_token_types = torch.tensor([[0] * (len(ea_tokens) + 2) + [1] * (len(eb_tokens) + 1)], device=self.device)
            batch_types.append(bert_token_types)

            bert_ids = torch.tensor([tokenizer.convert_tokens_to_ids(bert_tokens)], device=self.device)

            # Build a vector out of the numerical features, sorted by feature name
            f = [features[k] for k in sorted_features]
            f = torch.FloatTensor(f, device=self.device)

            # Concatenate them into a single input vector for this instance
            batch_ids.append(bert_ids)
            batch_features.append(f)

        # Create an input matrix from all the elements (the batch)
        num_features = batch_features[0].shape[0]
        # batch = torch.cat(batch).view((len(batch), input_dim))
        bert_inputs = batch_ids
        bert_types = batch_types
        feature_matrix = torch.cat(batch_features).view(len(batch_features), num_features)

        return bert_inputs, bert_types, feature_matrix


class BQN(BaseApproximator):
    """ This approximator user BERT instead of embeddings """

    def build_layers(self, num_feats):
        # Store the helper to use it on the forward method
        # This is necessary for the instance to recognize the embeddings as parameters of the network

        # config = BertConfig.from_pretrained('bert-base-uncased')
        k = num_feats + 50 * 2#(config.hidden_size * 2)

        # Layers of the network
        return nn.Sequential(
            nn.Linear(k, 20),
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            nn.Linear(20, 2),
        )

    def __init__(self, num_feats, helper, zero_init_params, device):
        super().__init__(num_feats, zero_init_params)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        embeddings_matrix = list(utils.get_bert_embeddings().parameters())[0]
        # Do PCA on the embeddings to take the top 50 components
        embeddings_matrix = utils.pca(embeddings_matrix, 50)
        self.embeddings = nn.EmbeddingBag.from_pretrained(embeddings_matrix, mode='mean')
        self.emb_dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        # Parse the input data into tensor form

        # Convert the raw data to tensor form
        batch_ids, batch_types, batch_features = list(), list(), list()

        tokenizer = self.tokenizer
        emb_dropout = self.emb_dropout
        sorted_features = list(sorted(data[0]['features']))

        # Create an input vector for each of the elements in data
        for datum in data:
            # Get the raw input
            features = datum['features']
            entity_a = datum['A']
            entity_b = datum['B']

            # Put together the entity pair as a sequence for BERT's
            ea_tokens = list(it.chain.from_iterable(tokenizer.tokenize(t) for t in sorted(entity_a)))
            eb_tokens = list(it.chain.from_iterable(tokenizer.tokenize(t) for t in sorted(entity_b)))

            ea_ids = torch.tensor([tokenizer.convert_tokens_to_ids(ea_tokens)], device=self.device)
            eb_ids = torch.tensor([tokenizer.convert_tokens_to_ids(eb_tokens)], device=self.device)

            # ea_embeds = emb_dropout(EmbeddingsHelper.aggregate_embeddings(self.embeddings(ea_ids)))
            # eb_embeds = emb_dropout(EmbeddingsHelper.aggregate_embeddings(self.embeddings(eb_ids)))

            ea_embeds = self.embeddings(ea_ids).squeeze()
            eb_embeds = self.embeddings(eb_ids).squeeze()

            # Build a vector out of the numerical features, sorted by feature name
            f = [float(features[k]) for k in sorted_features]
            f = torch.tensor(f, device=self.device)

            f = torch.cat([f, ea_embeds, eb_embeds])

            # Concatenate them into a single input vector for this instance
            batch_features.append(f)

        # Create an input matrix from all the elements (the batch)
        num_features = batch_features[0].shape[0]

        feature_matrix = torch.cat(batch_features).view(len(batch_features), num_features)

        values = self.layers(feature_matrix)

        return values


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
