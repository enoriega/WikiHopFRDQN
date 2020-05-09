import itertools as it
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import *

# from bert.bert_orm import entity_origin_to_embeddings
from cache import Cache

# import logging logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(name)s - %(
# levelname)s - %(message)s') logging.warning('This will get logged to a file')

EXPLORATION = 0
EXPLOITATION = 1


def death_gradient(parameters):
    gradient = np.row_stack([p.grad.numpy().reshape(-1, 1) for p in parameters if p.grad is not None])
    return np.isclose(np.zeros(gradient.shape), gradient).astype(int).sum() / float(len(gradient))


def zero_init(m):
    """Initializes the parameters of a module to Zero"""

    # Right now it considers only the parameters of a linear module
    if type(m) == nn.Linear:
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)


class BaseApproximator(nn.Module):
    def __init__(self, num_feats, zero_init_params=False, device="cpu", use_embeddings=False):
        super().__init__()
        self.cache = Cache(size=1000)
        self.num_feats = num_feats

        self.device = device
        self.use_embeddings = use_embeddings

        self.layers = self.build_layers(num_feats)

        if zero_init_params:
            self.layers.apply(zero_init)

    @abstractmethod
    def build_layers(self, num_feats):
        pass

    @abstractmethod
    def dictionary_to_tensor(self, data, use_embeddings):
        pass

    def backprop(self, data, gamma=0.8, alpha=1.0):

        state = data['state']
        action = data['action']
        reward = data['reward']
        next_state = data['next_state']
        next_action = data['next_action']

        self.eval()
        action_values = self(self.dictionary_to_tensor(state, self.use_embeddings))
        self.train()
        # The next_action_values computation is tricky, as it involves looking at many possible states
        with torch.no_grad():
            next_action_values = self(self.dictionary_to_tensor(next_state, self.use_embeddings))

        # updates = [r + gamma * q.max() for r, q in zip(rewards, next_action_values.detach())]
        # This change shortcuts the update to not account for the next action state when the reward is observed, because
        # this is when the transition was to the final state, which by definition has a value of zero
        # TODO Uncomment for QLearning
        # update = reward if reward != 0 else reward + gamma * next_action_values.max()
        # TODO Uncomment for SARSA
        update = reward if reward != 0 else reward + gamma * next_action_values[next_action]

        target_values = action_values.clone().detach()

        # target_values[col_ix] += (alpha * (update - target_values[col_ix]))
        target_values[action] = update

        # loss = F.smooth_l1_loss(action_values, target_values)
        loss = F.mse_loss(action_values, target_values)

        return loss

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
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(self.config.hidden_size, 200),
            nn.Tanh(),
            nn.Dropout(p=0.2)
        )

    def make_input(self, datum, sorted_features):
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

            # Build a vector out of the numerical features, sorted by feature name
            f = [float(features[k]) for k in sorted_features]
            f = torch.tensor(f, device=self.device)

            return embeds, f

    def forward(self, data, ids=None):

        # Best practice
        if ids is None:
            ids = list()

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
                # with torch.no_grad():
                #     # Get the raw input
                #     features = datum['features']
                #     entity_a = datum['A']
                #     entity_b = datum['B']
                #     origins_a = datum['originsA']
                #     origins_b = datum['originsB']
                #
                #     ea_embeds = self.get_embeddings(entity_a, origins_a)
                #     ea_embeds = ea_embeds.sum(dim=0)
                #     ea_embeds /= ea_embeds.norm().detach()
                #
                #     eb_embeds = self.get_embeddings(entity_b, origins_b)
                #     eb_embeds = eb_embeds.sum(dim=0)
                #     eb_embeds /= eb_embeds.norm().detach()
                #
                #     embeds = torch.cat([ea_embeds, eb_embeds]).detach()
                #     batch_embeds.append(embeds)
                #
                #     # Build a vector out of the numerical features, sorted by feature name
                #     f = [float(features[k]) for k in sorted_features]
                #     f = torch.tensor(f, device=self.device)
                #     batch_features.append(f)

                embeds, f = self.make_input(datum, sorted_features)
                batch_embeds.append(embeds)
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

    def backprop(self, data, gamma=0.9, alpha=1.0):
        loss = super().backprop(data, gamma, alpha)
        self.bert_helper.clear_cache()
        return loss


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

    def forward(self, data, ids=None):
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
    PA_OOV = torch.tensor([0])
    PB_OOV = torch.tensor([1])

    def __init__(self, num_feats, helper, zero_init_params, device, use_embeddings):
        super().__init__(num_feats, zero_init_params, device, use_embeddings)
        self.embeddings = helper
        self.OOVEmbeddings = nn.Embedding(2, 100)

    def build_layers(self, num_feats):
        return nn.Sequential(
            nn.Linear(num_feats, 2),
            # nn.Sigmoid(),
        )

    def forward(self, data):
        values = self.layers(data)
        return values

    def dictionary_to_tensor(self, data, embeddings):
        features = [data[k] for k in sorted(data) if "Lemma_" not in k]
        features = torch.FloatTensor(features)

        def group_participant_features(data, prefix):
            keys = sorted(filter(lambda k: k.startswith(prefix), data))
            grouped_keys = it.groupby(keys, key=lambda k: k.split("_")[1])

            lemmas = list()
            for group, values in grouped_keys:
                indices = [int(data[k]) for k in values]
                lemmas.append(indices)

            return lemmas

        if embeddings:

            pa_indices = group_participant_features(data, "paLemma")
            pa_indices = [torch.tensor([pa]) for pa in pa_indices]
            pb_indices = group_participant_features(data, "pbLemma")
            pb_indices = [torch.tensor([pb]) for pb in pb_indices]

            if len(pa_indices) > 0:
                pa = torch.cat([self.embeddings(ix) for ix in pa_indices]).mean(dim=0)
            else:
                pa = self.OOVEmbeddings(LinearQN.PA_OOV).squeeze()

            if len(pb_indices) > 0:
                pb = torch.cat([self.embeddings(ix) for ix in pb_indices]).mean(dim=0)
            else:
                pb = self.OOVEmbeddings(LinearQN.PB_OOV).squeeze()

            return torch.cat([features, pa, pb])

        else:
            return features


class MLP(LinearQN):
    """This is a linear approximator that doesn't use the embeddings at all"""

    def __init__(self, num_feats, helper, zero_init_params, device, use_embeddings):
        super().__init__(num_feats, helper, zero_init_params, device, use_embeddings)

    def build_layers(self, num_feats):
        # return nn.Sequential(
        #     nn.Linear(num_feats, 100),
        #     nn.LeakyReLU(),
        #     nn.Dropout(),
        #     nn.Linear(100, 20),
        #     nn.LeakyReLU(),
        #     nn.Dropout(),
        #     nn.Linear(20, 10),
        #     nn.LeakyReLU(),
        #     nn.Dropout(),
        #     nn.Linear(10, 2),
        # )

        return nn.Sequential(
            nn.Linear(num_feats, 100),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(100, 2),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(20, 10),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(10, 2),
        )

    def forward(self, data):
        values = self.layers(data)
        return values
