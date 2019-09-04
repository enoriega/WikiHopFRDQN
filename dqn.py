import itertools as it

import torch
import torch.nn.functional as F
from torch import nn


# noinspection PyArgumentList
class DQN(nn.Module):

    def __init__(self, num_feats, embeddings_helper):
        super().__init__()

        # Store the helper to use it on the forward method
        self.e_helper = embeddings_helper
        # This is necessary for the instance to recognize the embeddings as parameters of the network
        self.pretrained_embeddings = embeddings_helper.pretrained_embeddings
        self.fresh_embeddings = embeddings_helper.fresh_embeddings

        k = num_feats + embeddings_helper.dimensions() * 2

        # Layers of the network
        self.layers = nn.Sequential(
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

    def backprop(self, data, gamma=0.9):

        # Parse the data
        data = [d for d in data if len(d['new_state']['candidates']) > 0]
        states, actions, rewards, next_states = zip(*(d.values() for d in data))
        action_values = self(states)
        # The next_action_values computation is tricky, as it involves looking at many possible states
        with torch.no_grad():
            pairs, next_action_values = self.select_action(next_states)

        updates = [r + gamma * q.max() for r, q in zip(rewards, next_action_values.detach())]

        target_values = action_values.clone().detach()

        for row_ix, action in enumerate(actions):
            col_ix = 0 if action == "exploration" else 1
            target_values[row_ix, col_ix] = updates[row_ix]

        loss = F.mse_loss(action_values, target_values)

        return loss

    def select_action(self, data):

        ret_tensors = list()
        ret_pairs = list()

        for d in data:
            # First, compute cartesian product of the candidate entities
            candidate_entities = d['candidates']
            candidate_pairs = list(it.product(candidate_entities, candidate_entities))
            inputs = [{'features': d['features'], 'A': a, 'B': b} for a, b in candidate_pairs]
            action_values = self(inputs)
            # Get the index of the row with the max value
            max_val = float("-inf")
            row_ix = 0
            row_vals = None

            for ix, row in enumerate(action_values):
                row_max = row.max()
                if row_max > max_val:
                    max_val = row_max
                    row_ix = ix
                    row_vals = row

            ret_tensors.append(row_vals)
            ret_pairs.append(candidate_pairs[row_ix])

        return ret_pairs, torch.cat(ret_tensors).view((len(ret_tensors), -1))

    def tensor_form(self, data):
        # Convert the raw data to tensor form
        batch = list()

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
            f = [features[k] for k in sorted(features)]
            f = torch.FloatTensor(f)

            # Concatenate them into a single input vector for this instance
            f = torch.cat([f, ea, eb])

            # Store it into a list
            batch.append(f)

        # Create an input matrix from all the elements (the batch)
        input_dim = batch[0].shape[0]
        batch = torch.cat(batch).view((len(batch), input_dim))

        return batch

    @staticmethod
    def raw2json(raw_vals):
        ret = list()

        for row in raw_vals.split(1):
            row = row.squeeze()
            ret.append({"Exploration": row[0].item(), "Exploitation": row[1].item()})

        return ret

# if __name__ == "__main__":
#     glove_path = "/Users/enrique/github/WikiHopFR/glove/glove.6B.50d.txt"
#     voc_path = "/Users/enrique/github/WikiHopFR/w2vvoc.txt"
#
#     helper = EmbeddingsHelper(glove_path, voc_path)
#
#     test_data = [{
#         'features': {
#             "iteration": float(5),
#             "alpha": float(10.3)
#         },
#         'A': ['enrique', 'noriega'],
#         'B': ['Cecilia', 'Montano']
#     }, {
#         'features': {
#             "iteration": float(34),
#             "alpha": float(15.0)
#         },
#         'A': ['Diego', 'Andres'],
#         'B': ['Gina', 'Chistosina']
#     }]
#
#     k = len(test_data[0]['features'])
#
#     network = DQN(k, helper)
#
#     with torch.no_grad():
#         vals = network(test_data)
#
#     x = DQN.raw2json(vals)
#
#     # embeddings = nn.Embedding.from_pretrained(torch.from_numpy(helper.matrix))
#
#     # entity_a = ["hermosillo", "sonora"]
#     # entity_b = ["tucson", "arizona"]
#     # dist = helper.distance(entity_a, entity_b)
#     # words = ['the', 'rain', 'mexico', 'montytronic', 'warmian']
#     # embs = [helper[w].detach().numpy() for w in words]
#     # embs = np.stack(embs)
#     # indices = torch.LongTensor([helper[w] for w in words])
#     # embs = embeddings(indices)
#     i = 0
