import numpy as np
import torch
from torch import nn
from embeddings import EmbeddingsHelper


class DQN(nn.Module):

    def __init__(self, num_feats, embeddings_helper):
        super().__init__()

        # Store the helper to use it on the forward method
        self.e_helper = embeddings_helper
        # This is necessary for the instance to recognize the embeddings as parameters of the network
        self.pretrained_embeddings = embeddings_helper.pretrained_embeddings
        self.fresh_embeddings = embeddings_helper.fresh_embeddings

        k = num_feats + embeddings_helper.dimensions()*2

        # Layers of the network
        self.layers = nn.Sequential(
            nn.Linear(k, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, data):

        # Parse the input data into tensor form
        batch = self.tensor_form(data)

        values = self.layers(batch)

        return values

    def tensor_form(self, data):
        # Convert the raw data to tensor form
        batch = list()

        # Create an input vector for each of the elements in data
        for input in data:
            # Get the raw input
            features = input['features']
            entity_a = input['A']
            entity_b = input['B']

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


if __name__ == "__main__":
    glove_path = "/Users/enrique/github/WikiHopFR/glove/glove.6B.50d.txt"
    voc_path = "/Users/enrique/github/WikiHopFR/w2vvoc.txt"

    helper = EmbeddingsHelper(glove_path, voc_path)

    test_data = [{
        'features': {
            "iteration": float(5),
            "alpha": float(10.3)
        },
        'A':['enrique', 'noriega'],
        'B':['Cecilia', 'Montano']
    }, {
        'features': {
            "iteration": float(34),
            "alpha": float(15.0)
        },
        'A':['Diego', 'Andres'],
        'B':['Gina', 'Chistosina']
    }]

    network = DQN(2, helper)

    with torch.no_grad():
        vals = network(test_data)

    x = DQN.raw2json(vals)

    # embeddings = nn.Embedding.from_pretrained(torch.from_numpy(helper.matrix))

    # entity_a = ["hermosillo", "sonora"]
    # entity_b = ["tucson", "arizona"]
    # dist = helper.distance(entity_a, entity_b)
    # words = ['the', 'rain', 'mexico', 'montytronic', 'warmian']
    # embs = [helper[w].detach().numpy() for w in words]
    # embs = np.stack(embs)
    # indices = torch.LongTensor([helper[w] for w in words])
    # embs = embeddings(indices)
    i = 0
