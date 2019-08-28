import numpy as np
import torch
from torch import nn
from embeddings import EmbeddingsHelper


class DQN(nn.Module):

    def __init__(self, embeddings_helper):
        super().__init__()

        # Store the helper to use it on the forward method
        self.e_helper = embeddings_helper
        # This is necessary for the instance to recognize the embeddings as parameters of the network
        self.pretrained_embeddings = embeddings_helper.pretrained_embeddings
        self.fresh_embeddings = embeddings_helper.fresh_embeddings

    def forward(self, data):

        # Parse the input data into tensor form
        batch = self.tensor_form(data)

        # TODO Feed it forward into the layers of the network

        return None

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
            ea = helper.aggregated_embedding(entity_a)
            eb = helper.aggregated_embedding(entity_b)

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
        # TODO Implement this
        return [{"Exploration": 9.1, "Exploitation": 1.0} for v in raw_vals]


if __name__ == "__main__":
    # TODO: Parameterize paths
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

    network = DQN(helper)
    network(test_data)

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
