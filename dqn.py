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

    def forward(self, input):
        return None


if __name__ == "__main__":
    # TODO: Parameterize paths
    glove_path = "/Users/enrique/github/WikiHopFR/glove/glove.6B.50d.txt"
    voc_path = "/Users/enrique/github/WikiHopFR/w2vvoc.txt"

    helper = EmbeddingsHelper(glove_path, voc_path)

    # embeddings = nn.Embedding.from_pretrained(torch.from_numpy(helper.matrix))

    entity_a = ["hermosillo", "sonora"]
    entity_b = ["tucson", "arizona"]
    dist = helper.distance(entity_a, entity_b)
    # words = ['the', 'rain', 'mexico', 'montytronic', 'warmian']
    # embs = [helper[w].detach().numpy() for w in words]
    # embs = np.stack(embs)
    # indices = torch.LongTensor([helper[w] for w in words])
    # embs = embeddings(indices)
    i = 0
