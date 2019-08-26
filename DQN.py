import numpy as np
import torch
from torch import nn
from embeddings import EmbeddingsHelper

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
