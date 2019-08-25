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

    words = ['the', 'rain', 'mexico', 'montytronic', 'warmian']
    embs = np.stack([helper[w].detach().numpy() for w in words])
    # indices = torch.LongTensor([helper[w] for w in words])
    # embs = embeddings(indices)
    i = 0
