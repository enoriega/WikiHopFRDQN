import torch
from transformers import *


def get_bert_embeddings():
    bert = BertModel.from_pretrained('bert-base-uncased')
    return bert.embeddings.word_embeddings


# Inspired from http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch/
def pca(x, components):
    with torch.no_grad():
        x_mean = torch.mean(x, 0)
        x = x - x_mean.expand_as(x)
        U, S, V = torch.svd(torch.t(x))
        return torch.mm(x, U[:, :components])





