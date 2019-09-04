import torch
import numpy as np
import itertools as it
from torch import nn


class EmbeddingsHelper:

    def __init__(self, data_path, voc_path):
        self.matrix, self.existing_terms, self.missing_terms = self.load_embeddings_from_file(
            data_path, voc_path)

        self.device = torch.device("cpu")

        self.pretrained_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(self.matrix), freeze=False)
        self.fresh_embeddings = nn.Embedding(len(self.missing_terms), self.dimensions())

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.pretrained_embeddings = self.pretrained_embeddings.cuda()
            self.fresh_embeddings = self.fresh_embeddings.cuda()


    @staticmethod
    def load_embeddings_from_file(data_path, voc_path):
        # Load the vocabulary terms
        with open(voc_path) as f:
            sanitized_terms = {l[:-1] for l in f if l[:-1]}

        # Load the embeddings matrix
        with open(data_path) as f:
            lines = [l for ix, l in enumerate(f) if ix > 0 and l.split(maxsplit=1)[0] in sanitized_terms]

        it1, it2 = it.tee((t[0], np.asarray([float(d) for d in t[1:]])) for t in (l.split() for l in lines))

        existing_terms = {i[0]: ix for ix, i in enumerate(it1)}
        matrix = np.stack([i[1] for i in it2]).astype(float)

        missing_terms = {t: ix for ix, t in enumerate((sanitized_terms - existing_terms.keys()) | {"xnumx", "OOV"})}

        return matrix, existing_terms, missing_terms

    @staticmethod
    def is_number(word):
        try:
            float(word)
            return True
        except ValueError as e:
            return False

    @staticmethod
    def sanitize_word(word, keep_numbers=True):
        w = word.lower().strip()
        if w in {"-lrb-", "-rrb-", "-lsb-", "-rsb-"}:
            return ""
        elif "http" in w or "https" in w or "com" in w or "org" in w:
            return ""
        elif EmbeddingsHelper.is_number(w):
            if keep_numbers:
                return "xnumx"
            else:
                return ""
        else:
            return ''.join(c for c in w if ("a" <= c <= "z") or c == "_")

    def index(self, word):
        sanitized = EmbeddingsHelper.sanitize_word(word)
        if sanitized in self.existing_terms:
            return True, self.existing_terms[sanitized]
        else:
            if sanitized in self.missing_terms:
                return False, self.missing_terms[sanitized]
            else:
                return False, self.missing_terms["OOV"]

    def __getitem__(self, word):
        is_pretrained, ix = self.index(word)
        if is_pretrained:
            return self.pretrained_embeddings(torch.LongTensor([ix]).to(self.device)).float()
        else:
            return self.fresh_embeddings(torch.LongTensor([ix]).to(self.device)).float()

    def __len__(self):
        return len(self.existing_terms)

    def dimensions(self):
        return self.matrix.shape[1]

    def aggregated_embedding(self, tokens):
        embs_a = [self[a] for a in tokens]
        return EmbeddingsHelper.aggregate_embeddings(embs_a)

    @staticmethod
    def aggregate_embeddings(embs):
        m = torch.stack([e.squeeze() for e in embs])
        return torch.mean(m, dim=0)

    def distance(self, entities):

        ret = list()

        for entity_a, entity_b in entities:
            ea = self.aggregated_embedding(entity_a)
            eb = self.aggregated_embedding(entity_b)

            ret.append(torch.dist(ea, eb).detach().item())

        return ret
