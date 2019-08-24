import numpy as np
import itertools as it


class EmbeddingsHelper(object):

    # TODO: Parameterize paths
    def __init__(self):
        self.vocabulary_path = "/Users/enrique/github/WikiHopFR/w2vvoc.txt"
        self.matrix, self.existing_terms, self.missing_terms = self.load_embeddings_from_file(
            "/Users/enrique/github/WikiHopFR/glove/glove.6B.50d.txt")

    def load_embeddings_from_file(self, path):
        # Load the vocabulary terms
        with open(self.vocabulary_path) as f:
            sanitized_terms = {l[:-1] for l in f if l[:-1]}

        # Load the embeddings matrix
        with open(path) as f:
            lines = [l for ix, l in enumerate(f) if ix > 0 and l.split(maxsplit=1)[0] in sanitized_terms]

        it1, it2 = it.tee((t[0], np.asarray([float(d) for d in t[1:]])) for t in (l.split() for l in lines))

        existing_terms = {i[0]: ix for ix, i in enumerate(it1)}
        matrix = np.stack([i[1] for i in it2])

        missing_terms = {t: ix for ix, t in enumerate((sanitized_terms - existing_terms.keys()) | {"xnumx", "OOV"})}

        return matrix, existing_terms, missing_terms
