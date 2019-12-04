import csv
import torch
import pickle
import itertools as it
from operator import attrgetter
from tqdm import tqdm
from collections import namedtuple
from transformers import *


PATH = "sentences_w_relations.tsv"

# Structs to store the data
Entry = namedtuple("Entry", "doc sentence text")
ProcessedDatum = namedtuple("ProcessedDatum", "doc sen pooler states")


class Serializer:
    def __init__(self):
        import os.path
        self.fp = open('bert_cache.pickle', 'ab')
        if os.path.exists('record.txt'):
            with open('record.txt', 'r') as f:
                self.record = {l[:-1] for l in f}
        else:
            self.record = set()

    def serialize(self, d):
        key = (d.doc, d.sen)
        pickle.dump(d, self.fp)
        self.record.add(key)

    def close(self):
        self.fp.close()
        with open('record.txt', 'w') as f:
            for i in self.record:
                f.write('%s\n' % str(i))

    def __contains__(self, item):
        return str(item) in self.record


if __name__ == "__main__":
    # Parse the TSV file with the data
    with open(PATH) as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [Entry(r[0], int(r[1]), r[-1]) for r in reader]

    ser = Serializer()

    # Group it by document
    groups = [(g, list(e)) for g, e in it.groupby(rows, key=attrgetter('doc'))]

    # Load bert
    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Set the model to eval mode
    bert.eval()
    # Don't keep track of the gradients
    with torch.no_grad():
        # Loop over the document contents
        for doc, entries in tqdm(groups):
            # Fetch the actual sentences that are going to be passed to bert
            sentences = [(ix, e.text) for ix, e in enumerate(entries)]
            # Just select the first 512 elements, the max sequence allowed by the library
            input_sequences = [(ix, torch.tensor([tokenizer.encode(s)[:512]])) for ix, s in sentences if (doc, ix) not in ser]
            # Pass them through bert and collect all the ouputs
            raw_outputs = [(ix, bert(input)) for ix, input in input_sequences]
            # Collect the data and store it
            for sen_ix, (hidden_states, pooler) in raw_outputs:
                states = hidden_states.squeeze().numpy()
                pooler = pooler.squeeze().numpy()
                datum = ProcessedDatum(doc, sen_ix, pooler, states)
                ser.serialize(datum)

        ser.close()




