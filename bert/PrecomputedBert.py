import torch
import numpy as np
import itertools as it

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from transformers import BertTokenizer

from bert.bert_orm import Sentence
from bert.berter import parse_sentences


class PrecomputedBert:

    def __init__(self, sentences_path='sentences_w_relations_new.tsv',
                 database_path='sqlite:////Volumes/My Passport/test2.sqlite'):
        original_sentences = parse_sentences(sentences_path)
        self.os = {(v.doc, v.sentence): v.text.split() for v in
                   it.chain.from_iterable(x[1] for x in original_sentences)}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.engine = create_engine(database_path)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def entity_origin_to_embeddings(self, eo) -> torch.Tensor:
        # Parse the entity origin to fetch the sentence with the mention
        doc, ix = eo['hash'], eo['sen']
        key = (doc, ix)
        start, end = eo['interval']
        tokens = self.os[key]

        assert end < len(tokens)
        # Map the original interval to the torch tokenized interval
        offset = 0
        all_units = list()
        for token_ix, token in enumerate(tokens):
            units = self.tokenizer.tokenize(token)
            all_units.extend(units)
            if token_ix == start:
                n_start = offset

            if token_ix == end:
                n_end = offset
                break

            offset += len(units)

        # n_tokens = all_units[n_start:n_end]
        # Fetch the embeddings from SQLAlchemy
        embeddings = self.fetch_embeddings(doc, ix, n_start, n_end)

        return embeddings

    def fetch_embeddings(self, doc: str, sen: int, start: int, end: int):
        data = self.session.query(Sentence). \
            filter(Sentence.doc == doc). \
            filter(Sentence.index == sen).first()

        hidden_states = data.states[start:end]

        ret = [hs.vector.data for hs in hidden_states]

        ret = torch.from_numpy(np.stack(ret))

        return ret

    def close(self):
        self.session.close()
