import pickle

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, PickleType, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from tqdm import tqdm

from bert.berter import ProcessedDatum

Base = declarative_base()


class Vector(Base):
    __tablename__ = 'vector'
    id = Column(Integer, primary_key=True)
    hidden_state_id = Column(Integer, ForeignKey("hiddenstate.id"), index=True)
    data = Column(PickleType)

    def __init__(self, data):
        self.data = data


class HiddenState(Base):
    __tablename__ = 'hiddenstate'
    id = Column(Integer, primary_key=True)
    index = Column(Integer)
    sentence_id = Column(Integer, ForeignKey("sentence.id"), index=True)
    vector = relationship("Vector", uselist=False)

    def __init__(self, index, state):
        self.index = index
        self.vector = state


class Sentence(Base):
    __tablename__ = 'sentence'
    id = Column(Integer, primary_key=True)
    doc = Column(String, index=True)
    index = Column(Integer, index=True)
    pooler_id = Column(Integer, ForeignKey('vector.id'))
    pooler = relationship("Vector")
    states = relationship("HiddenState")

    def __init__(self, doc, index):
        self.doc = doc
        self.index = index


Index('fr_index', Sentence.doc, Sentence.index)


def add_elem(d: ProcessedDatum, s):
    sen = Sentence(d.doc, d.sen)
    pooler = Vector(d.pooler)

    sen.pooler = pooler

    for ix, state in enumerate(d.states):
        v = Vector(state)
        o = HiddenState(ix, v)
        sen.states.append(o)

    session.add(sen)


def populate_database(e, s):
    Base.metadata.create_all(e)
    pbar = tqdm()

    with open('bert_cache.pickle', 'rb') as f:
        try:
            while True:
                d = pickle.load(f)
                add_elem(d, s)
                s.commit()
                pbar.update(1)
        except EOFError:
            pass
    pbar.close()



if __name__ == '__main__':
    engine = create_engine('sqlite:///test2.sqlite')
    Session = sessionmaker(bind=engine)
    session = Session()
    populate_database(engine, session)
    session.close()
