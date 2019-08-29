import json
import torch
import yaml
from flask import Flask, request

from dqn import DQN
from embeddings import EmbeddingsHelper

app = Flask(__name__)

with open('config.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

embeddings_cfg = cfg['embeddings']

glove_path = embeddings_cfg['glove_path']
voc_path = embeddings_cfg['voc_path']

helper = EmbeddingsHelper(glove_path, voc_path)

network = None


@app.route('/')
def api_root():
    return 'Welcome to the magic of Deep Reinforcement Learning'


@app.route('/forward', methods=['GET', 'PUT'])
def forward():
    if request.method == "PUT":
        data = json.loads(request.data)

        # If this is the first time this is called, lazily build the network
        # This is necessary to compute the number features dynamically
        global network
        if not network:
            k = len(data[0]['features'])
            network = DQN(k, helper)

        # Don't need to hold to the gradient here
        with torch.no_grad():
            raw_values = network(data)
            values = DQN.raw2json(raw_values)

        return json.dumps(values)
    else:
        return "Use the PUT method"


@app.route('/backwards', methods=['GET', 'PUT'])
def backwards():
    if request.method == "PUT":
        data = json.loads(request.data)
        return "Performed a back propagation step"
    else:
        return "Use the PUT method"


@app.route('/distance', methods=['GET', 'PUT'])
def distance():
    if request.method == "PUT":
        # Don't need to hold to the gradient here
        with torch.no_grad():
            data = json.loads(request.data)
            pairs = [(d["A"], d["B"]) for d in data]
            dists = helper.distance(pairs)

        return json.dumps(dists)

    else:
        return "Use the PUT method"


if __name__ == '__main__':
    app.run()
