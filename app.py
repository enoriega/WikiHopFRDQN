import json
import torch
import yaml
from flask import Flask, request

from dqn import DQN
from embeddings import EmbeddingsHelper

app = Flask(__name__)

with open('config.yml', Loader=yaml.FullLoader) as f:
    cfg = yaml.load(f)

embeddings_cfg = cfg['embeddings']

glove_path = embeddings_cfg['glove_path']
voc_path = embeddings_cfg['voc_path']

helper = EmbeddingsHelper(glove_path, voc_path)
network = DQN(helper)


@app.route('/')
def api_root():
    return 'Welcome to the magic of Deep Reinforcement Learning'


@app.route('/forward', methods=['GET', 'PUT'])
def forward():
    if request.method == "PUT":
        data = json.loads(request.data)
        raw_values = network(data)
        values = DQN.raw2json(raw_values)
        return values
        # return {"Exploration": 9.1, "Exploitation": 1.0}
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
        with torch.no_grad():
            data = json.loads(request.data)
            pairs = [(d["A"], d["B"]) for d in data]
            dists = helper.distance(pairs)

        return json.dumps(dists)

    else:
        return "Use the PUT method"


if __name__ == '__main__':
    app.run()
