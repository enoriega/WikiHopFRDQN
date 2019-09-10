import os
import json
import torch
import yaml
import torch.optim as optim
from flask import Flask, request

import dqn
from dqn import DQN
from embeddings import EmbeddingsHelper

app = Flask(__name__)

with open('config.yml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

embeddings_cfg = cfg['embeddings']

glove_path = embeddings_cfg['glove_path']
voc_path = embeddings_cfg['voc_path']

model_dir = cfg['model_dir']

helper = EmbeddingsHelper(glove_path, voc_path)

network = None
trainer = None


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
            if torch.cuda.is_available():
                network = network.cuda()

        # Don't need to hold to the gradient here
        with torch.no_grad():
            raw_values = network(data)
            values = DQN.raw2json(raw_values)

        return json.dumps(values)
    else:
        return "Use the PUT method"


@app.route('/select_action', methods=['GET', 'PUT'])
def select_action():
    if request.method == "PUT":
        data = json.loads(request.data)

        # If this is the first time this is called, lazily build the network
        # This is necessary to compute the number features dynamically
        global network
        if not network:
            k = len(data[0]['features'])
            network = DQN(k, helper)
            if torch.cuda.is_available():
                network = network.cuda()

        # Don't need to hold to the gradient here
        with torch.no_grad():
            pairs, values = network.select_action(data)
            ret = [{"index": v.argmax().item(), "A": list(p[0]), "B": list(p[1])} for p, v in zip(pairs, values)]

        return json.dumps(ret)
    else:
        return "Use the PUT method"


@app.route('/backwards', methods=['GET', 'PUT'])
def backwards():
    if request.method == "PUT":
        data = json.loads(request.data)
        global network
        if not network:
            new_state, _ = dqn.rename_me(data[0]['new_state'])
            k = len(new_state[0]['features'])
            network = DQN(k, helper)
            if torch.cuda.is_available():
                network = network.cuda()

        global trainer
        if not trainer:
            trainer = optim.RMSprop(network.parameters())

        loss = network.backprop(data)

        # Optimize the model
        trainer.zero_grad()
        loss.backward()
        for param in network.parameters():
            param.grad.data.clamp_(-1, 1)
        trainer.step()

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


@app.route('/save', methods=['GET', 'POST'])
def save():
    if request.method == "POST":
        model_name = request.args.get("name")
        path = os.path.join(model_dir, model_name)

        global network
        if not network:
            return "No model instance created yet", 503

        state = network.state_dict()
        torch.save(state, path)

        return "Saved the model into %s" % model_name
    else:
        return "Use the POST method"


@app.route('/load', methods=['GET', 'POST'])
def load():
    if request.method == "POST":
        model_name = request.args.get("name")
        # k = int(request.args.get("k"))
        path = os.path.join(model_dir, model_name)

        state = torch.load(path)
        k = state['layers.0.weight'].shape[1] - helper.dimensions()*2
        global network
        network = DQN(k, helper)
        network.load_state_dict(state)
        if torch.cuda.is_available():
            network = network.cuda()

        return "Loaded the model from %s" % model_name
    else:
        return "Use the POST method"


@app.route('/reset', methods=['GET'])
def reset():
    global network, trainer, helper
    network = None
    trainer = None

    helper = EmbeddingsHelper(glove_path, voc_path)

    return "Reset the parameters"


if __name__ == '__main__':
    app.run()
