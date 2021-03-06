import os
import json
import sys

import torch
import yaml
import torch.optim as optim
from flask import Flask, request
from transformers import BertConfig

import dqn
from bert.PrecomputedBert import PrecomputedBert
from dqn import DQN, LinearQN, MLP, BQN, BaseApproximator
from embeddings import EmbeddingsHelper

wsgi = Flask(__name__)

# Initialization stuff
with open("config.yml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

embeddings_cfg = cfg['embeddings']
device = cfg['device']
bert_cfg = cfg['bert']

glove_path = embeddings_cfg['glove_path']
voc_path = embeddings_cfg['voc_path']
freeze_embeddings = embeddings_cfg['freeze']

model_dir = cfg['model_dir']

# Set up the global variables
helper = None

network: BaseApproximator = None
trainer: optim.Optimizer = None
zero_init: bool = False
Approximator: type = None  # This is the default approximator class used
#########################


@wsgi.before_request
def configuration_hook():
    """ This is a hook that handles the configuration"""
    if 'zero_init' in request.args:
        global zero_init
        val = request.args.get('zero_init')
        if val.lower() == 'true':
            zero_init = True
        elif val.lower() == 'false':
            zero_init = False

    if 'approximator' in request.args:
        global Approximator, helper

        val = request.args.get('approximator')
        if val.lower() == 'linear':
            Approximator = LinearQN
            helper = EmbeddingsHelper(glove_path, voc_path, freeze_embeddings)
        elif val.lower() == 'dqn':
            Approximator = DQN
            helper = EmbeddingsHelper(glove_path, voc_path, freeze_embeddings)
        elif val.lower() == 'mlp':
            Approximator = MLP
            helper = EmbeddingsHelper(glove_path, voc_path, freeze_embeddings)
        elif val.lower() == 'bqn':
            Approximator = BQN
            helper = PrecomputedBert(bert_cfg['sentences_path'], bert_cfg['database_path'])

        else:
            raise NotImplementedError("Approximator %s not implemented" % val)


@wsgi.route('/')
def api_root():
    return 'Welcome to the magic of Deep Reinforcement Learning'


@wsgi.route('/select_action', methods=['GET', 'PUT'])
def select_action():
    if request.method == "PUT":
        data = json.loads(request.data)

        # If this is the first time this is called, lazily build the network
        # This is necessary to compute the number features dynamically
        global network
        if not network:
            new_state, _ = dqn.process_input_data(data[0])
            k = len(new_state[0]['features'])
            network = Approximator(k, helper, zero_init, device=device)
            if torch.cuda.is_available():
                network = network.cuda()

        # Don't need to hold to the gradient here
        with torch.no_grad():
            network.eval()
            pairs, values = network.select_action(data)
            ret = [{"index": v.argmax().item(), "A": list(p[0]), "B": list(p[1])} for p, v in zip(pairs, values)]

        return json.dumps(ret)
    else:
        return "Use the PUT method"


@wsgi.route('/backwards', methods=['GET', 'PUT'])
def backwards():
    if request.method == "PUT":
        data = json.loads(request.data)
        global network
        if not network:
            new_state, _ = dqn.process_input_data(data[0]['new_state'])
            k = len(new_state[0]['features'])
            network = Approximator(k, helper, zero_init_params=zero_init, device=device)
            if torch.cuda.is_available():
                network = network.cuda()

        global trainer
        if not trainer:
            trainer = optim.RMSprop(network.parameters())

        network.train()
        loss = network.backprop(data)

        # Optimize the model
        trainer.zero_grad()
        loss.backward()
        for param in network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        trainer.step()

        del loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # print("Death gradients: %f" % dqn.death_gradient(network.parameters()))

        return "Performed a back propagation step"

    else:
        return "Use the PUT method"


@wsgi.route('/distance', methods=['GET', 'PUT'])
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


@wsgi.route('/save', methods=['GET', 'POST'])
def save():
    if request.method == "POST":
        model_name = request.args.get("name")
        path = os.path.join(model_dir, model_name)

        global network
        if not network:
            return "No model instance created yet", 503

        is_debug = wsgi.config['DEBUG']

        # Don't save anything if this is a debugging session
        if not is_debug:
            state = network.state_dict()
            torch.save(state, path)

        return "Saved the model into %s" % model_name
    else:
        return "Use the POST method"


@wsgi.route('/load', methods=['GET', 'POST'])
def load():
    if request.method == "POST":
        global network, Approximator
        model_name = request.args.get("name")
        path = os.path.join(model_dir, model_name)

        state = torch.load(path, map_location=torch.device('cpu'))

        if Approximator == DQN:
            k = state['layers.0.weight'].shape[1] - helper.dimensions() * 2
        elif Approximator == BQN:
            k = state['layers.0.weight'].shape[1] - BertConfig.from_pretrained('bert-base-uncased').hidden_size * 2
        else:
            k = state['layers.0.weight'].shape[1]

        network = Approximator(k, helper, zero_init_params=zero_init, device=device)
        network.load_state_dict(state)
        if torch.cuda.is_available():
            network = network.cuda()

        return "Loaded the %s model from %s" % (str(Approximator), model_name)

    else:
        return "Use the POST method"


@wsgi.route('/reset', methods=['GET'])
def reset():
    global network, trainer, helper
    network = None
    trainer = None

    # helper = EmbeddingsHelper(glove_path, voc_path)

    return "Reset the parameters"


if __name__ == '__main__':
    from waitress import serve

    serve(wsgi, listen="*:5000", threads=20)
