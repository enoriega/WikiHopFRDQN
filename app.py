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
from embeddings import EmbeddingsHelper, load_embeddings_matrix
from torch.nn import EmbeddingBag, Embedding


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
use_embeddings = embeddings_cfg['activate']

model_dir = cfg['model_dir']

# Set up the global variables
helper = None

network: BaseApproximator = None
optimizer: optim.Optimizer = None
lr_scheduler = None
zero_init: bool = False
Approximator: type = None  # This is the default approximator class used
prev_grad = dict()
last_loss = None


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
            # helper = EmbeddingsHelper(glove_path, voc_path, freeze_embeddings)
            matrix = load_embeddings_matrix(glove_path)
            helper = torch.nn.EmbeddingBag.from_pretrained(matrix)
        elif val.lower() == 'dqn':
            Approximator = DQN
            helper = EmbeddingsHelper(glove_path, voc_path, freeze_embeddings)
        elif val.lower() == 'mlp':
            Approximator = MLP
            # helper = EmbeddingsHelper(glove_path, voc_path, freeze_embeddings)
            matrix = load_embeddings_matrix(glove_path)
            helper = torch.nn.EmbeddingBag.from_pretrained(matrix)
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
            k = len(list(filter(lambda key: "Lemma_" not in key, data.keys())))
            if use_embeddings:
                k += 200
            network = Approximator(k, helper, zero_init, device=device, use_embeddings=use_embeddings)
            if torch.cuda.is_available():
                network = network.cuda()

        # Don't need to hold to the gradient here
        with torch.no_grad():
            network.eval()
            tensor = network.dictionary_to_tensor(data, use_embeddings)
            values = network(tensor)
            ret = values.tolist()

        return json.dumps(ret)
    else:
        return "Use the PUT method"


@wsgi.route('/backwards', methods=['GET', 'PUT'])
def backwards():
    if request.method == "PUT":
        data = json.loads(request.data)
        global network
        if not network:
            k = len(data)
            if use_embeddings:
                k += 200
            network = Approximator(k, helper, zero_init_params=zero_init, device=device, use_embeddings=use_embeddings)
            if torch.cuda.is_available():
                network = network.cuda()

        learning_rate = data['rate']

        global optimizer, lr_scheduler
        if not optimizer:
            optimizer = optim.RMSprop(network.parameters(), lr= learning_rate)
            # lr_scheduler = optim.lr_scheduler.StepLR(trainer, 1, gamma=0.99)

        network.train()

        loss = network.backprop(data)

        # Assign learning rate to parameters
        for group in optimizer.param_groups:
            group['lr'] = learning_rate

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        current_grad = dict()
        for ix, param in enumerate(network.parameters()):
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                current_grad[ix] = param.grad.data
        optimizer.step()

        # Record the change in the gradient
        change = 0
        global prev_grad
        for key in current_grad:
            if key not in prev_grad:
                change += current_grad[key].sum().item()
            else:
                change += (prev_grad[key] - current_grad[key]).sum().item()

        prev_grad = current_grad

        global last_loss
        last_loss = loss.item()

        del loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # lr_scheduler.step()
        # print("Death gradients: %f" % dqn.death_gradient(network.parameters()))

        return str(change)

    else:
        return "Use the PUT method"


@wsgi.route('/last_loss', methods=['GET'])
def get_loss():
    try:
        global last_loss
        if last_loss:
            return str(last_loss)
        else:
            "Woops"
    except:
        return str(0.0)


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
            del state['embeddings.weight']
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

        network = Approximator(k, helper, zero_init_params=zero_init, device=device, use_embeddings=use_embeddings)
        if type(helper) in {EmbeddingBag, Embedding}:
            state['embeddings.weight'] = helper.weight
        network.load_state_dict(state)
        if torch.cuda.is_available():
            network = network.cuda()

        return "Loaded the %s model from %s" % (str(Approximator), model_name)

    else:
        return "Use the POST method"


@wsgi.route('/reset', methods=['GET'])
def reset():
    global network, optimizer, helper, prev_grad, Approximator, lr_scheduler
    network = None
    optimizer = None
    lr_scheduler = None
    prev_grad = dict()
    # Approximator = None

    # helper = EmbeddingsHelper(glove_path, voc_path)

    return "Reset the parameters"


if __name__ == '__main__':
    from waitress import serve

    serve(wsgi, listen="*:5000", threads=20)
