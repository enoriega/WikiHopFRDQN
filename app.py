import json
import torch
from flask import Flask, url_for, request

from embeddings import EmbeddingsHelper

app = Flask(__name__)

glove_path = "/Users/enrique/github/WikiHopFR/glove/glove.6B.50d.txt"
voc_path = "/Users/enrique/github/WikiHopFR/w2vvoc.txt"

helper = EmbeddingsHelper(glove_path, voc_path)


@app.route('/')
def api_root():
    return 'Welcome to the magic of Deep Reinforcement Learning'


@app.route('/forward', methods=['GET', 'PUT'])
def forward():
    if request.method == "PUT":
        data = json.loads(request.data)
        return {"Exploration": 9.1, "Exploitation": 1.0}
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
