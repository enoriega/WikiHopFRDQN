import json
from flask import Flask, url_for, request

app = Flask(__name__)


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
        data = json.loads(request.data)
        return json.dumps([1.0, 0.0])
    else:
        return "Use the PUT method"


if __name__ == '__main__':
    app.run()
