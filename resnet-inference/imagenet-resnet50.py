import os

import numpy as np
import torch
from PIL import Image
from flask import Flask, request
from torchvision import transforms
from werkzeug.utils import secure_filename

device = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_FOLDER = '/tmp'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def __load_labels(label_file):
    label = []
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            label.append(line.strip())
    return label


def preprocess(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def postprocess(output, label_file="./labels.txt"):
    output = torch.nn.functional.softmax(output[0], dim=0).tolist()
    labels = __load_labels(label_file)
    top_k = np.array(output).argsort()[-1:][::-1]
    result = {}
    for i in top_k:
        result[labels[i]] = output[i]
    return result


def running(input):
    net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    net = net.to(device)
    net.eval()
    output = net(input)
    return output


@app.route('/', methods=['POST'])
def serving():
    file = request.files['file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filename)
    input = preprocess(filename).to(device)
    output = running(input)
    result = postprocess(output)
    return result


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")

    # curl -XPOST http://127.0.0.1:5000 -F "file=@./dog.jpg"
