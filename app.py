# This is a CodeChallenge's Python project written from assigment1.ipynb notebook

import torch
from flask import Flask, jsonify, request

from src.data_utils import create_indices, create_tensors, download_data, read_data
from src.model import BoW
from src.train import train_bow
from src.inference import perform_inference

app = Flask(__name__)


iterations_to_train = 10

# choose device and tensor
if torch.cuda.is_available():
    device = 'cuda'
    tensor_type = torch.cuda.LongTensor
else:
    device = 'cpu'
    tensor_type = torch.LongTensor

# Prepare data to use
download_data()
train_data = read_data('data/classes/train.txt')
test_data = read_data('data/classes/test.txt')
word_to_index, tag_to_index = create_indices(train_data)
train_tensors = create_tensors(train_data, word_to_index, tag_to_index)
test_tensors = create_tensors(test_data, word_to_index, tag_to_index)

# Model parameters
number_of_words = len(word_to_index)
number_of_tags = len(tag_to_index)
model = BoW(number_of_words, number_of_tags, tensor_type).to(device)

# Training and testing of the model
train_bow(model, train_tensors, test_tensors, device, tensor_type, iterations_to_train)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    if len(sentence) > 1000:
        return jsonify({'error': 'Sentence is too big'}), 400

    predicted_tag = perform_inference(model, sentence, word_to_index, tag_to_index, device, tensor_type)
    return jsonify({'sentence': sentence, 'predicted_tag': predicted_tag})


if __name__ == '__main__':
    app.run()
