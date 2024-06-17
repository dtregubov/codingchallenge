# This is a CodeChallenge's Python project written from assigment1.ipynb notebook

from flask import Flask, jsonify, request

from src.data_processor import DataProcessor
from src.data_utils import download_data

app = Flask(__name__)


# Download data to use
download_data()

# Initialize the data processor and train model
data_processor = DataProcessor('data/classes/train.txt', 'data/classes/test.txt')
data_processor.train()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    predicted_tag = data_processor.perform_inference(sentence)
    return jsonify({'sentence': sentence, 'predicted_tag': predicted_tag})


if __name__ == '__main__':
    app.run()
