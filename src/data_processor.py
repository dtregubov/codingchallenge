import random
from typing import Tuple

import torch
from torch import nn

from src.data_utils import create_indices, create_tensors, read_data, sentence_to_tensor
from src.model import BoW


class DataProcessor:
    iterations_to_train: int = 2
    cuda_available: bool = torch.cuda.is_available()

    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        # Select device and tensor
        self.device = 'cuda' if self.cuda_available else 'cpu'
        self.tensor_type = torch.cuda.LongTensor if self.cuda_available else torch.LongTensor
        # Initialise model
        self.model, self.word_to_index, self.tag_to_index = self._initialize_model()
        # Prepare model settings
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def _initialize_model(self) -> Tuple[BoW, dict, dict]:
        # Prepare data to use
        train_data = read_data(self.train_file)
        test_data = read_data(self.test_file)
        word_to_index, tag_to_index = create_indices(train_data)
        self.train_tensors = create_tensors(train_data, word_to_index, tag_to_index)
        self.test_tensors = create_tensors(test_data, word_to_index, tag_to_index)

        # Model parameters
        number_of_words = len(word_to_index)
        number_of_tags = len(tag_to_index)
        model = BoW(number_of_words, number_of_tags, self.tensor_type).to(self.device)

        return model, word_to_index, tag_to_index

    # Perform training and testing of BoW model.
    def train(self) -> None:
        # Make sure the model is on the correct device
        self.model.to(self.device)

        for iteration in range(self.iterations_to_train):
            # perform training of the model
            self.model.train()
            random.shuffle(self.train_tensors)
            total_loss = 0.0
            train_correct = 0

            for sentence, tag in self.train_tensors:
                sentence = torch.tensor(sentence).type(self.tensor_type).to(self.device)
                tag = torch.tensor([tag]).type(self.tensor_type).to(self.device)
                output = self.model(sentence)
                predicted = torch.argmax(output.data.detach()).item()
                loss = self.criterion(output, tag)
                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if predicted == tag.item():
                    train_correct += 1

            # perform testing of the model
            self.model.eval()
            test_correct = 0
            with torch.no_grad():
                for sentence, tag in self.test_tensors:
                    sentence = torch.tensor(sentence).type(self.tensor_type).to(self.device)
                    output = self.model(sentence)
                    predicted = torch.argmax(output.data).item()
                    if predicted == tag:
                        test_correct += 1

            # print model performance results
            log = f'ITER: {iteration + 1}, ' \
                  f'train loss/sent: {total_loss / len(self.train_tensors): .4f}, ' \
                  f'train accuracy: {train_correct / len(self.train_tensors): .4f}, ' \
                  f'test accuracy: {test_correct / len(self.test_tensors): .4f}.'

            print(log)

    # Perform inference for input sentence on the trained BoW model.
    def perform_inference(self, sentence: str) -> str:
        # Preprocess the input sentence to match the model's input format
        sentence_tensor = sentence_to_tensor(sentence, self.word_to_index, self.tensor_type)

        # Move the input tensor to the same device as the model
        sentence_tensor = sentence_tensor.to(self.device)

        # Make sure the model is in evaluation mode and on the correct device
        self.model.eval()
        self.model.to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(sentence_tensor)

        # Move the output tensor to CPU if it's on CUDA
        if self.device == 'cuda':
            output = output.cpu()

        # Convert the model's output to a predicted class/tag
        predicted_class = torch.argmax(output).item()

        # Reverse lookup to get the tag corresponding to the predicted class
        for tag, index in self.tag_to_index.items():
            if index == predicted_class:
                return tag

        # Return an error message if the tag is not found
        return 'Tag not found'
