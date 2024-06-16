# This is a CodeChallenge's Python project written from assigment1.ipynb notebook

import torch

from src.data_utils import create_indices, create_tensors, delete_data, download_data, read_data
from src.model import BoW
from src.train import perform_inference, train_bow


def main() -> None:
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

    # Run the trained model with
    sentence = input('Enter a sentence to predict tag: ')
    predicted_tag = perform_inference(model, sentence, word_to_index, tag_to_index, device, tensor_type)
    print(f'Predicted Tag: {predicted_tag}')

    # Delete data after using
    delete_data()


if __name__ == "__main__":
    main()
