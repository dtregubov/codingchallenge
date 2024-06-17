import random

import torch


def train_bow(model, train_tensors, test_tensors, device, tensor_type, iterations: int):
    """
    Perform training and testing of BoW model.

    Args:
        model (torch.nn.Module): The trained BoW model.
        train_tensors (list): List tensors for train data.
        test_tensors (list): List tensors for test data.
        device (str): "cuda" or "cpu" based on availability.
        tensor_type (tensortype): type of torch.LongTensor
        iterations (int): Iterations of training

    Returns:
        None
    """
    # prepare model settings
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Make sure the model is on the correct device
    model.to(device)

    for iteration in range(iterations):
        # perform training of the model
        model.train()
        random.shuffle(train_tensors)
        total_loss = 0.0
        train_correct = 0

        for sentence, tag in train_tensors:
            sentence = torch.tensor(sentence).type(tensor_type).to(device)
            tag = torch.tensor([tag]).type(tensor_type).to(device)
            output = model(sentence)
            predicted = torch.argmax(output.data.detach()).item()
            loss = criterion(output, tag)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if predicted == tag.item():
                train_correct += 1

        # perform testing of the model
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for sentence, tag in test_tensors:
                sentence = torch.tensor(sentence).type(tensor_type).to(device)
                output = model(sentence)
                predicted = torch.argmax(output.data).item()
                if predicted == tag:
                    test_correct += 1

        # print model performance results
        log = f'ITER: {iteration + 1}, ' \
            f'train loss/sent: {total_loss / len(train_tensors): .4f}, ' \
            f'train accuracy: {train_correct / len(train_tensors): .4f}, ' \
            f'test accuracy: {test_correct / len(test_tensors): .4f}.'

        print(log)
