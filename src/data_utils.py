import os
import shutil


# Download data for training model
def download_data() -> None:
    os.system('wget -P data/classes https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt')
    os.system('wget -P data/classes https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt')
    os.system('wget -P data/classes https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt')


def delete_data() -> None:
    shutil.rmtree('data')


# Read data from file and collect it to list as list of tag and sentence
def read_data(filename: str) -> list:
    data = []
    with open(filename, 'r', encoding='UTF8') as file:
        for line in file:
            line = line.lower().strip().split(' ||| ')
            data.append(line)
    return data


# create words and tags indices
def create_indices(data: list, check_unk=False) -> tuple:
    word_to_index = {'<unk>': 0}
    tag_to_index = {}

    for line in data:
        for word in line[1].split(' '):
            if not check_unk:
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index)
            else:
                if word not in word_to_index:
                    word_to_index[word] = word_to_index['<unk>']

        if line[0] not in tag_to_index:
            tag_to_index[line[0]] = len(tag_to_index)

    return word_to_index, tag_to_index


# create tensors using indices
def create_tensors(data: list, word_to_index: dict, tag_to_index: dict) -> list:
    return [([word_to_index.get(word, word_to_index['<unk>']) for word in line[1].split(' ')], tag_to_index[line[0]])
            for line in data]
