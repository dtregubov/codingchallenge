from unittest.mock import mock_open, patch
from src.data_utils import read_data, create_indices, create_tensors


def test_read_data():
    expected = [['label1', 'word1 word2'], ['label2', 'word3 word4']]

    with patch('builtins.open', mock_open(read_data='label1 ||| word1 word2\nlabel2 ||| word3 word4')) as m:
        result = read_data('dummy.txt')

    assert result == expected
    m.assert_called_once_with('dummy.txt', 'r', encoding='UTF8')


def test_create_indices():
    data = [['label1', 'word1 word2'], ['label2', 'word3 word4']]
    expected_word_to_index = {'<unk>': 0, 'word1': 1, 'word2': 2, 'word3': 3, 'word4': 4}
    expected_tag_to_index = {'label1': 0, 'label2': 1}

    word_to_index, tag_to_index = create_indices(data)

    assert word_to_index == expected_word_to_index
    assert tag_to_index == expected_tag_to_index


def test_create_tensors():
    data = [['label1', 'word1 word2'], ['label2', 'word3 word4']]
    word_to_index = {'<unk>': 0, 'word1': 1, 'word2': 2, 'word3': 3, 'word4': 4}
    tag_to_index = {'label1': 0, 'label2': 1}
    expected_tensors = [([1, 2], 0), ([3, 4], 1)]

    tensors = list(create_tensors(data, word_to_index, tag_to_index))

    assert tensors == expected_tensors
