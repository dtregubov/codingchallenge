import pytest
from src.data_processor import DataProcessor
from src.model import BoW


@pytest.fixture(scope='module')
def data_processor():
    data_processor = DataProcessor('tests/test_data/train.txt', 'tests/test_data/test.txt')
    data_processor.train(iterations=1)
    return data_processor


def test_model_initialization(data_processor):
    assert data_processor.model is not None
    assert data_processor.word_to_index is not None
    assert data_processor.tag_to_index is not None


def test_model_evaluation_mode(data_processor):
    assert data_processor.model.training is False


def test_model_type_bow(data_processor):
    assert isinstance(data_processor.model, BoW)


def test_prediction(data_processor):
    sentence = 'I love programming'
    predicted_tag = data_processor.perform_inference(sentence)
    assert isinstance(predicted_tag, str)
    assert predicted_tag in data_processor.tag_to_index


def test_existed_sentence_prediction(data_processor):
    existed_sentence = 'Effective but too-tepid biopic'
    existed_sentence_tag = '2'
    predicted_tag = data_processor.perform_inference(existed_sentence)
    assert existed_sentence_tag == predicted_tag
