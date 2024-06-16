from typing import Union

import torch
from torch import nn


# create a simple neural network with embedding layer, bias, and xavier initialization
class BoW(nn.Module):
    def __init__(self, nwords, ntags, tensor_type):
        super().__init__()
        self.embedding = nn.Embedding(nwords, ntags)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.bias = torch.zeros(ntags, requires_grad=True).type(tensor_type)

    def forward(self, x):
        emb = self.embedding(x)
        out = torch.sum(emb, dim=0) + self.bias
        out = out.view(1, -1)
        return out


# convert sentence to tensor using word indices counted fromm train dataset
def sentence_to_tensor(
    sentence: str,
    word_to_index: dict,
    tensor_type: Union[torch.LongTensor, torch.cuda.LongTensor],
) -> Union[torch.LongTensor, torch.cuda.LongTensor]:
    return tensor_type([word_to_index.get(word, word_to_index['<unk>']) for word in sentence.split(' ')])
