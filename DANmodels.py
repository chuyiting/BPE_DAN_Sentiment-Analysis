import torch
from torch import nn
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples, SentimentExample
from utils import Indexer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List
from tokenizer import BasicTokenizer, Tokenizer


class SentimentDatasetDAN(Dataset):
    def __init__(self, examples: List[SentimentExample], indexer: Indexer=None, tokenizer: Tokenizer=None):

        if indexer is not None:
            # sentences is indexed List[List[index: int]] index 1 UNK to training data
            self.sentences = [[1 if indexer.index_of(word) == -1 else indexer.index_of(word) for word in ex.words] for ex in examples]
            self.sentences = [torch.tensor(sentence, dtype=int) for sentence in self.sentences]
        elif tokenizer is not None:
            self.sentences = [" ".join(ex.words) for ex in examples]
            print("doing encoding...")
            self.sentences = tokenizer.encode(self.sentences)
            print("finish encoding")
            self.sentences = [torch.tensor(sentence) for sentence in self.sentences]

        self.sentences = pad_sequence(self.sentences, batch_first=True, padding_value=0)
        self.labels = [ex.label for ex in examples]

    def __getitem__(self, index) :
        if self.sentences[index].dtype not in {torch.int32, torch.int64, torch.uint8, torch.int16, torch.int8}:
            print("some sentence is weird...")
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
    

class DANModel(nn.Module):
    def __init__(self, hidden_size=256, embedding_layer=None):
        super().__init__()

        self.embedding = embedding_layer
        self.embedding_dim = embedding_layer.embedding_dim
        print(f'word embedding dimension: {self.embedding_dim}')
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # x = x.int()
        x = self.embedding(x) # (B, T, embedding_dim)
        x = self.dropout1(x)
        x = torch.mean(x, dim=1) # (B, embedding_dim) average embedding
        x = F.relu(self.fc1(x))
        # x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return self.log_softmax(x)
        
