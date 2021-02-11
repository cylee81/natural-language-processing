# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, embedding):
        self.model = network
        self.model.train(False)
        self.embedding = embedding

    def predict(self, ex_words: List[str]) -> int:
        
        word_sum = np.zeros(300)
        for j in range(len(ex_words)):
            word_embedding = self.embedding.get_embedding(ex_words[j])
            word_sum += np.array(word_embedding)
        
        word_avg = word_sum / len(ex_words)
        train_x_batch = torch.unsqueeze(torch.from_numpy(word_avg).float(), 0)
        out = self.model.forward(train_x_batch)[0]
        return torch.argmax(out)

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]

def file2embed(sentiment_example, word_embed, max_len=52, word_vec_len=300):
    total_sen = len(sentiment_example)
    res = []
    labels = []
    word_sum = np.zeros(word_vec_len)

    for i in range(total_sen):
        len_sent = len(sentiment_example[i].words)
        word_sum = np.zeros(word_vec_len)
        for j in range(len_sent):
            embedding = word_embed.get_embedding(sentiment_example[i].words[j])
            word_sum += np.array(embedding)
        word_avg = word_sum / total_sen
        res.append(word_avg)
        label = sentiment_example[i].label
        labels.append(label)
    return np.array(res), np.array(labels)

    
def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    train_x, train_y = file2embed(train_exs, word_embeddings)
    train_x = train_x*1000
    val_x, val_y = file2embed(dev_exs, word_embeddings)
    model = Net()
    epochs = 100
    batch_size = 32
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=1, factor=0.5)
    model.train(True)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # train phase
        n_iter = len(train_x) // batch_size
        running_loss = 0.0
        running_acc = 0.0
        idx = np.linspace(0, len(train_x)-1, len(train_x), dtype=int)
        random.shuffle(idx)
        train_x = train_x[idx, ...]
        train_y = train_y[idx, ...]
        for i in range(n_iter):
            train_x_batch = train_x[i:i+batch_size, :]
            train_y_batch = train_y[i:i+batch_size]
            train_x_batch = torch.from_numpy(train_x_batch).float() 
            train_y_batch = torch.from_numpy(train_y_batch)
            optimizer.zero_grad()
            outputs = model(train_x_batch)
            loss = loss_func(outputs, train_y_batch)
            loss.backward()
            output_argmax = torch.argmax(outputs, axis=1)
            # running_acc += sum(output_argmax.numpy() == train_y_batch.numpy()) / len(output_argmax)
            optimizer.step()
        #     running_loss += loss.item()
        epoch_loss = running_loss/n_iter
        # epoch_acc = running_acc/n_iter
        # print(f"loss: {epoch_loss}")
        # print(f"acc: {epoch_acc}")
        scheduler.step(epoch_loss)

    return NeuralSentimentClassifier(model, word_embeddings)


# def file2embed_pad(sentiment_example, word_embed, max_len=52, word_vec_len=50):
#     total_sen = len(sentiment_example)
#     res = np.zeros(shape=(total_sen, max_len*50))
#     labels = []
#     for i in range(total_sen):
#         word_ems = []
#         for j in range(max_len):
#             if j >= len(sentiment_example[i].words):
#                 word_ems.append(word_embed.get_embedding("PAD"))
#             else:
#                 word_ems.append(word_embed.get_embedding(sentiment_example[i].words[j]))
#         res[i, :] = np.array(word_ems)
#         # label = np.zeros(2)
#         # label[sentiment_example[i].label] = 1
#         label = sentiment_example[i].label
#         labels.append(label)
#     return np.array(res), np.array(labels)
