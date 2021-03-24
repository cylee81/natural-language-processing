# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
from torch import optim
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class LSTM(nn.Module):

    def __init__(self, num_hidden=5, num_layers=2, output_size=2):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(27, 15)
        self.embedding.weight.requires_grad_(True)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lstm = nn.LSTM(
            input_size=15,
            hidden_size=num_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            )
        self.fc = nn.Linear(num_hidden, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def init_weight(self):
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=1)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)

    def forward(self, text):
        embeds = self.embedding(text)
        h0 = torch.zeros(self.num_layers, embeds.size(0), self.num_hidden)
        c0 = torch.zeros(self.num_layers, embeds.size(0), self.num_hidden)
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        out_space = self.fc(lstm_out)
        out_scores = self.log_softmax(out_space)
        return out_scores

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, network, indexer):
        self.model = network
        self.indexer = indexer
        self.model.train(False)
    def predict(self, context):
        index = [self.indexer.index_of(i) for i in context]
        index = np.array(index)
        index = torch.from_numpy(index).long()
        index = torch.unsqueeze(index, 0)
        out = self.model.forward(index)
        out = torch.argmax(out[0][-1])
        return out

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabu lary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    raw_train_x = train_cons_exs+train_vowel_exs
    train_y = [0]*len(train_cons_exs) + [1]*len(train_vowel_exs)

    train_x = []
    for data in raw_train_x:
        index = [vocab_index.index_of(i) for i in data]
        train_x.append(index)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    print(f"shape of train_x: {train_x.shape}")
    print(np.max(train_x))

    model = LSTM()
    epochs = 30
    batch_size = 64
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train(True)
    loss_func = nn.NLLLoss()
    for epoch in range(epochs):
        # train phase
        n_iter = len(train_x) // batch_size
        running_loss = 0.0
        idx = np.linspace(0, len(train_x)-1, len(train_x), dtype=int)
        random.shuffle(idx)
        train_x = train_x[idx, ...]
        train_y = train_y[idx]
        for i in range(n_iter):
            train_x_batch = train_x[i*batch_size:i*batch_size+batch_size, :]
            train_y_batch = train_y[i*batch_size:i*batch_size+batch_size]
            train_x_batch = torch.from_numpy(train_x_batch).float()
            train_y_batch = torch.from_numpy(train_y_batch)
            optimizer.zero_grad()
            outputs = model(train_x_batch.long())
            outputs = outputs[:, -1,:]
            outputs = torch.squeeze(outputs, 1)
            loss = loss_func(outputs, train_y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss/n_iter
        # epoch_acc = running_acc/n_iter
        print(f"loss: {epoch_loss}")
        # print(f"acc: {epoch_acc}")
    return RNNClassifier(model, vocab_index)

#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, rnn, indexer):
        self.rnn = rnn
        self.indexer = indexer
        self.rnn.train(False)

    def get_next_char_log_probs(self, context):
        idxs = [self.indexer.index_of(k) for k in context]
        idxs = torch.tensor(idxs, dtype=torch.long)
        probs = self.rnn(idxs.unsqueeze(0))
        probs = probs.squeeze(0)
        probs = probs[len(context)-1]
        return probs.detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        m = len(next_chars)
        context = list(context + next_chars[0:m-1])
        n = len(context)
        idxs = [self.indexer.index_of(k) for k in context]
        idxs = torch.tensor(idxs, dtype=torch.long)
        probs = self.rnn(idxs.unsqueeze(0))
        probs = probs.squeeze(0)
        i = n - m
        for char in next_chars:
            log_prob += probs[i][self.indexer.index_of(char)]
            i += 1
        return log_prob.item()

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    model = LSTM(output_size=27, num_hidden=25, num_layers=1)
    epochs = 25
    batch_size = 32
    lr = 0.01
    chunk_size = 20
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train(True)
    loss_func = nn.NLLLoss()

    for epoch in range(epochs):
    # train phase
        running_loss = 0.0
        for i in range(0, len(train_text), chunk_size*batch_size):
            train_x_batch = []
            train_y_batch = []
            for j in range(i, i+chunk_size*batch_size, chunk_size):
                if j+chunk_size < len(train_text)-1:
                    text_batch = train_text[j: j+chunk_size-1]
                    label_batch = train_text[j+1: j+chunk_size]
                    x_batch = [vocab_index.index_of(k) for k in text_batch]
                    y_batch = [vocab_index.index_of(k) for k in label_batch]
                    train_x_batch.append(x_batch)
                    train_y_batch.append(y_batch)
            optimizer.zero_grad()
            outputs = model(torch.tensor(train_x_batch).long())
            outputs = torch.transpose(outputs, 1, 2)
            loss = loss_func(outputs, torch.tensor(train_y_batch))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss/(i+1)
        print(f"loss: {epoch_loss}")
    return RNNLanguageModel(model, vocab_index)