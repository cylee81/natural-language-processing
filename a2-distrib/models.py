# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


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


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self):
        raise NotImplementedError
    def predict(self, ex_words: List[str]) -> int:

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:


def file2embed(sentiment_example, max_len=52, word_vec_len=50):
#     file = 'data/train.txt'
#     max_len = 52
    total_sen = len(sentiment_example)
    res = np.zeros(shape=(total_sen, max_len*50))
    labels = []
    for i in range(total_sen):
        word_ems = []
        for j in range(max_len):
            if j >= len(sentiment_example[i].words):
                word_ems.extend(word_embed.get_embedding("PAD"))
            else:
                word_ems.extend(word_embed.get_embedding(sentiment_example[i].words[j]))
        res[i, :] = np.array(word_ems)
        labels.append(sentiment_example[i].label)
    return res, np.array(labels)
    
def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    train_x, train_y = 




    raise NotImplementedError

