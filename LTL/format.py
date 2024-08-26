"""
These functions preprocess the observations.
When trying more sophisticated encoding for LTL, we might have to modify this code.
"""

import os
import json
import re
import torch
import torch_ac
import gym
import numpy as np
# import utils

# from envs import *
from LTL.ltl_wrappers import LTLEnv
from LTL.ast_builder import ASTBuilder

def get_obss_preprocessor(env, gnn, progression_mode):
    obs_space = env.observation_space
    vocab_space = ['push', 'pickandplace']
    vocab = None

    if isinstance(env, LTLEnv): #LTLEnv Wrapped env
        env = env.unwrapped
        if progression_mode == "partial":
            obs_space = {"image": obs_space.spaces["features"].shape, "progress_info": len(vocab_space)}
            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": preprocess_images([obs["features"] for obs in obss], device=device),
                    "progress_info":  torch.stack([torch.tensor(obs["progress_info"], dtype=torch.float) for obs in obss], dim=0).to(device)
                })

        else:
            obs_space = {"image": obs_space.spaces["features"].shape, "text": max(22, len(vocab_space) + 10)}  # {'image': (76,), 'text': 22}
            vocab_space = {"max_size": obs_space["text"], "tokens": vocab_space}  # {'max_size': 22, 'tokens': ['J', 'W', 'R', 'Y']}

            vocab = Vocabulary(vocab_space)  # {'next': 1, 'until': 2, 'and': 3, 'or': 4, 'eventually': 5, 'always': 6, 'not': 7, 'True': 8, 'False': 9, 'J': 10, 'W': 11, 'R': 12, 'Y': 13}
            tree_builder = ASTBuilder(vocab_space["tokens"])  # encoder LTL formula to generate AST
            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": preprocess_images([obs["features"] for obs in obss], device=device),
                    "text":  preprocess_texts([obs["text"] for obs in obss], vocab, vocab_space, gnn=gnn, device=device, ast=tree_builder)
                })

        preprocess_obss.vocab = vocab  # it can be viewed as add an attribute to the function
                                       # which can be printed

    # Check if obs_space is an image space
    elif isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)  # list:16 (each list : len(list)=76) -> {ndarray: (16, 76)}
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_texts(texts, vocab, vocab_space, gnn=False, device=None, **kwargs):
    '''

    texts:
    [('until', ('not', 'Y'), 'J'),
    ('until', ('not', 'J'), ('and', 'R', ('until', ('not', 'Y'), 'W'))),
    ('until', ('not', 'W'), 'R'),
    ('until', ('not', 'R'), 'Y'),
    ('until', ('not', 'J'), 'W'),
    ('until', ('not', 'Y'), ('and', 'R', ('until', ('not', 'J'), 'W'))),
    ('until', ('not', 'W'), 'J'),
    ('until', ('not', 'W'), 'R'),
    ('until', ('not', 'J'), 'Y'),
    ('until', ('not', 'W'), ('and', 'R', ('until', ('not', 'Y'), 'J'))),
    ('until', ('not', 'J'), ('and', 'W', ('until', ('not', 'Y'), 'R'))),
    ('until', ('not', 'J'), ('and', 'R', ('until', ('not', 'W'), 'Y'))),
    ('until', ('not', 'W'), 'R'),
    ('until', ('not', 'Y'), ('and', 'R', ('until', ('not', 'W'), 'J'))),
    ('until', ('not', 'J'), ('and', 'Y', ('until', ('not', 'R'), 'W'))),
    ('until', ('not', 'Y'), 'J')]

    vocab:
    {'next': 1, 'until': 2, 'and': 3, 'or': 4, 'eventually': 5, 'always': 6, 'not': 7, 'True': 8, 'False': 9, 'J': 10, 'W': 11, 'R': 12, 'Y': 13}

    vocab_space:
    {'max_size': 22, 'tokens': ['J', 'W', 'R', 'Y']}

    gnn: True

    device: cpu

    kwargs: {'ast': <utils.ast_builder.ASTBuilder object at 0x7fd09ca4c470>}

    '''
    if (gnn):
        return preprocess4gnn(texts, kwargs["ast"], device)

    return preprocess4rnn(texts, vocab, device)


def preprocess4rnn(texts, vocab, device=None):
    """
    This function receives the LTL formulas and convert them into inputs for an RNN
    """
    var_indexed_texts = []
    max_text_len = 10

    for text in texts:
        text = str(text) # transforming the ltl formula into a string
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

def preprocess4gnn(texts, ast, device=None):
    """
    This function receives the LTL formulas and convert them into inputs for a GNN
    """
    return np.array([[ast(text).to(device)] for text in texts])


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, vocab_space):
        self.max_size = vocab_space["max_size"]
        self.vocab = {}

        # populate the vocab with the LTL operators
        for item in ['next', 'until', 'and', 'or', 'eventually', 'always', 'not', 'True', 'False']:
            self.__getitem__(item)

        for item in vocab_space["tokens"]:
            self.__getitem__(item)

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

if __name__ == '__main__':
    pass