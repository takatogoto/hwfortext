from __future__ import division
import sys,json,math
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_table(filename):
    # Returns a dictionary containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    table = {}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            vec=np.array([float(x) for x in line_split[1:]])
            table[w]=vec
    return table

def get_vector(w, table):
    # Returns a numpy array of a word in table
    # w: word token
    # table: lookup table obtained from load_table()
    
    ## TODO: delete this line and implement me
    return table[w]


def cossim(v1,v2):
    # v1 and v2 are numpy arrays
    # Compute the cosine simlarity between them.
    # Should return a number between -1 and 1
    
    ## TODO: delete this line and implement me
    pass

def show_nearest(table, v, exclude_w, n = 1, sim_metric=cossim):
    #table: lookup table obtained from load_table()
    #v: query word vector (numpy arrays)
    #exclude_w: the words you want to exclude in the responses. It is a set in python.
    #sim_metric: the similarity metric you want to use. It is a python function
    # which takes two word vectors as arguments.

    # return: an iterable (e.g. a list) of n tuples of the form (word, score) where the nth tuple indicates the nth most similar word to the input word and the similarity score of that word and the input word after excluding exclude_w
    # if fewer than n words are available the function should return a shorter iterable
    #
    # example:
    #[(cat, 0.827517295965), (university, -0.190753135501)]
    
    ## TODO: delete this line and implement me
    pass

