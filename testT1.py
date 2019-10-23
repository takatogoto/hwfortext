from distsim import *

w1 = 'king'
w2 = 'man'
w3 = 'woman'

print("Test loading word vector for GloVe...")
table = load_table('glove.100d.5K.txt')
print(get_vector(w1, table))
print(get_vector(w2, table))
print(get_vector(w3, table))

print("Test loading word vector for word2vec...")
table = load_table('word2vec.100d.5K.txt')
print(get_vector(w1, table))
print(get_vector(w2, table))
print(get_vector(w3, table))
