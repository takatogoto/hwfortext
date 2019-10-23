from distsim import *

w1 = 'king'
w2 = 'man'
w3 = 'woman'

print("Test cosine similarity for GloVe...")
table = load_table('glove.100d.5K.txt')
v1 = get_vector(w1, table)
v2 = get_vector(w2, table)
v3 = get_vector(w3, table)
print(cossim(v1, v2))
print(cossim(v1, v3))
print(cossim(v2, v3))

print("Test cosine similarity for word2vec...")
table = load_table('word2vec.100d.5K.txt')
v1 = get_vector(w1, table)
v2 = get_vector(w2, table)
v3 = get_vector(w3, table)
print(cossim(v1, v2))
print(cossim(v1, v3))
print(cossim(v2, v3))
