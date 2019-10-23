from distsim import *

query = 'king'
print("Test nearest neighbors for GloVe...")
table = load_table('glove.100d.5K.txt')
v = get_vector(query, table)
print(show_nearest(table, v, set([query]), n=10))

print("Test nearest neighbors for word2vec...")
table = load_table('word2vec.100d.5K.txt')
v = get_vector(query, table)
print(show_nearest(table, v, set([query]), n=10))
