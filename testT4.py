from distsim import *

w1 = 'king'
w2 = 'man'
w3 = 'woman'

print("Test word analogy for GloVe...")
table = load_table('glove.100d.5K.txt')
v1 = get_vector(w1, table)
v2 = get_vector(w2, table)
v3 = get_vector(w3, table)
v4 = v1 - v2 + v3
print(show_nearest(table, v4, set([w1, w2, w3]), n=1))
      
print("Test word analogy for word2vec...")
table = load_table('word2vec.100d.5K.txt')
v1 = get_vector(w1, table)
v2 = get_vector(w2, table)
v3 = get_vector(w3, table)
v4 = v1 - v2 + v3
print(show_nearest(table, v4, set([w1, w2, w3]), n=1))
