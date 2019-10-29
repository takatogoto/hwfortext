#!/usr/bin/env python
import argparse

# complete this code
# you will need to run this script twice to predict glove and word2vec respectively
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-emb", default="glove.100d.5K.txt", help="glove emb file")
  parser.add_argument("-analogy", default="word-test.v3.txt", help="analogy file")
  parser.add_argument("-outfile", default="pred.glove.txt", help="w2v output file")
  
  print(args.emb)
                                             

if __name__ == '__main__':
  main()
