#!/usr/bin/env python
from __future__ import division
import argparse
from pathlib import Path
from distsim import * # directory

# complete this code
# you will need to run this script twice to predict glove and word2vec respectively
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-emb", default="glove.100d.5K.txt", help="glove emb file")
  parser.add_argument("-analogy", default="word-test.v3.txt", help="analogy file")
  parser.add_argument("-outfile", default="pred.glove.txt", help="w2v output file")
  
  args = parser.parse_args()
  emb = args.emb
  analogy = args.analogy
  outfile = args.outfile
  
  #print('embbeding file is ', emb)
  #print('analogy file is ', analogy)
  #print('output file is ', outfile)
  
  table = load_table(emb)
  
  # path
  outfilepath = Path(outfile)
  analogypath = Path(analogy)
  
  # open new file
  #f = open(outfilepath, 'w')
  #f.close() only python3
  
  outlist = [] # output list
  
  # for compute best-1, best-5, best-10
  tn = 0.0
  ncbi = 0.0
  ncbv = 0.0
  ncbx = 0.0
  b1 = 0.0
  b5 = 0.0
  b10 = 0.0

  with analogypath.open('r') as r:
    next(r)
    #for strline in r:
    for i, strline in enumerate(r):
      if ':' in strline:
        #print(i, strline.strip('\n'))
        if i !=0:

          result = (gn + ': {best1:.2f} {best5:.2f} {best10:.2f}'.format(
              best1 =b1, best5=b5, best10=b10))
          outlist.append(result)
          print(result)
        
          # reset 1best 5best 10best
          tn = 0.0
          ncbi = 0.0
          ncbv = 0.0
          ncbx = 0.0
          b1 = 0.0
          b5 = 0.0
          b10 = 0.0
          
        gn = strline.strip(':''\n')

      else:
        tn += 1.0        
        # calculate w1-w2+w4
        W = strline.split()
        w1 = W[0]
        w2 = W[1]
        w3 = W[2]
        w4 = W[3]
        v1 = get_vector(w1, table)
        v2 = get_vector(w2, table)
        v4 = get_vector(w4, table)
        v3_cal = v1 - v2 + v4
        #print(type(w3)) ==> <type 'unicode'> in python2

        b1re = show_nearest(table, v3_cal, set([w1, w2, w4]), n=1)
        b5re = show_nearest(table, v3_cal, set([w1, w2, w4]), n=5)
        b10re = show_nearest(table, v3_cal, set([w1, w2, w4]), n=10)
        #print(type(b1re[0][0])) ==> <type 'str'> in python2
      
        for tp in b1re:
          if str(w3) in tp:
            ncbi += 1.0
        for tp in b5re:
          if str(w3) in tp:
            ncbv += 1.0
        for tp in b10re:
          if str(w3) in tp:
            ncbx += 1.0

        # compute 1best 5best 10best
        b1 = ncbi/tn
        b5 = ncbv/tn
        b10 = ncbx/tn
     
        if i%100 == 0:
          #print('Guess', w3, 'then ', show_nearest(table, v3_cal, set([w1, w2, w4]), n=5))
          continue
        
      

  result = (gn + ': {best1:.2f} {best5:.2f} {best10:.2f}'.format(
      best1 =b1, best5=b5, best10=b10))
  outlist.append(result)
  print(result)

  #with outfilepath.open('a') as w:
  #  for s in outlist:
  #    w.write(f'{s}\n')
  f = open(outfile, 'w')
  for s in outlist:
    f.write(s+'\n')
  f.close()
                                              
if __name__ == '__main__':
  main()
