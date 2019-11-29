#!/bin/python

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.
    Of course, this is an optional function.
    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    # matching lexicons
    folder = 'lexicon/'
    x1dir = folder + 'location'
    x2dir = folder + 'people.person.lastnames'
    x3dir = folder + 'product'
    x4dir = folder + 'transportation.road'
    x5dir = folder + 'business.consumer_product'

    from pathlib import Path
    def lexicon_list(path):
        lexpath = Path(path)
        lexlist=[]
        with lexpath.open('r', encoding='utf-8') as r:
            for i, strline in enumerate(r):
                splitstr = str(strline.encode("utf-8")).split()
                lexlist.extend(splitstr)
        return lexlist
    
    global x1_list, x2_list, x3_list, x4_list, x5_list

    #if not 'x1_list' in globals():
    #    x1_list = lexicon_list(x1dir)
    
    #if not 'x2_list' in globals():
    #    x2_list = lexicon_list(x2dir)
    
    #if not 'x3_list' in globals():
    #    x3_list = lexicon_list(x3dir)
    
    if not 'x4_list' in globals():
        x4_list = lexicon_list(x4dir)
    
    if not 'x5_list' in globals():
        x5_list = lexicon_list(x5dir)

def token2features(sent, i, orig_len, add_neighs = True):
    """Compute the features of a token.
    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.
    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.
    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.
    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    
    ftrs = []
    # bias
    ftrs.append("BIAS")
    if i >= orig_len:
        ftrs.append("<PAD>")
        return ftrs
    
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == orig_len-1:
        ftrs.append("SENT_END")

    # the word itself
    #word = unicode(str(sent[i]), 'utf-8') # str doesn't have an isnumeric method
    word = str(sent[i])
    
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())


    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric(): # str doesn't have an isnumeric method
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # additional features of the word

    #if word in x1_list:
    #    ftrs.append("IS_X1")
    
    #if word in x2_list:
    #    ftrs.append("IS_X2")
    
    #if word in x3_list:
    #    ftrs.append("IS_X3")
    
    if word in x4_list:
        ftrs.append("IS_X4")

    if word in x5_list:
        ftrs.append("IS_X5")
        
    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, orig_len, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, orig_len, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs


    
if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        orig_len = len(sent)
        for i in range(len(sent)):
            print(sent[i], ":", token2features(sent, i, orig_len))
