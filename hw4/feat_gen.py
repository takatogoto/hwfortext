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
    loca = folder + 'location'
    pep = folder + 'people.person.lastnames'
    bigd = folder + 'bigdict'

    from pathlib import Path
    def lexicon_list(path):
        lexpath = Path(path)
        lexlist=[]
        with lexpath.open('r', encoding='utf-8') as r:
            for i, strline in enumerate(r):
                splitstr = str(strline.encode("utf-8")).split()
                lexlist.extend(splitstr)
        return lexlist
    
    global loca_list, peop_list, bigd_list

    loca_list = lexicon_list(loca)
    peop_list = lexicon_list(pep)
    bigd_list = lexicon_list(bigd)


    global mono_fea
    mono_fea = []
    for sent in train_sents:
        #print sent
        orig_len= len(sent)
        #print "len", len(sent)
        for i in range(len(sent)):
            mono_fea.append(token2features(
                sent, i, orig_len, add_neighs = False))

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
        # sent: a sentence, list of str
    # i: index for computing feature extraction in sent
    # orig_len: orignal length of sent
    # add_neighs: flag for computing neighbors
    # print add_neighs


    if not add_neighs:
        #print "false"
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
        if word in loca_list:
            ftrs.append("IS_LOCAL")
        if word in  peop_list:
            ftrs.append("IS_PEOPLE")
        if word in bigd_list:
            ftrs.append("IS_BIGDICT")
        return ftrs

    # previous/next word feats
    
    elif add_neighs and orig_len==0:
        return mono_fea[i]

    else:
        assert len(mono_fea) == orig_len, 'expect[{0}], got[{1}]'.format(orig_len, len(mono_fea))
        import copy
        ftrs = copy.copy(mono_fea[i])
        #print 'ftrs', ftrs
        if i > 0:
            #print "iter", i, fea
            for pf in mono_fea[i-1]:
                ftrs.append("PREV_" + pf)
                
        if i < orig_len-1:
            #print "iter", i,  mono_fea[i+1]
            for pf in  mono_fea[i+1]:
                #print "pf ", pf
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
