#!/bin//python
from feats import Feats
import scipy.sparse
import numpy as np
from sklearn.metrics import *
import struct_perceptron

class Tagger:
    """Parent class for taggers, supports training, tagging and evaluation."""

    def tag_sent(self, sent):
        """Tag sentence with the predicted labels."""
        pass

    def fit_data(self, sents, labels):
        """Learn the parameters of the model from the given labeled data."""
        pass

    def tag_sent(self, sent):
        """Predict the best tags for a sequence."""
        pass

    def tag_data(self, sents):
        """Tag all the sentences in the list of sentences."""
        pred = []
        for s in sents:
            pred.append(self.tag_sent(s))
        return pred

    def evaluate_data(self, sents, labels):
        """Evaluates the tagger on the given corpus of sentences and the set of true labels."""
        preds = [x.tolist() for x in self.tag_data(sents)]
        assert len(preds) == len(labels)
        # Compute tokenwise predictions and labels
        all_preds = []
        all_labels = []
        for i in range(len(preds)):
            assert len(preds[i]) == len(labels[i])
            for p in preds[i]:
                all_preds.append(p)
            for l in labels[i]:
                all_labels.append(l)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Token-wise accuracy", accuracy_score(all_labels, all_preds)*100)
            print("Token-wise F1 (macro)", f1_score(all_labels, all_preds, average='macro')*100)
            print("Token-wise F1 (micro)", f1_score(all_labels, all_preds, average='micro')*100)
            print("Sentence-wise accuracy", accuracy_score([''.join(l) for l in labels], 
                                                           [''.join(l) for l in preds])*100)
            print(classification_report(all_labels, all_preds))
        return preds

class LogisticRegressionTagger(Tagger):
    """A simple logistic regression based classifier.

    Converts the sequence labeling task to independent per-token classification.
    The features for each token are generated using a feats.Feats() object.
    """
    def __init__(self, feats = Feats()):
        self.feats = feats
        from sklearn.linear_model import LogisticRegression
        self.cls = LogisticRegression()
        from sklearn import preprocessing
        self.le = preprocessing.LabelEncoder()

    def tag_sent(self, sent):
        """Returns the predicted tags of a sentence.

        input: a sentence as a list of strings.
        output: predicted labels as a list of string.
        """
        fvs = []
        for i in range(len(sent)):
            fidxs = self.feats.token2fidxs(sent, i)
            fv = self.idxs2featurevector(fidxs)
            fvs.append(fv)
        X = scipy.sparse.vstack(fvs)
        y = self.cls.predict(X)
        return self.le.inverse_transform(y)

    def idxs2featurevector(self, idxs):
        """Given the indexes of the features, construct a sparse feature vector."""
        assert self.feats.frozen == True
        from scipy.sparse import csc_matrix
        import numpy as np
        fdata = np.full((len(idxs)), True, dtype=np.bool)
        frow = np.full((len(idxs)), 0, dtype=np.int32)
        fv = csc_matrix((fdata, (frow, idxs)), dtype=np.bool, shape=(1,self.feats.num_features))
        return fv

    def fit_data(self, sents, labels):
        """Train the tagger on the given dataset.

        The input is a sequence of sentences and corresponding labels,
        where each sentence and sequence of labels are lists of strings.
        """

        # get the set of all the labels
        all_labels = []
        for ls in labels:
            for l in ls:
                all_labels.append(l)
        # transform it to a list of classes
        # size N (number of total tokens)
        y = self.le.fit_transform(all_labels)
        print(y.shape)
        # get the feature indices
        # list of size N (number of total tokens)
        Xidxs = self.feats.index_data(sents)

        print("Features computed")
        # convert to feature vectors
        # list of size N
        Xfeats = []
        for sentIdxs in Xidxs:
            for tokIdxs in sentIdxs:
                Xfeats.append(self.idxs2featurevector(tokIdxs))
                
        # stack them to create a single feature matrix
        # of size NxD, where D is the total number of features

        assert len(Xfeats) == len(all_labels)
        X = scipy.sparse.vstack(Xfeats)
        print(X.shape)
        # train the classifier
        self.cls.fit(X,y)

class CRFPerceptron(Tagger):
    """A Conditional Random Field version of the sequence tagger.

    The underlying model uses features for the "emission" factors, but ignores
    them for the transition. Thus, if the number of labels is L, number of features
    is D, then the parameters for this model contain (in this order):
    - start transition weights: size L
    - end transition weights: size L
    - intermediate transitions: size LxL
    - emission feature weights: size LxD

    The features are that used are the same ones as logistic regression, i.e. look
    at feats.py/feat_gen.py for details.

    The training for the CRF is based on structured perceptron. Please change the 
    parameters of the StructuredPerceptron below if needed (see struct_perceptron.py
    for more details).

    The MAP inference is based on Viterbi, currently unimplemented in viterbi.py.
    If the viterbi_test.py passes succesfully, this tagger should train/tag correctly.
    """
    def __init__(self, epochs = 10, batch_size = 8, feats = Feats()):
        self.feats = feats
        from sklearn import preprocessing
        self.le = preprocessing.LabelEncoder()
        self.cls = struct_perceptron.StructuredPerceptron(self, max_iter=epochs, batch_size=batch_size, 
                                                          average=True, verbose=True)
        # max_len is set to be longer than any sentences in trainset
        # if you want to use smaller number, you may need to need to spend some time debugging
        self.max_len = 40
        
        self.batch_size = batch_size

    def tag_sent(self, sent):
        """Calls viterbi code to find the best tags for a sentence."""
        # Compute the features for the sentence
        Xidxs = []
        for i in range(len(sent)):
            fidxs = self.feats.token2fidxs(sent, i)
            Xidxs.append(fidxs)
        # All the inference code
        yhat = self.inference([Xidxs], self.cls.w)
        # Convert the labels to string
        return self.le.inverse_transform(yhat[0])

    # These functions are specific to how weights are stored in CRFs

    def get_start_trans_idx(self, y):
        """Get the weight index that represents S->y transition."""
        # no offset here, these are at the beginning
        assert y < self.num_classes
        return y

    def get_end_trans_idx(self, y):
        """Get the weight index that represents y->E transition."""
        # offset only because the first L are for start trans
        assert y < self.num_classes
        offset = self.num_classes
        return offset + y

    def get_trans_idx(self, yp, yc):
        """Get the weight index that represents yp->yc transition."""
        # offset only because the first 2xL are for start/end trans
        L = self.num_classes
        assert yp < L
        assert yc < L
        offset = 2*L
        index = yp*L + yc
        return offset + index

    def get_ftr_idx(self, fidx, y):
        """Get the weight index that represents feat(fidx,y)."""
        # offset because of transition weights, which are 2*L + L^2
        L = self.num_classes
        offset = 2*L + L*L
        index = self.feats.num_features*y + fidx
        return offset + index

    def joint_feature(self, Xs, ys):
        """For a given sentence (represented as seq of feature indices) and
        a tag sequence (represented by a seq of integers), compute the joint
        feature vector.
        """

        assert len(ys) == len(Xs)
        fv = np.full((1, self.size_joint_feature), 0, dtype=np.int32)
        # init_trans
        fv[0,self.get_start_trans_idx(ys[0])] = 1
        # final_trans
        fv[0,self.get_end_trans_idx(ys[-1])] = 1
        # intermediate transitions
        for i in range(1, len(ys)):
            tidx = self.get_trans_idx(ys[i-1], ys[i])
            fv[0,tidx] = fv[0,tidx] + 1
        # features
        for i in range(len(ys)):
            X = Xs[i]
            y = ys[i]
            for c in X:
                fidx = self.get_ftr_idx(c, y)
                fv[0,fidx] = fv[0,fidx] + 1
        return fv #.tocsc()

    def fit_data(self, sents, labels):
        """Train the tagger by calling the structured perceptron code."""
        # get the set of all the labels
        all_labels = []
        for ls in labels:
            for l in ls:
                all_labels.append(l)
        self.le.fit(all_labels)
        
        # Get the sequence of gold label sequences, i.e. y in seq of seq of ints
        y = []
        for ls in labels:
            y.append(self.le.transform(ls))
        print("Classes:", len(self.le.classes_), self.le.classes_)
        # compute all the token features, store as seq of seq of feature indices
        # i.e. each token has a list of feature indices
        Xidxs = self.feats.index_data(sents, self.max_len)
    
        assert len(Xidxs) == len(y)
        print(len(Xidxs), self.feats.num_features)

        # train
        self.num_classes = len(self.le.classes_)
        L = self.num_classes
        self.size_joint_feature = 2*L + L*L + L*self.feats.num_features
        print("Number of weights", self.size_joint_feature)
        print("Starting training")
        # profiling code below, in case code is incredibly slow
        # import cProfile, pstats, StringIO
        # pr = cProfile.Profile()
        # pr.enable()
        
        self.cls.fit(Xidxs, y, False)

        # pr.disable()
        # s = StringIO.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print s.getvalue()

    def load_weights(self, w, L, X):
        B, N = len(X), len(X[0])

        start_scores = np.zeros((B, L))
        end_scores = np.zeros((B, L))
        trans_scores = np.zeros((B, L, L))
        emission_scores = np.zeros((B, N, L))

        for j in range(L):
            start_scores[:, j] = np.repeat(w[0,self.get_start_trans_idx(j)], B)
            end_scores[:, j] = np.repeat(w[0,self.get_end_trans_idx(j)], B)

            # transition                                                                                                  
            for k in range(L):
                trans_scores[:, j, k] = np.repeat(w[0,self.get_trans_idx(j, k)], B)

            # emission        
            for b in range(B):
                for i in range(N):
                    score = 0.0
                    for fidx in X[b][i]:
                        score += w[0,self.get_ftr_idx(fidx, j)]
                    emission_scores[b][i][j] = score

        return start_scores, end_scores, trans_scores, emission_scores

    def inference(self, X, w):
        """Run Viterbi inference.

        This methods is a wrapper that converts the CRF weights into
        different arrays of scores that represent transition and emission.
        Then this method can call the general purpose Viterbi code in
        viterbi.py to compute the best label sequence.

        This function just returns the best sequence, y.
        """
        from viterbi import run_viterbi
        L = self.num_classes
        start_scores, end_scores, trans_scores, emission_scores = self.load_weights(w, L, X)

        # now run the viterbi code!
        score, yhat = run_viterbi(emission_scores, trans_scores, start_scores, end_scores)
        
        return yhat

    def loss(self, yhat, y):
        """Tokenwise 0/1 loss, for printing and evaluating during training."""
        tot = 0.0
        for i in range(len(y)):
            if yhat[i] != y[i]:
                tot += 1.0
        return tot

    def max_loss(self, labels):
        """Maximum loss that a sentence that get, same as length tokenwise mismatch."""
        return len(labels)

