import numpy as np
import copy
def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.
    B - batch size
    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an BxNxL array
    - Transition scores (Yp -> Yc), as an BxLxL array
    - Start transition scores (S -> Y), as an BxLx1 array
    - End transition scores (Y -> E), as an BxLx1 array

    You have to return a tuple (scores, y_pred), where:
    - scores (B): scores of the best predicted sequences
    - y_pred (BxN): an list of list of integers representing the best sequence.
    """
    B = start_scores.shape[0]
    L = start_scores.shape[1]
    N = emission_scores.shape[1]
    assert end_scores.shape[1] == L
    assert trans_scores.shape[1] == L
    assert trans_scores.shape[2] == L
    assert emission_scores.shape[2] == L

    # score set to 0
    scores = np.zeros(B)
    y_pred = np.zeros((B, N))
    
    for i in range(N):
        # stupid sequence
        y_pred[:, i] = i * np.ones(B)
    return (scores, y_pred.tolist())
