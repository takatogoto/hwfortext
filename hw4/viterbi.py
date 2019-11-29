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
    y_pred = np.zeros((B, N), dtype=int)

    # my code
    #print 'B, N, L are', B, N, L
    # creat array for viteri score
    T = np.zeros((B, N, L)) # BxNxL
    # creat array for backpointer
    # backpointer stores index, so it should be integer
    backpointer = -np.zeros((B, N, L), dtype=int) # BxNxL
    
    # initialize backpinter to last label
    #backpointer[:, -1, :] = L+1

    # initialize T
    T[:, 0, :] = start_scores + emission_scores[:, 0, :] # BXL
    # initialize backpointer
    backpointer[:, 0, :] = np.tile(np.array(range(L)), (B,1))  # BXL
    #print 'T i', 0, T[:, 0 , :]

    
    for i in range(N)[1:]:
        for l in range(L) :
            # calmax store value of \psi_t (y',l)+T(i-1,y')
            #calmax = np.tile(trans_scores[:, :, l].reshape(B,L), (B,1)) +T[:,i-1, :] # BxL
            #calmax = np.tile(trans_scores[:, :, l], (B,1)) +T[:,i-1, :]
            calmax = trans_scores[:, :, l].reshape(B, L) +T[:,i-1, :] # BxL
            #print "calmax shape", calmax.shape
            T[:, i, l] = emission_scores[:, i, l] + np.max(calmax, axis=1) # Bx1
            backpointer[:, i, l] = np.argmax(calmax, axis=1)   # Bx1
            #print 'prev val ', l,  calmax
    
    # be careful to endscores
    T[:, -1, :] +=  end_scores # Bx1
    
    # take max to path
    y_pred[:, -1]= np.argmax(T[:, -1, :], axis=1).astype(int)
    for i in range(N-1)[::-1]:
        #y_pred[:, i] = int(backpointer[:, i+1, y_pred[:, i+1]])
        #y_pred[:, i] = backpointer[:, i+1, y_pred[:, i+1]]
        # refered from Piazza discussion
        y_pred[:, i] = backpointer[np.arange(B), i+1, y_pred[:, i+1]]
    # scores are just max T[]
    scores = np.max(T[:,-1,:], axis=1)
    
    return (scores, y_pred.tolist())
