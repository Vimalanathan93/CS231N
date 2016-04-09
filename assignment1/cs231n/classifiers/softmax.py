import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train= X.shape[0]
    num_classes=W.shape[1]
    loss=0.0
    for i in xrange(num_train):
        scores = X[i,:].dot(W)
        c=np.max(scores)
        scores-=c #Normalization for stability
        sums=0.0
        for j in scores:
            sums+= np.exp(j)
        loss+=  - scores[y[i]] + np.log(sums)
        for k in xrange(num_classes):
            m=np.exp(scores[k])/sums
            dW[:,k] += (m-(k==y[i]))*X[i,:]
     
    loss /= num_train
    loss += 0.5*reg*np.sum(W*W)
    dW += reg*W
            
        

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_classes=W.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores=X.dot(W) #N X 10
    scores -= np.max(scores) # N X 10
    correct_scores=np.exp(scores[np.arange(num_train),y]) # N X 1
    expi=np.exp(scores) # N X 10
    sums= np.sum(expi,axis=1) # N X 1
    loss = -np.sum(np.log(correct_scores/sums))/num_train # 1X1 
    loss += 0.5*reg*np.sum(W*W) # 1X1
    m = expi.T/sums
    m=m.T
    z = np.zeros(m.shape)
    z[np.arange(num_train),y] = 1
    dW = np.dot(X.T,(m-z))
     
    
    dW /= num_train
    dW +=reg*W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

