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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss_total = 0
  for i in xrange(num_train):
      scores = X[i,:].dot(W)
      correct_class_score = scores[y[i]]
      level_score = scores - np.max(scores)
      score_exp = np.sum(np.exp(scores))
      loss = -correct_class_score + np.log(score_exp)
      loss_total += loss
      dW[:,y[i]] += -X[i]
      for j in xrange(num_classes):
          prob = np.exp(level_score[j])/np.sum(np.exp(level_score))
          dW[:,j] += prob*X[i]
         

  loss = loss / float(num_train) + 0.5 * reg * np.sum(W*W)
  dW = dW / float(num_train) + reg * W
   
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
  num_train = X.shape[1]
  num_class = W.shape[0]
  dW = np.zeros(W.shape) 

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #num_classes = W.shape[1]
  #num_train = X.shape[0]
  #scores = X.dot(W)
  #level_score = scores - np.max(scores,axis=1, keepdims=True)
  #score_exp = np.exp(level_score)
  #prob = score_exp/np.sum(score_exp, axis = 1, keepdims=True)
  
  #loss = -np.sum(score_exp, axis=0, keepdims=True))
  
  #loss = loss - prob[y,np.arange(num_train)]
  #loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)
  
  #prob[y,np.arange(num_train)] += -1.0
  #dW = prob.dot(X.T) / float(num_train) + reg*W
  
  #####not working found solution
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  loss /= num_train 
  loss +=  0.5* reg * np.sum(W * W)
  

  # compute the gradient
  # first compute dS: the gradient of the loss function with respect to the scores
  dS= softmax_output.copy()
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg* W 
                        
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

