from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        # Contador de margens violadas para a amostra i
        margins_violated = 0

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                margins_violated += 1
                
                # Gradiente para as classes incorretas que violaram a margem
                dW[:, j] += X[i]
        
        # Gradiente para a classe correta
        dW[:, y[i]] -= margins_violated * X[i]
        
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Média sobre o número de dados de treino
    dW /= num_train

    # Adiciona a derivada da regularização L2
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]

    # Matriz de pontuações (N, C)
    scores = X.dot(W)
    
    # Extrai a pontuação da classe correta para cada amostra (dimensão N, 1)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    
    # Calcula a matriz de margens max(0, scores - correct_class_score + 1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    
    # Zera a margem para as classes corretas (j = y_i)
    margins[np.arange(num_train), y] = 0
    
    # Média da perda de dados + Regularização L2
    data_loss = np.sum(margins) / num_train
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Cria uma matriz binária indicando onde a margem foi violada (> 0)
    binary = (margins > 0).astype(float)
    
    # Conta quantas margens foram violadas por exemplo (soma nas colunas)
    margin_violations_per_example = np.sum(binary, axis=1)
    
    # Na coluna da classe correta, subtrai a contagem total de violações
    binary[np.arange(num_train), y] = -margin_violations_per_example
    
    # Gradiente dos dados + derivada da regularização L2
    dW = X.T.dot(binary) / num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
