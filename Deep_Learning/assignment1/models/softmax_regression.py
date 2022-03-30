""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        N = X.shape[0]  # batch size
        num_features = self.weights['W1'].shape[0]
        hidden_size = self.weights['W1'].shape[1]

        Z = X @ self.weights['W1']  # (N, hidden_size)
        A = self.ReLU(Z)  # (N, hidden_size)
        p = x_pred = self.softmax(A)  # (N, hidden_size)
        loss = self.cross_entropy_loss(x_pred, y)  # (1, 1)
        accuracy = self.compute_accuracy(x_pred, y)  # (1, 1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        # W gradient = dL/dW = dL/dp * dp/dA * dA/dZ * dZ/dW

        # dL/dp
        dL_dp = np.zeros((N, hidden_size))  # (N, hidden_size)
        dL_dp[range(N), y] = -(1/N) * (1/p[range(N), y])

        # dp/dA, matrix multiply
        dp_dA = np.zeros((N, hidden_size, hidden_size))  # (N, hidden_size, hidden_size)
        dL_dA = np.zeros((N, hidden_size))  # (N, hidden_size)
        for i in range(dp_dA.shape[0]):
            s = p[i, :].reshape(hidden_size, 1)
            dev_matrix = -s * s.T  # (hidden_size, hidden_size)
            dev_matrix[range(hidden_size), range(hidden_size)] = (s * (1-s).T)[range(hidden_size), range(hidden_size)]
            dp_dA[i, :, :] = dev_matrix
            dL_dA[i, :] = dL_dp[i, :] @ dev_matrix

        # dA/dZ, element-wise multiply
        dA_dZ = self.ReLU_dev(Z)  # (N, hidden_size)
        dL_dZ = dL_dA * dA_dZ  # (N, hidden_size)

        # dZ/dW, matrix multiply
        dZ_dW = X  # (N, num_features)
        dL_dW = dZ_dW.T @ dL_dZ   # (num_features, hidden_size)
        self.gradients['W1'] = dL_dW

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy
