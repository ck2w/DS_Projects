""" 			  		 			     			  	   		   	  			  	
MLP Model.  (c) 2021 Georgia Tech

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

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        N = X.shape[0]  # batch size
        num_features = self.weights['W1'].shape[0]
        hidden_size1 = self.weights['W1'].shape[1]
        hidden_size2 = self.weights['W2'].shape[1]

        Z1 = X @ self.weights['W1'] + self.weights['b1']  # (N, hidden_size1)
        Z2 = self.sigmoid(Z1)  # (N, hidden_size1)
        Z3 = Z2 @ self.weights['W2'] + self.weights['b2']  # (N, hidden_size2)
        p = x_pred = self.softmax(Z3)  # (N, hidden_size2)
        loss = self.cross_entropy_loss(x_pred, y)  # (1, 1)
        accuracy = self.compute_accuracy(x_pred, y)  # (1, 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        # dL/dW2 = dL/dp * dp/dZ3 * dZ3/dW2 = dL/dZ3 * dZ3/dW2
        # dL/db2 = dL/dp * dp/dZ3 * dZ3/db2
        # dL/dW1 = dL/dp * dp/dZ3 * dZ3/dZ2 * dZ2/dZ1 * dZ1/dW1
        # dL/dW1 = dL/dp * dp/dZ3 * dZ3/dZ2 * dZ2/dZ1 * dZ1/db1

        # dL/dp
        dL_dp = np.zeros((N, hidden_size2))  # (N, hidden_size2)
        dL_dp[range(N), y] = -(1 / N) * (1 / p[range(N), y])

        # dp/dZ3
        dp_dZ3 = np.zeros((N, hidden_size2, hidden_size2))  # (N, hidden_size2, hidden_size2)
        dL_dZ3 = np.zeros((N, hidden_size2))  # (N, hidden_size2)
        for i in range(dp_dZ3.shape[0]):
            s = p[i, :].reshape(hidden_size2, 1)
            dev_matrix = -s * s.T  # (hidden_size2, hidden_size2)
            dev_matrix[range(hidden_size2), range(hidden_size2)] = (s * (1 - s).T)[range(hidden_size2), range(hidden_size2)]
            dp_dZ3[i, :, :] = dev_matrix
            dL_dZ3[i, :] = dL_dp[i, :] @ dev_matrix

        # dZ3/dZ2
        dZ3_dZ2 = self.weights['W2']  # (hidden_size1, hidden_size2)
        dL_dZ2 = dL_dZ3 @ dZ3_dZ2.T  # (N, hidden_size1)

        # dZ2/dZ1
        dZ2_dZ1 = self.sigmoid_dev(Z1)  # (N, hidden_size1)
        dL_dZ1 = dL_dZ2 * dZ2_dZ1  # (N, hidden_size1)


        ## weights

        # dZ3/dW2
        dZ3_dW2 = Z2  # (N, hidden_size1)
        dL_dW2 = dZ3_dW2.T @ dL_dZ3  # (hidden_size1, hidden_size2)

        # dZ3/db2
        dZ3_db2 = np.ones(N)  # (1, N)
        dL_db2 = dZ3_db2 @ dL_dZ3  # (hidden_size2)

        # dZ1/dW1
        dZ1_dW1 = X  # (N, num_features)
        dL_dW1 = dZ1_dW1.T @ dL_dZ1  # (num_features, hidden_size1)

        # dZ3/db2
        dZ1_db1 = np.ones(N)  # (1, N)
        dL_db1 =  dZ1_db1 @ dL_dZ1  # (hidden_size1)

        self.gradients['W1'] = dL_dW1  # (num_features, hidden_size1)
        self.gradients['b1'] = dL_db1  # (hidden_size1)
        self.gradients['W2'] = dL_dW2  # (hidden_size1, hidden_size2)
        self.gradients['b2'] = dL_db2  # (hidden_size2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy
