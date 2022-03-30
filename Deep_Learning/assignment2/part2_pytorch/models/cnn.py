"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        # data: 3*32*32

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=7,
                               stride=1,
                               padding=0)
        # H=W=(32-7+1)=26
        # out: (32, 26, 26)

        self.relu1 = nn.ReLU()
        # out: (32, 26, 26)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # H=W=(26-2)/2+1=13
        # out: (32, 13, 13)

        self.fc1 = nn.Linear(in_features=32*13*13,
                             out_features=10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.conv1(x)
        outs = self.relu1(outs)
        outs = self.pool1(outs)
        outs = self.fc1(torch.flatten(outs, 1))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
