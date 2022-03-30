"""
MyModel model.  (c) 2021 Georgia Tech

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


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        # building resnet34
        # reference: 
        # [1] https://arxiv.org/abs/1512.03385 (original paper)
        # [2] https://www.zhihu.com/search?type=content&q=resnet
        # [3] https://www.kaggle.com/poonaml/building-resnet34-from-scratch-using-pytorch
        # [4] https://www.analyticsvidhya.com/blog/2021/09/building-resnet-34-model-using-pytorch-a-guide-for-beginners/
        # [5] https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb
        # [6] https://github.com/rasbt/deeplearning-models/blob/4153375a9288d6ae5a7070132b47754e2f344a41/pytorch_ipynb/cnn/resnet-ex-1.ipynb

        # hyper-params
        C0, C1, C2, C3 = 3, 32, 64, 128
        kernel_size = 3
        stride = 1
        padding = 1
        pool_kernel_size = 2
        num_classes = 10

        # block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=C0, out_channels=C1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C1),
            nn.ReLU(),
            nn.Conv2d(in_channels=C1, out_channels=C1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )

        # shortcut 1
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=C1, out_channels=C2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C2)
        )

        # block 2.1
        self.block2_1 = nn.Sequential(
            nn.Conv2d(in_channels=C1, out_channels=C2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C2),
            nn.ReLU(),
            nn.Conv2d(in_channels=C2, out_channels=C2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C2),
        )

        # block 2.2
        self.block2_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )

        # shortcut 2
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=C2, out_channels=C3, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C3)
        )

        # block 3.1
        self.block3_1 = nn.Sequential(
            nn.Conv2d(in_channels=C2, out_channels=C3, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C3),
            nn.ReLU(),
            nn.Conv2d(in_channels=C3, out_channels=C3, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=C3),
        )

        # block 3.2
        self.block3_2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )

        # fully-connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=C3 * 4 * 4, out_features=C3),
            nn.ReLU(),
            nn.Linear(in_features=C3, out_features=num_classes)
        )

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        outs = self.block1.forward(x)
        res1 = self.shortcut1.forward(outs)
        outs = self.block2_1.forward(outs)
        outs = self.block2_2.forward(outs + res1)
        res2 = self.shortcut2(outs)
        outs = self.block3_1.forward(outs)
        outs = self.block3_2.forward(outs + res2)
        # flatten before entering fully-connected layers
        outs = torch.flatten(outs, 1)
        outs = self.fc(outs)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
