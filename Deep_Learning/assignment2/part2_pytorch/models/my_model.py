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
        # data: 3*32*32
        kernel_size = 3
        padding_size = 1  # same size padding
        striding_size = 1

        ##############################################
        ## block 1
        self.conv11 = nn.Conv2d(in_channels=3,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=padding_size,
                                )
        self.relu11 = nn.ReLU()
        self.norm11 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=1,
                                )
        self.relu12 = nn.ReLU()
        self.norm12 = nn.BatchNorm2d(32)
        self.conv13 = nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=kernel_size,
                                padding=1,
                                )
        self.relu13 = nn.ReLU()
        self.norm13 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_shortcut1 = nn.Conv2d(in_channels=32,
                                       out_channels=64,
                                       kernel_size=kernel_size,
                                       padding=padding_size,
                                       )
        self.norm_shortcut1 = nn.BatchNorm2d(64)


        ##############################################
        ## block 2
        self.conv21 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=kernel_size,
                                padding=padding_size,
                                )
        self.relu21 = nn.ReLU()
        self.norm21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=kernel_size,
                                padding=1,
                                )
        self.relu22 = nn.ReLU()
        self.norm22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=kernel_size,
                                padding=1,
                                )
        self.relu23 = nn.ReLU()
        self.norm23 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv_shortcut2 = nn.Conv2d(in_channels=64,
                                        out_channels=128,
                                        kernel_size=kernel_size,
                                        padding=padding_size,
                                        )
        self.norm_shortcut2 = nn.BatchNorm2d(128)


        ##############################################
        ## block 3
        self.conv31 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=kernel_size,
                                padding=padding_size,
                                )
        self.relu31 = nn.ReLU()
        self.norm31 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=kernel_size,
                                padding=1,
                                )
        self.relu32 = nn.ReLU()
        self.norm32 = nn.BatchNorm2d(128)
        self.conv33 = nn.Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=kernel_size,
                                padding=1,
                                )
        self.relu33 = nn.ReLU()
        self.norm33 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)


        # fully connected
        self.fc1 = nn.Linear(in_features=128 * 4 * 4,
                             out_features=128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128,
                             out_features=10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        # conv block 1
        outs = self.conv11(x)
        outs = self.relu11(outs)
        outs = self.norm11(outs)
        outs = self.conv12(outs)
        outs = self.relu12(outs)
        outs = self.norm12(outs)
        outs = self.conv13(outs)
        outs = self.relu13(outs)
        outs = self.norm13(outs)
        outs = self.maxpool1(outs)
        res1 = self.conv_shortcut1(outs)
        res1 = self.norm_shortcut1(res1)

        # conv block 2
        outs = self.conv21(outs)
        outs = self.relu21(outs)
        outs = self.norm21(outs)
        outs = self.conv22(outs)
        outs = self.relu22(outs)
        outs = self.norm22(outs)
        outs = self.conv23(outs)
        outs = self.relu23(outs + res1)
        outs = self.norm23(outs)
        outs = self.maxpool2(outs)
        res2 = self.conv_shortcut2(outs)
        res2 = self.norm_shortcut2(res2)

        # conv block 3
        outs = self.conv31(outs)
        outs = self.relu31(outs)
        outs = self.norm31(outs)
        outs = self.conv32(outs)
        outs = self.relu32(outs)
        outs = self.norm32(outs)
        outs = self.conv33(outs)
        outs = self.relu33(outs + res2)
        outs = self.norm33(outs)
        outs = self.maxpool3(outs)


        # fully connected
        outs = self.fc1(torch.flatten(outs, 1))
        outs = self.relu4(outs)
        outs = self.fc2(outs)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
