# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    """
    get a output from the sigmoid function
    :param x: input
    :return: output
    """
    return ((1 / (1 + np.exp(-1 * x))) - 0.5) * 2
def f2(x):
    """
        get a output from a linear function
        :param x: input
        :return: output
        """
    return x

def f3(x):
    """
        get a output from the sigmoid function
        :param x: input
        :return: output
        """
    return ((1 / (1 + np.exp(-1.5 * x))) - 0.5) * 2

def sigmoid(x, beta=1.0):
    """
        get a output from the sigmoid function
        :param x: input
        :return: output
        """
    return ((1 / (1 + np.exp(-1 * beta * x))) - 0.5) * 2


def f4(x):
    """
        get a output from the x^4
        :param x: input
        :return: output
        """
    return -1* np.power((x-1), 4)  + 1


def plot():
    """
    Plot functions above
    :return: None
    """
    xdomain = np.array(np.linspace(0, 1.0, num=10), dtype=float)

    y_f1 = sigmoid(xdomain, 1)
    y_f2 = sigmoid(xdomain, 2)
    y_f3 = sigmoid(xdomain, 3)
    y_f4 = sigmoid(xdomain, 4)
    y_f5 = sigmoid(xdomain, 8)

    # y_f1 = f1(xdomain)
    # y_f2 = f2(xdomain)
    # y_f3 = f3(xdomain)
    # y_f4 = f4(xdomain)


    # plt.plot(xdomain, area, label='Circle')
    plt.plot(xdomain, y_f1, linestyle='-', color='r', label='sigmoid(1)')
    plt.plot(xdomain, y_f2, linestyle='-', color='b', label='sigmoid(2)')
    plt.plot(xdomain, y_f3, linestyle='-', color='g', label='sigmoid(3)')
    plt.plot(xdomain, y_f4, linestyle='-', color='m', label='sigmoid(4)')
    plt.plot(xdomain, y_f5, linestyle='-', color='c', label='sigmoid(8)')

    plt.ylabel('Weight of Prior knowledge')
    plt.legend()
    plt.show()



# plot()