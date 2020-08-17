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

for i in range(1, 10):
  with open('probs/probs_0_{}.txt'.format(i), 'r') as f:
    probs_0 = list(map(float, f.readlines()))

  with open('probs/probs_2_{}.txt'.format(i), 'r') as f:
    probs_2 = list(map(float, f.readlines()))

  probs_5 = dict()

  # only 1 iteration, for reading in 'probs_5_8_{}.txt'?
  for coeff in [4, 8]:
    with open('probs/probs_5_{}_{}.txt'.format(coeff, i), 'r') as f:
      probs_5[coeff] = list(map(float, f.readlines()))

  x = np.array(range(len(probs_5[8]))) # x = 100
  # np.linspace(left, right, number_of_returned_value): Returns number spaces evenly w.r.t interval
  # here len(prob_0) = 939 (which is some predefined batch size)
  probs_1 = np.linspace(1, 0, len(probs_0))
  print('len(probs_0) = ', len(probs_0)) # 939
  print('len(probs_5[8]) = ', len(probs_5[8])) # 100

  plt.cla()
  plt.ylim(0, 1.0)
  plt.xlim(0, 100)
  # plt.plot(x, probs_0, 'r', label='LR (Data Driven)')
  plt.plot(x, probs_1, 'k', label='Intuition Only')
  # plt.plot(x, probs_2, 'g', label='Gan Framework, constant weights')
  plt.plot(x, probs_5[8], 'b', label='coeff = 8')
  #plt.plot(x, probs_5[2], 'r', label='coeff = 2')
  plt.plot(x, probs_5[4], 'g', label='coeff = 4')
  #plt.plot(x, probs_5[8], 'k', label='coeff = 8')
  plt.legend()
  plt.savefig('{}_1.png'.format(i))