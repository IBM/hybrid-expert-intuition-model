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

from __future__ import print_function
import random
import numpy as np

def init_theta_true(params, is_linear=True, seed=42):
    # Initialize PRNG Seed
    if seed is not None and type(seed) == int:
        np.random.seed(seed)
    else:
        np.random.seed(None)

    if is_linear:
        # Linear true model (py ∝ exp(θX))
        Theta_true_lin = np.random.randn(params['n'], len(params['d']))
        Theta_true_sq = np.zeros((params['n'], len(params['d'])))
    else:
        # Squared true model (py ∝ exp((θX)^2))
        Theta_true_lin = np.zeros((params['n'], len(params['d'])))
        Theta_true_sq = np.random.randn(params['n'], len(params['d']))

    return Theta_true_lin, Theta_true_sq

def gen_data(m, params, Theta_true_lin, Theta_true_sq, seed=0):
    # Initialize PRNG Seed
    if seed is not None and type(seed) == int:
        np.random.seed(seed)
    else:
        np.random.seed(None)

    X  = np.random.randn(m, params['n'])

    PY = (np.exp(X.dot(Theta_true_lin) + (X.dot(Theta_true_sq)) ** 2)) / np.sum(PY, axis=1)[:, None]
    
    Y  = np.where(np.cumsum(np.random.rand(m)[:, None] < np.cumsum(PY, axis=1), axis=1) == 1)[1]
    Y  = np.eye(len(params['d']))[Y, :]

    return X, Y

if __name__ == '__main__':
    # Initialize True Distribution Model Parameters
    params = {
        'true_model': 'linear' # linear, nonlinear, both
        'c_lin': 10,
        'c_quad': 2.0,
        'b_lin': 30,
        'b_quad': 14,
        'h_lin': 10,
        'h_quad': 2,
        'd': np.array([1, 2, 5, 10, 20]).astype(np.float32),
        'n': 20,
        'theta_true_seed': 42,
        'data_gen_seed': 0,
        'data_split_seed': 1234
    }

    ## Generate X, Y Distribution
    Theta_true_lin, Theta_true_sq = init_theta_true(params,
                                                    is_linear=(true_model == params['true_model']),
                                                    seed=params['theta_true_seed'])
    X, Y = gen_data(1000, params, Theta_true_lin, Theta_true_sq, seed=params['data_gen_seed'])

    # Set Data Split PRNG Seed
    random.seed(params['data_split_seed'])
    np.random.seed(params['data_split_seed'])

    # Repeat Instances of M Times
    for m in range(1000):
        ## Generate Intuition Features
        z_i = random.randint(0, X.shape[1]-1)     # Define Intuition Feature Index
        Z = np.copy(X[:, z_i])                    # Obtain the Intuition Feature

        ## Test Dataset Sample Percentile Parameters
        kp = 20     # Percentage of data for testing (k + p)%
        k = 9       # Get the top kth percentile of the data
        p = kp - k  # Obtain the remaining set of the data from kp

        ## Obtain Indexes
        tk_idx = np.argwhere(Z >= np.percentile(Z, 100-k))
        bk_idx = np.argwhere(Z < np.percentile(Z, 100-k))

        # Generate Train and Test Indicies
        p_idx = np.random.choice(bk_idx[:, 0], size=int(len(Z)*(p/100)), replace=False)
        test_idx = tk_idx[:, 0].tolist() + p_idx.tolist()
        train_idx = [i for i in range(len(Z)) if i not in test_idx]

        # Obtain Split Dataset
        X_train, Y_train, Z_train = X[train_idx], Y[train_idx], Z[train_idx]
        X_test, Y_test, Z_test = X[test_idx], Y[test_idx], Z[test_idx]

        # TODO: Using Given Z and Y Values, Derive Parameters for Expert Intuition Function
