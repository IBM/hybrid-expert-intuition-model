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

def prior_knolwedge_categorized(price, n_bins = 12):
    """
    function of prior knowledge for categorized price
    :param price: input price - this price should already be bucketized. The value is in the range [0, n_bin)
    :param n_bins: number of bins
    :return: expected prob.
    """
    # The winning probably is linearly proportional to the inverse of price
    winning_prob = 1. - (1.) / (n_bins -1) * price # linear function
    return winning_prob

def prior_knolwedge_normalized(intuition, name="CREDIT"):
    """
    function of prior knowledge for normalized price
    :param intuition: input intuition var
    :return: expected prob.
    """
    if name == "CREDIT":
        winning_prob = intuition
    else:
        print("# data mame error!")
        exit()

    return winning_prob


def prior_knolwedge_normalized_v2(price):
    """
    function of prior knowledge for normalized price
    :param price: input price
    :return: expected prob.
    """
    winning_prob = 0.8 - 0.7 * price

    return winning_prob
