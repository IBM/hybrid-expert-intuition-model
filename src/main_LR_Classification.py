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

import argparse
from prediction.LR_Categorized import *
import numpy as np

parser = argparse.ArgumentParser(description='LR Classification')
parser.add_argument('--input', type=str, help='xlsx input', required=True)
parser.add_argument('--export_plot', type=int, help='to export plot', default=0, required=False)
parser.add_argument('--prior', type=int, default= 0, help='0: no prior, 1: conditional, 2: weighted, 3: prior only', required=False)
parser.add_argument('--weight', type=float, default= 0.5, help='when prior==1, the weight of prior', required=False)
parser.add_argument('--n_bins', type=int, default=10,
                    help='# of bins for categorization')

args = parser.parse_args()

print(args)

# get feature
df_feature, df_label, df_price = getFeature(args.input, op_price=False, op_value=1, n_bins=args.n_bins, debug=False) # op_value = 1 (categorization)
n_bins = args.n_bins + 2 # increased to consider min and max

all_feature = np.array(df_feature) # [4683, 36]
all_price = np.array(df_price, dtype=int) # [4683, 1]
all_label = np.array(df_label, dtype=int) # [4683, 1]

l_result = []

# 10 random cross-validation
for i in range(10):
    id_train, id_test = split_data(len(df_feature), seed=i, ratio=0.8)

    train_feature = all_feature[id_train]
    test_feature = all_feature[id_test]

    train_label = all_label[id_train]
    test_label = all_label[id_test]

    train_price = all_price[id_train]
    test_price = all_price[id_test]

    # 1. train regression model (G(x) ->s*)
    reg_model = PriceRegression(train_feature, train_price)

    # 2. train and test classifier
    result = LR_WinPrediction(reg_model,
                     train_feature, train_label, train_price,
                     test_feature, test_label, test_price, weight=args.weight, n_bins=n_bins,
                              op_prior=args.prior, op_plot=args.export_plot)
    l_result.append(result)

print("###############################################################")
print('##args.prior\tweight\tmean accuracy\tSTD of accuracy')

print("\t" + str(args.prior) + "\t" + str(args.weight) + "\t" + str(np.mean(l_result)) + "\t" + str(np.std(l_result)))



