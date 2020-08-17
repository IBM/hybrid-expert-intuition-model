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
from prediction.LR_XR_Regression import *
import numpy as np

parser = argparse.ArgumentParser(description='LR XR Clasisfication')
parser.add_argument('--input', type=str, help='xlsx input', required=True)
parser.add_argument('--export_plot', type=int, help='to export plot', default=0, required=False)
parser.add_argument('--batch_size', type=int, default=939, help='batch size for training', required=False)
parser.add_argument('--learning_rate', type=float, default= 0.1, help='learning rate for optimization', required=False)
parser.add_argument('--reg_rate', type=float, default= 100, help='learning rate for optimization', required=False)
parser.add_argument('--training_epochs', type=int, default= 600, help='max_training_size', required=False)
parser.add_argument('--debug', type=bool, default= True, help='debug option (on or off)', required=False)


args = parser.parse_args()

print(args)

# get feature
df_feature, df_label, df_price = getFeature(args.input, op_price=False, op_value = 2) # op_value = 2 (normalized)

all_feature = np.array(df_feature) # [4683, 36]
all_label = np.array(df_label, dtype=int) # [4683, 1]
all_price = np.array(df_price) # [4683, 1]
n_classes = len(set(list(all_label)))

print ("# classes", n_classes)
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

    # 1. train and test classifier
    result = LR_XR_WinPrediction(args,
                     train_feature, train_label, train_price,
                     test_feature, test_label, test_price, class_dim = n_classes, op_plot = args.export_plot)
    l_result.append(result)

print("###############################################################")
print('##lr_classification\taccuracy')

print("\t" +  str(np.mean(l_result)) + "\t" + str(np.std(l_result)))



