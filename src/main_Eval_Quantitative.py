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

import numpy as np
from numpy import genfromtxt

from prediction.GAN_Regression import *
from prediction.LR_XR_Regression import *
import argparse

parser = argparse.ArgumentParser(description='Quantitative Evaluation')
parser.add_argument('--input', type=str, help='ibm', required=True)
parser.add_argument('--export_plot', type=int, help='to export plot', default=0, required=False)
parser.add_argument('--op_classifier', type=int, default= 2, help='0: DDM(Data-driven model), 1: EI, 2: DDM_EI (GAN), 3: LR_XR, '
                    '4: DDM_EI (linear), 5: DDM_EI (functional alpha, GAN), 6: DDM_EI (functional alpha, linear)', required=False)
parser.add_argument('--sigmoid_coeff', type=float, default= 8., help='coefficient for sigmoid', required=False)
parser.add_argument('--prior_weight', type=float, default= 0.5, help='the weight of prior if we use maual alpha (op_classifier=2, 4)', required=False)
parser.add_argument('--confusion', type=float, default= 0.5, help='difference for judging confusion if we use maual alpha (op_classifier=2, 4)', required=False)
parser.add_argument('--num-steps', type=int, default=1000,
                        help='the number of training steps to take')
parser.add_argument('--hidden-size', type=int, default=16,
                    help='MLP hidden size')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--log-every', type=int, default=10,
                    help='print loss after this many steps')
parser.add_argument('--dc_weight', type=float, default=1.1,
                    help='weight in discriminator')
parser.add_argument('--op_valid', type=int, default=1,
                    help='1: from major cluster, 2: from minor cluster, 3: from major + minor clusters')
parser.add_argument('--debug', type=int, default=0,
                    help='0: false, 1: true')
parser.add_argument('--outlier_ratio', type=float, default=0.25,
                    help='0: from major cluster, 1: from minor cluster')
parser.add_argument('--show_trend', type=int, default=0,
                    help='0: no plotting, 1: plotting')

# FOR LR_XR
parser.add_argument('--batch_size', type=int, default= 939, help='batch size for training', required=False)
parser.add_argument('--learning_rate', type=float, default= 0.1, help='learning rate for optimization', required=False)
parser.add_argument('--reg_rate', type=float, default= 100, help='learning rate for optimization', required=False)
parser.add_argument('--training_epochs', type=int, default= 600, help='max_training_size', required=False)

args = parser.parse_args()

print(args)

# get feature
if args.input == "CREDIT":
    dataset = genfromtxt('data/Input_credit.csv', delimiter=',')

    features = dataset[:, :-2]  # we will use 38 (gurantor)
    all_feature = np.hstack([features[:, :38], features[:, 39:]])
    df_label = dataset[:, -1]

    all_label = np.array(df_label, dtype=int)
    all_intuition = features[:, 38]
    all_intuition = np.reshape(all_intuition, (len(all_intuition), 1))
    n_classes = len(set(list(all_label)))

else:
    print ("# dataname error!")
    exit()


# For Output
str_output = ""
l_result = []

# To have data for Intuition
id_major, id_minor = split_data_byClustering(all_feature, ratio=args.outlier_ratio, option=1) # using isolation forest
major_data = all_feature[id_major]
minor_data = all_feature[id_minor]
# random validation
for i in range(10):

    ids_train, ids_test  = split_data(all_feature.shape[0], seed=i, ratio=0.8)
    # set0: train from major, set1, test from minor
    # set2, set3 = split_data(id_minor.shape[0], seed=i, ratio=0.8)  # set2: train from maior, set3, test from minor
    # set2 = id_minor

    if args.show_trend:
        num_repeats = 100
        train_feature = all_feature[ids_train]
        # randomly pick a data point in test data
        test_idx = np.random.randint(len(ids_test), size=1)
        #print(test_idx)
        # create data points using same feature values
        test_feature = np.repeat(all_feature[ids_test[test_idx]], num_repeats, axis=0)

        train_label = all_label[ids_train]
        # create repeated/same test label corresponding to the 'repeated/same' features
        test_label = np.repeat(all_label[ids_test[test_idx]], num_repeats, axis=0)

        train_price = all_intuition[ids_train]

        # how the test prices were generated:
        # using max and min as start and end, evenly spaced to yeild num of data points needed (100 in this experiment)
        test_price = np.linspace(train_price.min(axis=0), train_price.max(axis=0), num=num_repeats)
    else:
        train_feature = all_feature[ids_train]
        test_feature = all_feature[ids_test]

        train_label = all_label[ids_train]
        test_label = all_label[ids_test]

        train_price = all_intuition[ids_train]
        test_price = all_intuition[ids_test]

    if args.op_classifier == 0: # LR with no intuition

        result, _, probs_0 = GAN_WinPrediction_withOutliers(np.array([]),
                                                train_feature, train_label, train_price,
                                                test_feature, test_label, test_price,
                                                weight=args.prior_weight,
                                                op_prior=0, op_plot=args.export_plot,
                                                op_diff=args.confusion,
                                                op_valid=args.op_valid, op_classifier=args.op_classifier,
                                                debug=args.debug)

    elif args.op_classifier == 1:  # Intuition Only

        result, _, probs_1 = GAN_WinPrediction_withOutliers(np.array([]),
                                                train_feature, train_label, train_price,
                                                test_feature, test_label, test_price,
                                                weight=args.prior_weight,
                                                op_prior=3, op_plot=args.export_plot,
                                                op_diff=args.confusion,
                                                op_valid=args.op_valid, op_classifier=args.op_classifier,
                                                debug=args.debug)

    elif args.op_classifier == 2: # GAN_REG_MANUAL_WEIGHTS (ALPHA, DIFF) another GAN that is not as good as 5
        # GAN(s, x) ->s
        test_GAN_price = GANRegression(args, np.concatenate((train_price, train_feature), -1),
                                       np.concatenate((test_price, test_feature), -1),
                                       pricedim=1, debug=args.debug)
        result, _, probs_2 = GAN_WinPrediction_withOutliers(test_GAN_price,
                                  train_feature, train_label, train_price,
                                  test_feature, test_label, test_price,
                                  weight=args.prior_weight,
                                  op_prior=1, op_plot=args.export_plot, op_diff = args.confusion,
                                  op_valid = args.op_valid, op_classifier= args.op_classifier, debug = args.debug)

    elif args.op_classifier == 3:    # LR_XR (Linear Regression with Expectation Regularization)

        result, _, probs_3 = LR_XR_WinPrediction_withOutliers(args,
                                 train_feature, train_label, train_price,
                                 test_feature, test_label, test_price,
                                 class_dim=n_classes,
                                 op_plot=args.export_plot, debug=args.debug)

    elif args.op_classifier == 4: # Linear Regression for price detection

        reg_model = PriceRegression(train_feature, train_price)  # G(x) ->s*
        test_LR_price = reg_model.predict(test_feature)
        test_LR_price = np.reshape(np.array(test_LR_price), (len(test_LR_price), 1))

        result, _, probs_4 = GAN_WinPrediction_withOutliers(test_LR_price,
                                                train_feature, train_label, train_price,
                                                test_feature, test_label, test_price,
                                                weight=args.prior_weight,
                                                op_prior=1, op_plot=args.export_plot,
                                                op_diff=args.confusion,
                                                op_valid=args.op_valid, op_classifier=args.op_classifier,
                                                debug=args.debug)

    elif args.op_classifier == 5: # GAN_REG_FUNCTIONAL_WEIGHTS / Our Method (GAN)
        # GAN(s, x) ->s*
        test_GAN_price = GANRegression(args, np.concatenate((train_price, train_feature), -1),
                                       np.concatenate((test_price, test_feature), -1),
                                       pricedim=1, debug=args.debug)

        result, _, probs_5 = GAN_WinPrediction_difffunc_withOutliers(test_GAN_price,
                                                         train_feature, train_label,
                                                         train_price,
                                                         test_feature, test_label, test_price,
                                                         op_coeff=args.sigmoid_coeff,
                                                         op_plot=args.export_plot,
                                                         op_valid=args.op_valid, debug=args.debug)

    elif args.op_classifier == 6:  # Linear_REG_MANUAL_WEIGHTS (ALPHA, DIFF) / Our Method (Linear Reg)
        reg_model = PriceRegression(train_feature, train_price)  # G(x) ->s*
        test_LR_price = reg_model.predict(test_feature)
        test_LR_price = np.reshape(np.array(test_LR_price), (len(test_LR_price), 1))

        result, _, probs_6 = GAN_WinPrediction_difffunc_withOutliers(test_LR_price,
                                                         train_feature, train_label, train_price,
                                                         test_feature, test_label, test_price,
                                                         op_coeff=args.sigmoid_coeff,
                                                         op_plot=args.export_plot,
                                                         op_valid=args.op_valid, debug=args.debug)



    if args.op_classifier == 0:
        with open('probs/probs_0_{}.txt'.format(i), 'w') as f:
            f.write('\n'.join(map(str, probs_0)))
    elif args.op_classifier == 5:
        with open('probs/probs_5_{}_{}.txt'.format(int(args.sigmoid_coeff), i), 'w') as f:
            f.write('\n'.join(map(str, probs_5)))
    elif args.op_classifier == 2:
        with open('probs/probs_2_{}.txt'.format(i), 'w') as f:
            f.write('\n'.join(map(str, probs_2)))
    l_result.append(result)

# printing results

if args.op_classifier == 5 or args.op_classifier == 6:
    str_output += str(args.op_classifier) + "\t" + str(args.outlier_ratio) + "\t" + "\t" + str(args.sigmoid_coeff) + "\t" +\
                  str(args.confusion) + "\t" + str(args.sigmoid_coeff) + "\t" + str(args.op_valid) + "\t" + \
                  str(np.mean(l_result)) + "\t" + str(np.std(l_result)) + "\t" + str(l_result) + "\n"
else:
    str_output += str(args.op_classifier) + "\t" + str(args.outlier_ratio) + "\t"  + "\t" + str(args.prior_weight) + \
                  "\t" + str(args.confusion) + "\t" + str(args.prior_weight) + "\t" + str(args.op_valid) + "\t" + \
          str(np.mean(l_result)) + "\t" + str(np.std(l_result)) + "\t" + str(l_result) + "\n"

print("###############################################################")
print('##op_classifier\toutlier_ratio\tprior_weight\top_diff\tweight\top_valid\taccuracy\tstd\tall_results')

print(str_output)

