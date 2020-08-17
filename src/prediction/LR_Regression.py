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

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import auc, accuracy_score

from preprocessing.feature_processing import *
from prediction.prior_knowledge import *
from util.error_bar_plot import *

def LR_WinPrediction(reg_model, train_feature, train_label, train_price,
                     test_feature, test_label, test_price, weight = 0.5, op_prior = 0, op_plot = False, op_diff = 0.1, n_bins = 12, debug = False):
    """
    To train and test classifier using prior and regression

    :param reg_model: trained regression model
    :param train_feature: [N, 36]
    :param train_label: [N, 1]
    :param train_price: [N, 1]
    :param test_feature:  [M, 36]
    :param test_label:  [M, 1]
    :param test_price:  [M, 1]
    :param weight: weight of prior knowledge
    :param op_prior: 0 - do not use prior, 1 - use it in a hybrid way (our proposal), 2- always use the combined prediction with prior, , 3- prior only
    :param op_plot: True - export plot / False - Not
    :param op_diff: || s -s* ||_2 for hybrid clssification (if p_prior = 1)
    :param n_bins: number of total bins (only for debugging and plotting)
    :param debug: debug options
    :return: accuracy from testing data
    """

    train_price = np.reshape(train_price, (len(train_price), 1)) # [N, 1]
    test_price = np.reshape(test_price, (len(test_price), 1)) # [M, 1]

    # feature: (x, s)
    train_feature_all = np.concatenate([train_feature, train_price], axis=-1)
    test_feature_all = np.concatenate([test_feature, test_price], axis=-1)

    # y_hat
    LR_Classifier = LogisticRegression()
    LR_Classifier.fit(train_feature_all, train_label)

    test_price_star = reg_model.predict(test_feature)
    test_price_star = np.reshape(np.array(test_price_star), (len(test_price_star), 1))

    diff = abs(round_decimal(test_price) - round_decimal(test_price_star))
    prediction = LR_Classifier.predict_proba(test_feature_all)

    if debug:
        plt.clf()
        plt.hist(diff, bins=np.linspace(0, 1.0, num=40))  # arguments are passed to np.histogram
        plt.xlim(0, 1.0)
        plt.title("Histogram of ${||s-s^{*}||}$")
        # plt.show()
        plt.savefig("lr_linear_regression_histrogram(s-s_star).png")


    diff = list(diff)

    d_price_prob = {} # for degugging and exporting plot
    l_output_prob = []

    for i in range(n_bins):
        d_price_prob[i] = []

    for i in range(len(diff)):
        i_price = test_price[i]
        id_price = int(i_price * 10)
        if id_price == 10: id_price = 9  # out-of-bin handling

        y_hat = prediction[i][1] / (prediction[i][0] + prediction[i][1])
        y_prior = prior_knolwedge_normalized(i_price)
        y_compromised = (1 - weight) * y_hat + weight * y_prior

        if op_prior == 0:  # y_hat
            d_price_prob[id_price].append(y_hat)
            l_output_prob.append(y_hat)
        elif op_prior == 2:  # just compromised
            d_price_prob[id_price].append(y_compromised)
            l_output_prob.append(y_compromised)
        elif op_prior == 3:  # prior only
            d_price_prob[id_price].append(y_prior)
            l_output_prob.append(y_prior)
        else:  # conditional
            if diff[i] == 0:
                d_price_prob[id_price].append(y_hat)
                l_output_prob.append(y_hat)
            elif diff[i] >= op_diff:
                d_price_prob[id_price].append(y_prior)
                l_output_prob.append(y_prior)
            else:
                d_price_prob[id_price].append(y_compromised)
                l_output_prob.append(y_compromised)

    mean = []
    std = []
    x_range = []

    for i in range(n_bins):
        if len(d_price_prob[i]) == 0:
            mean.append(0)
            std.append(0)
        else:
            mean.append(np.mean(d_price_prob[i]))
            std.append(np.std(d_price_prob[i]))
        x_range.append(i * 0.1 + 0.05)

    if op_plot:
        # Call the function to create plot
        plt.clf()

        barplot(x_data=x_range
                , y_data=mean
                , error_data=std
                , x_label='Price'
                , y_label='Probability'
                , title='Winning Probability (Height: Average, Error: Standard Dev.)')
        plt.xlim(0, 1.0)
        plt.plot([0., 1.], [1., 0], 'k-', lw=2)  # domain knowledge
        plt.savefig("lr_regression_bar_plot_" + str(op_prior) + "_" + str(op_diff) + "_" + str(weight) + ".png")

    l_output_prediction = []
    for i in range(len(diff)):

        if l_output_prob[i] > 0.5:
            l_output_prediction.append(1.0)
        else:
            l_output_prediction.append(0.0)

    # Accuracy
    myAccuracy = accuracy_score(test_label, l_output_prediction)

    return myAccuracy


def PriceRegression(train_feature, train_price):
    """
    To train logistic regression
    :param train_feature: [N, 36]
    :param train_price: [N, 1]
    :return: regression model
    """
    LogReg = LinearRegression()
    LogReg.fit(train_feature, train_price)
    # y_score = LogReg.predict(X_test)
    # mse_score = np.mean((y_score - y_test) ** 2)
    # print "# reg_training mse:", mse_score

    return LogReg

