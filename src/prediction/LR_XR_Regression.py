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

import tensorflow as tf

from prediction.LR_Regression import *


# Import MNIST data

def kl(x, y):
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)

# result = kl(prob_a, prob_b)

class InputGenerator(object):
    """
    InputGenerator is generating (x, s) for GAN
    x: deal attribute, x: price
    """
    def __init__(self, feature, label, doshuffle=True):
        """
        to init generator
        :param feature: input (x, s) : [N, (num_attr+num_pricedim)]
        """
        self.data = feature
        self.label = label
        self.doshuffle = doshuffle

    def shuffle(self, seed = None):
        """
        to shuffle the order of data
        We use this every epoch
        :param seed: random seed

        """

        if seed == None:
            # np.random.seed(seed=int(time.time()))
            np.random.seed(seed=11)
        else:
            np.random.seed(seed)

        id_data = list(range(len(self.data)))
        np.random.shuffle(id_data)
        self.data = self.data[id_data]
        self.label = self.label[id_data]

    def getlength(self):
        """
        to return the size of data
        :return: number of data
        """
        return self.data.shape[0]

    def sample(self, N):
        """
        to sample N samples from data
        :param N:
        :return: [N, (num_attr+num_pricedim)]
        """
        self.shuffle()
        return self.data[:N], self.label[:N]

    def generator(self, batch_size):
        """
        To generator (batch_size) samples for training GAN
        :param batch_size: the number of data for a batch
        :return: return a batch [batch_size, (num_attr+num_pricedim))]
        """
        samples_per_epoch = self.getlength()
        number_of_batches = samples_per_epoch / batch_size
        counter = 0

        while True:

            X_batch = self.data[batch_size * counter:batch_size * (counter + 1), :]
            Y_batch = self.label[batch_size * counter:batch_size * (counter + 1), :]
            counter += 1
            yield (X_batch, Y_batch)

            # restart counter to yeild data in the next epoch as well
            if counter >= number_of_batches:
                counter = 0
                if self.doshuffle:
                    self.shuffle()


class LR_XR(object):
    """
    Logistic Regression with eXpectation Regularization
    """

    def __init__(self, params, featdim=100, classdim=10):
        """
        init lr_xr
        :param params: all parameters for lr_xr
        :param featdim: dim of feature
        :param classdim: dim of class
        """

        # tf Graph Input
        self.x = tf.placeholder(name="x", dtype=tf.float32, shape=[None, featdim])  # mnist data image of shape 28*28=784
        self.y = tf.placeholder(name="y", dtype=tf.float32, shape=[None, classdim])  # 0-9 digits recognition => 10 classes

        # tf intuition input from unlabeled
        self.u_x = tf.placeholder(tf.float32, [None, featdim])  # mnist data image of shape 28*28=784
        self.u_y = tf.placeholder(tf.float32, [None, classdim])  # 0-9 digits recognition => 10 classes

        # Set model weights

        with tf.variable_scope('XR'):
            W = tf.get_variable(
                'w',
                [featdim, classdim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable(
                'b',
                [classdim],
                initializer=tf.constant_initializer(0.0)
            )
            u_W = tf.get_variable(
                'u_w',
                [featdim, classdim],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            u_b = tf.get_variable(
                'u_b',
                [classdim],
                initializer=tf.constant_initializer(0.0)
            )

        # Construct model
        self.prediction = tf.nn.softmax(tf.matmul(self.x, W) + b)  # Softmax
        expected_reg = tf.nn.softmax(tf.multiply(tf.matmul(self.u_x, u_W) + u_b, 1 / classdim))  # Softmax

        # self.loss = - tf.reduce_sum(self.y*tf.log(self.prediction), reduction_indices=1)
        self.reg = kl(self.u_y, expected_reg)
        self.loss = - tf.reduce_sum(self.y * tf.log(self.prediction), reduction_indices=1) + \
                    0.01 * (tf.nn.l2_loss(W) + tf.nn.l2_loss(u_W))  + params.reg_rate * params.batch_size * self.reg
        # Minimize error using cross entropy
        self.cost = tf.reduce_mean(self.loss)
        # Gradient Descent
        self.optimizer = tf.train.AdamOptimizer(params.learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        # Calculate accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def init_LR_XR(args, feat_dim = 1, class_dim = 10, debug = False):
    """
    To build lr_xr

    :param args: intput arguments
    :param feat_dim: dimension of feature
    :param class_dim: dimension of class label
    :param debug: debug option (True: ON)
    :return: init lr_xr using tensorflow build
    """
    tf.reset_default_graph()
    model = LR_XR(args, featdim=feat_dim, classdim=class_dim)
    return model


def fit_and_test_LR_XR_withOutliers(model, args, train_feature, train_label,
                                    test_feature, test_label, intuition_set=([], [], []), debug=False):
    """
    main function for train and test LR_XR
    here we define generators for train and test data

    :param model: built model
    :param args: learning parameters
    :param train_feature: train feature: [N, 36]
    :param train_label: train label: [N, 1]
    :param test_feature: test feature: [M, 36]
    :param test_label: test label: [N, 1]
    :param intuition_set: in the set, [0]-intuition all other feature, [1]-intuition label, [2]-intuition var (THEY ARE FROM OUTLIER)
    :param debug: debug option (True: ON)
    :return:
    """
    train_input= InputGenerator(train_feature, train_label, doshuffle=True)
    test_input = InputGenerator(test_feature, test_label, doshuffle=True)
    final_test_input = InputGenerator(test_feature, test_label, doshuffle=False)
    test_output = train_and_test_withOutliers(model, train_input, test_input, final_test_input, args,
                                              intuition_set=intuition_set, classdim = train_label.shape[1], debug=debug)

    return test_output


def fit_and_test_LR_XR(model, args, train_feature, train_label, test_feature, test_label):

    train_input= InputGenerator(train_feature, train_label, doshuffle=True)
    test_input = InputGenerator(test_feature, test_label, doshuffle=True)
    final_test_input = InputGenerator(test_feature, test_label, doshuffle=False)
    test_output = train_and_test(model, train_input, test_input, final_test_input, args, classdim = train_label.shape[1])

    return test_output


def getProportion_withIntiution(data):
    price = data[:, -1]
    score_intuition = prior_knolwedge_normalized(price)

    np_intuition = np.zeros((len(data), 2), dtype=np.float32)
    for i in range(len(data)):
        if score_intuition[i] == 0.:
            np_intuition[i] = [1 - 0.001, 0.001]
        elif score_intuition[i] == 1.:
                np_intuition[i] = [1 - 0.999, 0.999]
        else:
            np_intuition[i] = [1-score_intuition[i], score_intuition[i]]
    return np_intuition

def train_and_test(model, train_input, test_input, final_test_input, params, classdim=1):


    display_step = 1
    l_test_output = []
    l_acc = []

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        train_sample = train_input.generator(params.batch_size)  # batch generator
        test_sample = test_input.generator(params.batch_size)  # batch generator
        final_test_sample = final_test_input.generator(1)  # batch generator

        # Training cycle
        for epoch in range(params.training_epochs):
            avg_cost = 0.

            # Loop over all batches
            for i in range(train_input.getlength() // params.batch_size):


                (batch_xs, batch_ys) = next(train_sample)
                (batch_u_xs, batch_u_ys) = next(test_sample)

                proportion = getProportion_withIntiution(batch_u_xs) # get intution using prior knowledge function (user input)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, reg, acc = session.run([model.optimizer, model.cost, model.reg, model.accuracy], feed_dict={model.x: batch_xs,
                                                              model.y: batch_ys,
                                                              model.u_x: batch_u_xs,
                                                              model.u_y: proportion})

                avg_cost += c / (train_input.getlength() / params.batch_size)
                l_acc.append(acc)

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "train_avg_acc=", np.mean(l_acc))

        for i in range(final_test_input.getlength()):
            (batch_xs, batch_ys) = next(final_test_sample)
            proportion = getProportion_withIntiution(batch_xs)

            test_output = session.run([model.prediction], feed_dict={model.x: batch_xs, model.y: batch_ys,
                                                                                         model.u_x: batch_xs,
                                                                                         model.u_y: proportion})
            l_test_output.append(test_output)

    return np.reshape(np.array(l_test_output), [final_test_input.getlength(), classdim])



def train_and_test_withOutliers(model, train_input, test_input, final_test_input,
                                params, intuition_set=([], [], []), classdim=1, debug=False):
    """
    train and testing lr_xr using outliers
    :param model: built model
    :param train_input: generator for train data
    :param test_input: generator for test data
    :param final_test_input: another generator to return its output
    :param params: parameters of lr_xr
    :param intuition_set: in the set, [0]-intuition all other feature, [1]-intuition label, [2]-intuition var (THEY ARE FROM OUTLIER)
    :param classdim: dim of class
    :param debug: 0-TRUE
    :return: test results from trained model
    """

    display_step = 1
    l_test_output = []
    l_acc = []

    intuition_price = np.reshape(intuition_set[2], (len(intuition_set[2]), 1))
    intuition_feature_all = np.concatenate([intuition_set[0], intuition_price], axis=-1)
    Intuition_Classifier = LogisticRegression()
    Intuition_Classifier.fit(intuition_feature_all, intuition_set[1])

    # init
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        train_sample = train_input.generator(params.batch_size)  # batch generator
        test_sample = test_input.generator(params.batch_size)  # batch generator
        final_test_sample = final_test_input.generator(1)  # batch generator

        # Training cycle
        for epoch in range(params.training_epochs):
            avg_cost = 0.

            # Loop over all batches
            for i in range(train_input.getlength() / params.batch_size):


                (batch_xs, batch_ys) = next(train_sample)
                (batch_u_xs, batch_u_ys) = next(test_sample)
                # get proportion from LR with intuition only dataset
                intuition = Intuition_Classifier.predict_proba(batch_xs)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, reg, acc = session.run([model.optimizer, model.cost, model.reg, model.accuracy],
                                             feed_dict={model.x: batch_xs,
                                                        model.y: batch_ys,
                                                        model.u_x: batch_u_xs,
                                                        model.u_y: intuition})
                # Compute average loss
                avg_cost += c / (train_input.getlength() / params.batch_size)
                l_acc.append(acc)
            # Display logs per epoch step
            if debug:
                if (epoch+1) % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "train_avg_acc=", np.mean(l_acc))

        # generating test ouputs
        for i in range(final_test_input.getlength()):
            (batch_xs, batch_ys) = next(final_test_sample)
            proportion = getProportion_withIntiution(batch_xs)

            test_output = session.run([model.prediction], feed_dict={model.x: batch_xs, model.y: batch_ys,
                                                                                         model.u_x: batch_xs,
                                                                                         model.u_y: proportion})
            l_test_output.append(test_output)



    return np.reshape(np.array(l_test_output), [final_test_input.getlength(), classdim])



def LR_XR_WinPrediction_withOutliers(args, train_feature, train_label, train_price,
                                     test_feature, test_label, test_price, intuition_set=([], [], []),
                                     op_plot = False,
                                     n_bins = 12, class_dim  =10, debug = False):
    """
    To train and test classifier using outlier
    :param args: parameters for LR_XR
    :param train_feature: [N, 36]
    :param train_label: [N, 1]
    :param train_price: [N, 1]
    :param test_feature:  [M, 36]
    :param test_label:  [M, 1]
    :param test_price:  [M, 1]
    :param intuition_set: in the set, [0]-intuition all other feature, [1]-intuition label, [2]-intuition var (THEY ARE FROM OUTLIER)
    :param op_plot: True - export plot / False - Not
    :param n_bins: number of total bins (only for debugging and plotting)
    :param debug: debug options
    :return: accuracy from testing data
    """

    train_price = np.reshape(train_price, (len(train_price), 1)) # [N, 1]
    test_price = np.reshape(test_price, (len(test_price), 1)) # [M, 1]
    intuition_price = np.reshape(intuition_set[2], (len(intuition_set[2]), 1))

    # feature: (x, s)
    train_feature_all = np.concatenate([train_feature, train_price], axis=-1)
    test_feature_all = np.concatenate([test_feature, test_price], axis=-1)
    intuition_feature_all = np.concatenate([intuition_set[0], intuition_price], axis=-1)

    # categorized labels
    train_label_cat = categorization_feature(train_label, n_bins = 2)
    test_label_cat = categorization_feature(test_label, n_bins = 2)

    # y_hat from (LR_XR)
    model = init_LR_XR(args, feat_dim = train_feature_all.shape[1], class_dim = class_dim)
    prediction = fit_and_test_LR_XR_withOutliers(model, args, train_feature_all, train_label_cat,
                                                 test_feature_all, test_label_cat, intuition_set=intuition_set, debug=debug)

    # plain LR for comparison
    LR_Classifier = LogisticRegression()
    LR_Classifier.fit(train_feature_all, train_label)
    prediction_lr = LR_Classifier.predict_proba(test_feature_all)

    Intuition_Classifier = LogisticRegression()
    Intuition_Classifier.fit(intuition_feature_all, intuition_set[1])
    intuition = Intuition_Classifier.predict_proba(test_feature_all)

    d_price_prob = {}
    d_price_prob_intuition_only = {}
    d_price_prob_lr = {}

    l_output_prob = []

    for i in range(n_bins):
        d_price_prob[i] = []
        d_price_prob_intuition_only[i] = []
        d_price_prob_lr[i] = []

    for i in range(len(test_label)):
        i_price = test_price[i]
        id_price = int(i_price * 10)
        if id_price == 10: id_price = 9  # out-of-bin handling

        y_hat = prediction[i][1] / (prediction[i][0] + prediction[i][1])
        y_prior = intuition[i][1] / (intuition[i][0] + intuition[i][1])
        y_hat_lr = prediction_lr[i][1] / (prediction_lr[i][0] + prediction_lr[i][1])

        l_output_prob.append(y_hat)
        d_price_prob[id_price].append(y_hat)

        # for comparison
        d_price_prob_intuition_only[id_price].append(y_prior)
        d_price_prob_lr[id_price].append(y_hat_lr)


    mean = []
    std = []

    mean_intuition = []
    std_intuition = []

    mean_lr = []
    std_lr = []

    x_range = []

    # bar plot
    # for i in range(n_bins):
    #     if len(d_price_prob[i]) == 0:
    #         mean.append(0)
    #         std.append(0)
    #     else:
    #         mean.append(np.mean(d_price_prob[i]))
    #         std.append(np.std(d_price_prob[i]))
    #     x_range.append(i  * 0.1 + 0.05)
    #
    # if op_plot:
    #     # Call the function to create plot
    #     plt.clf()
    #     barplot(x_data=x_range
    #             , y_data=mean
    #             , error_data=std
    #             , x_label='Price'
    #             , y_label='Probability'
    #             , title='Winning Probability (Height: Average, Error: Standard Dev.)')
    #
    #     plt.xlim(0, 1.0)
    #     plt.plot([0., 1.], [1., 0], 'k-', marker='o', lw=2)  # domain knowledge
    #     plt.savefig("gan_regression_bar_plot_" + str(op_prior) + "_" + str(op_diff) + "_" + str(weight) + ".png")

    # line plot
    for i in range(n_bins):
        if len(d_price_prob[i]) == 0:
            continue
        else:
            mean.append(np.mean(d_price_prob[i]))
            std.append(np.std(d_price_prob[i]))

        if len(d_price_prob_intuition_only[i]) == 0:
            continue
        else:
            mean_intuition.append(np.mean(d_price_prob_intuition_only[i]))
            std_intuition.append(np.std(d_price_prob_intuition_only[i]))

        if len(d_price_prob_lr[i]) == 0:
            continue
        else:
            mean_lr.append(np.mean(d_price_prob_lr[i]))
            std_lr.append(np.std(d_price_prob_lr[i]))

        x_range.append(i * 0.1 + 0.05)


    if op_plot:
        plt.clf()
        plt.plot(x_range, mean, 'm-', marker='o', lw=1, label='LR with Expectation Regularization')
        plt.plot(x_range, mean_lr, 'g-', marker='o', lw=1, label='LR with No Intuition')
        plt.plot(x_range, mean_intuition, 'b-', marker='o', lw=1, label='Intuition Only')
        plt.xlabel('Price')
        plt.ylabel('Winning Probability')
        plt.xlim(0, 1.0)
        plt.plot([0., 1.], [1., 0], 'k-', lw=1, label='Expert\'s Intuition')
        plt.legend(loc='upper right', shadow=True)

        plt.savefig("lr_xr_regression_bar_plot_" + str(op_prior) + "_" + str(op_diff) + "_" + str(weight) + ".png")

    l_output_prediction = []
    for i in range(len(test_label)):

        if l_output_prob[i] > 0.5:
            l_output_prediction.append(1.0)
        else:
            l_output_prediction.append(0.0)

    # Accuracy
    myAccuracy = accuracy_score(test_label, l_output_prediction)

    return myAccuracy


def LR_XR_WinPrediction(args, train_feature, train_label, train_price,
                     test_feature, test_label, test_price, weight = 0.5, op_prior = 0, op_plot = False,
                        op_diff = 0.1, n_bins = 12, class_dim  =10, debug = False):
    """
    To train and test classifier using prior and regression

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

    # categorized labels

    train_label_cat = categorization_feature(train_label, n_bins = 2)
    test_label_cat = categorization_feature(test_label, n_bins = 2)


    # y_hat (LR_XR)
    model = init_LR_XR(args, feat_dim = train_feature_all.shape[1], class_dim = class_dim)
    prediction = fit_and_test_LR_XR(model, args, train_feature_all, train_label_cat, test_feature_all, test_label_cat)

    # LR
    LR_Classifier = LogisticRegression()
    LR_Classifier.fit(train_feature_all, train_label)
    prediction_lr = LR_Classifier.predict_proba(test_feature_all)


    d_price_prob = {}
    d_price_prob_intuition_only = {}
    d_price_prob_lr = {}

    l_output_prob = []

    for i in range(n_bins):
        d_price_prob[i] = []
        d_price_prob_intuition_only[i] = []
        d_price_prob_lr[i] = []

    for i in range(len(test_label)):
        i_price = test_price[i]
        id_price = int(i_price * 10)
        if id_price == 10: id_price = 9  # out-of-bin handling

        y_hat = prediction[i][1] / (prediction[i][0] + prediction[i][1])
        y_prior = prior_knolwedge_normalized(i_price)
        y_hat_lr = prediction_lr[i][1] / (prediction_lr[i][0] + prediction_lr[i][1])

        l_output_prob.append(y_hat)
        d_price_prob[id_price].append(y_hat)

        # for comparison
        d_price_prob_intuition_only[id_price].append(y_prior)
        d_price_prob_lr[id_price].append(y_hat_lr)


    mean = []
    std = []

    mean_intuition = []
    std_intuition = []

    mean_lr = []
    std_lr = []

    x_range = []

    # bar plot
    # for i in range(n_bins):
    #     if len(d_price_prob[i]) == 0:
    #         mean.append(0)
    #         std.append(0)
    #     else:
    #         mean.append(np.mean(d_price_prob[i]))
    #         std.append(np.std(d_price_prob[i]))
    #     x_range.append(i  * 0.1 + 0.05)
    #
    # if op_plot:
    #     # Call the function to create plot
    #     plt.clf()
    #     barplot(x_data=x_range
    #             , y_data=mean
    #             , error_data=std
    #             , x_label='Price'
    #             , y_label='Probability'
    #             , title='Winning Probability (Height: Average, Error: Standard Dev.)')
    #
    #     plt.xlim(0, 1.0)
    #     plt.plot([0., 1.], [1., 0], 'k-', marker='o', lw=2)  # domain knowledge
    #     plt.savefig("gan_regression_bar_plot_" + str(op_prior) + "_" + str(op_diff) + "_" + str(weight) + ".png")

    # line plot
    for i in range(n_bins):
        if len(d_price_prob[i]) == 0:
            continue
        else:
            mean.append(np.mean(d_price_prob[i]))
            std.append(np.std(d_price_prob[i]))

        if len(d_price_prob_intuition_only[i]) == 0:
            continue
        else:
            mean_intuition.append(np.mean(d_price_prob_intuition_only[i]))
            std_intuition.append(np.std(d_price_prob_intuition_only[i]))

        if len(d_price_prob_lr[i]) == 0:
            continue
        else:
            mean_lr.append(np.mean(d_price_prob_lr[i]))
            std_lr.append(np.std(d_price_prob_lr[i]))

        x_range.append(i * 0.1 + 0.05)


    if op_plot:
        plt.clf()
        plt.plot(x_range, mean, 'm-', marker='o', lw=1, label='LR with Expectation Regularization')
        plt.plot(x_range, mean_lr, 'g-', marker='o', lw=1, label='LR with No Intuition')
        # plt.plot(x_range, mean_intuition, 'b-', marker='o', lw=1, label='Intuition Only')
        plt.xlabel('Price')
        plt.ylabel('Winning Probability')
        plt.xlim(0, 1.0)
        plt.plot([0., 1.], [1., 0], 'k-', lw=1, label='Expert\'s Intuition')
        plt.legend(loc='upper right', shadow=True)

        plt.savefig("lr_xr_regression_bar_plot_" + str(op_prior) + "_" + str(op_diff) + "_" + str(weight) + ".png")

    l_output_prediction = []
    for i in range(len(test_label)):

        if l_output_prob[i] > 0.5:
            l_output_prediction.append(1.0)
        else:
            l_output_prediction.append(0.0)

    # Accuracy
    myAccuracy = accuracy_score(test_label, l_output_prediction)

    return myAccuracy