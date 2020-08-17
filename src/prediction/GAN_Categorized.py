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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from prediction.LR_Categorized import *
import time

class InputGenerator(object):
    """
    InputGenerator is generating (x, s) for GAN
    x: deal attribute, x: price
    """
    def __init__(self, feature):
        """
        to init generator
        :param feature: input (x, s) : [N, (num_attr+num_pricedim)]
        """
        self.data = feature

    def shuffle(self, seed = None):
        """
        to shuffle the order of data
        We use this every epoch
        :param seed: random seed

        """

        if seed is None:
            # np.random.seed(seed=int(time.time()))
            np.random.seed(seed=11)
        else:
            np.random.seed(seed)

        id_data = list(range(len(self.data)))
        np.random.shuffle(id_data)
        self.data = self.data[id_data]

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
        return self.data[:N]

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

            X_batch = np.array(self.data[batch_size * counter:batch_size * (counter + 1)]).astype('float32')
            counter += 1
            yield X_batch

            # restart counter to yeild data in the next epoch as well
            if counter >= number_of_batches:
                counter = 0
                self.shuffle()


def linear(input, output_dim, scope=None, stddev=1.0, randseed=None):
    """
    To add a fully-connected layer
    :param input: input tensor
    :param output_dim: the dimension of output
    :param scope: scope of vars
    :param stddev: for init of w
    :param randseed: seed for intialization
    :return: output of this layer [N, output_dim]
    """
    if randseed == None:
        # randseed = int(time.time())
        randseed = 11
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev, seed=randseed)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input, h_dim, pricedim = 1, featdim = 45):
    """
    Generator in GAN (# G(x) -> s*)
    :param input: input vector [N, num of deal attribue + pricedim]
    :param h_dim: num of neurons in the hidden layer of geneerator
    :param pricedim: the number of possible categorized values
    :param featdim: the number ofo deal attributes
    :return: output of generator
    """
    # [price, x] -> to get x by spliting
    price, deal_attr_only = tf.split(input, [pricedim, featdim - pricedim], 1)

    h0 = tf.nn.softplus(linear(deal_attr_only, h_dim, 'g0'))
    h1 = linear(h0, pricedim, 'g1')
    generated_price = tf.nn.softmax(h1)

    # attach again with the new generated price [price*, x]
    output_generator = tf.concat([generated_price, deal_attr_only], 1)

    return output_generator


def discriminator(input, h_dim):
    """
    Discriminator for GAN
    :param input: input of discriminator [N, num of deal attribue + pricedim]
    :param h_dim: # of linear layer's hidden nodes
    :return: output of discrimnator [N, 1]
    """
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim , 'd1'))
    h2 = tf.nn.relu(linear(h1, h_dim/2, 'd2'))
    h3 = tf.nn.relu(linear(h2, 1, scope='d3'))

    return h3

def optimizer(loss, var_list):
    """
    Adam optimizer
    :param loss: loss var
    :param var_list: vars to consider
    :return: optimizer
    """
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    """
    Main class of GAN
    """
    def __init__(self, params, featdim = 1, pricedim = 1):

        with tf.variable_scope('G'):
            # input feature
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, featdim))
            # generated price
            self.G = generator(self.z, params.hidden_size, pricedim=pricedim, featdim=featdim)

        # for test (batch=1)
        with tf.variable_scope('G', reuse=True):
            self.test_z = tf.placeholder(tf.float32, shape=(1, featdim))
            self.G_test = generator(self.test_z, params.hidden_size, pricedim=pricedim, featdim=featdim)

        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, featdim))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size
            )

        # Define the loss for discriminator and generator networks
        self.loss_d = tf.reduce_mean(-params.dc_weight * log(self.D1) + log(self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, train_input, test_input, params, featdim=1, pricedim=1, debug=False):
    """
    To train gan

    :param model: GAN model
    :param train_input: input for training [N, featdim+pricedim]
    :param test_input: input for testing [M, featdim+pricedim]
    :param params: input params
    :param featdim: number of deal attribute features
    :param pricedim: number of price categories
    :param debug: degug option
    :return: generated prices [M, pricedim]
    """
    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        train_sample = train_input.generator(params.batch_size) # batch generator
        test_sample = test_input.generator(1)  # batch generator

        for step in range(params.num_steps + 1):

            # 1. update discriminator
            x = next(train_sample)
            z = x # using same feature for generator and discriimnator
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, featdim)),
                model.z: np.reshape(z, (params.batch_size, featdim))
            })

            # 2. update generator
            z = next(train_sample)
            if len(z) != params.batch_size * featdim:
                print("WARN: training sample z has length ", len(z), "can't be reshaped by ", (params.batch_size, featdim))
                continue

            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, featdim))
            })

            if debug:
                if step % params.log_every == 0:
                    print('{}\t{:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

        if debug:
            dis_1, dis_2 = session.run([model.D1, model.D2], {
                model.x: np.reshape(x, (params.batch_size, featdim)),
                model.z: np.reshape(z, (params.batch_size, featdim))
            })

            print(step, dis_2)

        # for generating prices for testing
        np_test_output = np.empty([0, pricedim])
        for i in range (int(test_input.getlength())):
            z = next(test_sample)
            output = session.run([model.G_test], {
                model.test_z: np.reshape(z, (1, featdim))
            })
            np_test_output = np.concatenate((np_test_output, output[0][:, :pricedim]), axis= 0) # return just price part

        return np_test_output




def GANRegression(args, train_feature, test_feature, pricedim = 1, debug = False):
    """
    To train GAN for regression

    :param args: intput arguments
    :param train_feature: [N, 36]
    :param test_feature: [N, 36]
    :param pricedim: the number of categorized valuees for price
    :param debug: debug option (True: ON)
    :return: testing data's regression output for another classifier
    """
    tf.reset_default_graph()
    # 2. define graph
    model = GAN(args, featdim=(train_feature.shape[1]), pricedim=pricedim)
    # 3. define generator
    train_input= InputGenerator(train_feature)
    test_input = InputGenerator(test_feature) # this is for making output after training (NOT USING FOR TRAINING)

    # 4. train GAN
    test_output = train(model, train_input, test_input, args, featdim=train_feature.shape[1], pricedim=pricedim, debug=debug) # price

    return test_output



def GAN_WinPrediction(test_GAN_price, train_feature, train_label, train_price,
                     test_feature, test_label, test_price, weight = 0.5, op_prior = 0, op_plot = False, op_diff = 4, n_bins = 12, debug = False):
    """
    To train and test classifier using prior and regression
    :param test_GAN_price: regeressed prices
    :param train_feature: [N, 36]
    :param train_label: [N, 1]
    :param train_price: [N, 1]
    :param test_feature:  [M, 36]
    :param test_label:  [M, 1]
    :param test_price:  [M, 1]
    :param weight: weight of prior knowledge
    :param op_prior: 0 - do not use prior, 1 - use it in a hybrid way (our proposal), 2- always use the combined prediction with prior
    :param op_plot: True - export plot / False - Not
    :param op_diff: || s -s* ||_2 for hybrid clssification (if p_prior = 1)
    :param n_bins: number of total bins
    :param debug: debug options
    :return: accuracy from testing data
    """

    # feature: (s)
    train_price_cat = categorization_feature(train_price, n_bins = n_bins)
    test_price_cat = categorization_feature(test_price, n_bins = n_bins)

    # feature: (x, s)
    train_feature_all = np.concatenate([train_feature, train_price_cat], axis=-1)
    test_feature_all = np.concatenate([test_feature, test_price_cat], axis=-1)

    # y_hat
    LR_Classifier = LogisticRegression()
    LR_Classifier.fit(train_feature_all, train_label)

    test_price_star = np.argmax(test_GAN_price, axis=-1)

    diff = abs(test_price - test_price_star)
    prediction = LR_Classifier.predict_proba(test_feature_all)

    if debug:
        plt.hist(diff, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram of ${||s-s^{*}||}^2_2$")
        #plt.show()
        plt.savefig("gan_histrogram(s-s_star).png")

    diff = list(diff)

    d_price_prob = {}
    l_output_prob = []

    for i in range(n_bins):
        d_price_prob[i] = []

    for i in range(len(diff)):
        i_price = test_price[i]

        y_hat = prediction[i][1] / (prediction[i][0] + prediction[i][1])
        y_prior = prior_knolwedge_categorized(i_price)
        y_compromised = (1-weight) * y_hat + weight * y_prior

        if op_prior == 0: # y_hat
            d_price_prob[i_price].append(y_hat)
            l_output_prob.append(y_hat)
        elif op_prior == 2: # just compromised
            d_price_prob[i_price].append(y_compromised)
            l_output_prob.append(y_compromised)
        else: # conditional
            if diff[i] == 0:
                d_price_prob[i_price].append(y_hat)
                l_output_prob.append(y_hat)
            elif diff[i] >= op_diff:
                d_price_prob[i_price].append(y_prior)
                l_output_prob.append(y_prior)
            else:
                d_price_prob[i_price].append(y_compromised)
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
        x_range.append(i)

    if op_plot:
        # Call the function to create plot
        barplot(x_data=x_range
                , y_data=mean
                , error_data=std
                , x_label='Price'
                , y_label='Probability'
                , title='Winning Probability (Height: Average, Error: Standard Dev.)')

        plt.plot([0., 11.], [1., 0], 'k-', lw=2) # domain knowledge
        plt.savefig("gan_bar_plot_" + str(op_prior) + "_" + str(op_diff) + "_" + str(weight) + ".png")

    l_output_prediction = []
    for i in range(len(diff)):

        if l_output_prob[i] > 0.5:
            l_output_prediction.append(1.0)
        else:
            l_output_prediction.append(0.0)

    # Accuracy
    myAccuracy = accuracy_score(test_label, l_output_prediction)

    return myAccuracy
