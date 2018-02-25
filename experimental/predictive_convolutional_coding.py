'''
Tensorflow experimental implementation of the paper
Deep Predictive Coding Network for Object Recognition
https://arxiv.org/pdf/1802.04762.pdf
'''

from observations import mnist
import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, accuracy_score

import datetime

(x_train, y_train), (x_test, y_test) = mnist('./data')

graph = tf.Graph()

convolutional_layer_specs = [
    {
        'filters': [2, 2, 1, 1],
        'strides': [1, 1, 1, 1]
    },
    {
        'filters': [2, 2, 1, 1],
        'strides': [1, 1, 1, 1]
    },
    {
        'filters': [3, 3, 1, 1],
        'strides': [1, 1, 1, 1]
    },
    {
        'filters': [3, 3, 1, 1],
        'strides': [1, 2, 2, 1]
    }
]

class ConvolutionalPredictiveNetwork():

    def __init__(self, input_tensor, batch_size, convolutional_layer_specs, output_shape):

        self.input = input_tensor
        self.batch_size = batch_size
        self.output_shape = output_shape
        self.layer_specs = convolutional_layer_specs

        self.forward_filters = []
        self.feedback_filters = []

        self.forward_biases = []
        self.feedback_biases = []

        self.forward_layers = []
        self.feedback_layers = []

    def variance(self, layer):

        layer_reshape = tf.reshape(layer, shape=[self.batch_size, -1])

        return tf.reshape(
            tf.reduce_mean(tf.square(layer_reshape - tf.reshape(tf.reduce_mean(layer_reshape, axis=1),
                                                                shape=(self.batch_size, 1))), axis=1),
            shape=(self.batch_size, 1, 1, 1)
        )

    def build_forward_network(self):
        with tf.name_scope('input_forward_conv'):
            self.forward_filters.append(
                tf.Variable(
                    tf.truncated_normal(self.layer_specs[0]['filters'], stddev=0.1)
                )
            )
            self.forward_biases.append(
                tf.Variable(
                    tf.truncated_normal([1,
                                         int((int(self.input.shape[1]) - self.layer_specs[0]['filters'][0]) /
                                             self.layer_specs[0]['strides'][1] + 1),
                                         int((int(self.input.shape[2]) - self.layer_specs[0]['filters'][1]) /
                                             self.layer_specs[0]['strides'][2] + 1),
                                         self.layer_specs[0]['filters'][3]], stddev=0.1)
                )
            )
            self.forward_layers.append(
                tf.nn.relu(
                    tf.nn.conv2d(self.input,
                                 self.forward_filters[-1],
                                 strides=self.layer_specs[0]['strides'],
                                 padding='VALID') + self.forward_biases[-1]
                )
            )

        for specs in self.layer_specs[1:]:
            with tf.name_scope('hidden_forward_conv'):
                self.forward_filters.append(
                    tf.Variable(tf.truncated_normal(specs['filters'], stddev=0.1))
                )
                self.forward_biases.append(
                    tf.Variable(
                        tf.truncated_normal([1,
                                             int((int(self.forward_layers[-1].shape[1]) - specs['filters'][0]) /
                                                 specs['strides'][1] + 1),
                                             int((int(self.forward_layers[-1].shape[2]) - specs['filters'][1]) /
                                                 specs['strides'][2] + 1),
                                             specs['filters'][3]], stddev=0.1)
                    )
                )
                self.forward_layers.append(
                    tf.nn.relu(
                        tf.nn.conv2d(self.forward_layers[-1],
                                     self.forward_filters[-1],
                                     strides=specs['strides'],
                                     padding='VALID') + self.forward_biases[-1]
                    )
                )

        with tf.name_scope('output_forward_dense'):

            self.conv_output_size = tf.constant(
                int(self.forward_layers[-1].shape[1] * self.forward_layers[-1].shape[2] * self.forward_layers[-1].shape[3])
            )

            conv_output = tf.reshape(self.forward_layers[-1], [-1, self.conv_output_size])

            self.output_weights = tf.Variable(tf.truncated_normal([self.conv_output_size, self.output_shape], stddev=0.1))
            self.output_biases = tf.Variable(tf.truncated_normal([self.output_shape], stddev=0.1))
            self.output_layer = tf.matmul(conv_output, self.output_weights) + self.output_biases

    def build_feedback_network(self):
        with tf.name_scope('output_prediction'):
            self.output_prediction_weights = tf.Variable(tf.truncated_normal([10, self.conv_output_size], stddev=0.1))
            self.output_prediction_biases = tf.Variable(tf.truncated_normal([self.conv_output_size], stddev=0.1))
            self.output_prediction = tf.matmul(self.output_layer, self.output_prediction_weights) + self.output_prediction_biases

            output_prediction_reshape = tf.reshape(self.output_prediction,
                                                   [self.batch_size,
                                                    int(self.forward_layers[-1].shape[1]),
                                                    int(self.forward_layers[-1].shape[2]),
                                                    int(self.forward_layers[-1].shape[3])])

        with tf.name_scope('updating_forward_conv'):
            self.forward_layers[-1] = tf.nn.relu(
                (tf.constant(1.0) - tf.constant(2.0) * tf.Variable(tf.constant(0.5)) / self.variance(self.forward_layers[-1])) *
                self.forward_layers[-1] + tf.Variable(tf.constant(0.5)) / self.variance(
                    self.forward_layers[-1]) * output_prediction_reshape
            )

        for layer_idx in range(len(self.forward_layers) - 1, 0, -1):

            with tf.name_scope('feedback_conv'):
                self.feedback_filters.append(tf.Variable(
                    tf.truncated_normal(
                        [int(self.forward_filters[layer_idx].shape[0]),
                         int(self.forward_filters[layer_idx].shape[1]),
                         int(self.forward_filters[layer_idx].shape[2]),
                         int(self.forward_filters[layer_idx].shape[3])], stddev=0.1
                    )
                )
                )
                self.feedback_layers.append(
                    tf.nn.conv2d_transpose(self.forward_layers[layer_idx], self.feedback_filters[-1],
                                           output_shape=[self.batch_size,
                                                         int(self.forward_layers[layer_idx - 1].shape[1]),
                                                         int(self.forward_layers[layer_idx - 1].shape[2]),
                                                         int(self.forward_layers[layer_idx - 1].shape[3])],
                                           strides=self.layer_specs[layer_idx]['strides'], padding='VALID')
                )

            with tf.name_scope('updating_forward_conv'):
                self.forward_layers[layer_idx - 1] = tf.reshape(tf.nn.relu(
                    (tf.constant(1.0) - tf.constant(2.0) * tf.Variable(tf.constant(0.5)) / self.variance(
                        self.forward_layers[layer_idx - 1])) *
                    self.forward_layers[layer_idx - 1] + tf.Variable(tf.constant(0.5)) / self.variance(
                        self.forward_layers[layer_idx - 1]) * self.feedback_layers[-1]
                ), [self.batch_size,
                    self.forward_layers[layer_idx - 1].get_shape().as_list()[1],
                    self.forward_layers[layer_idx - 1].get_shape().as_list()[2],
                    self.forward_layers[layer_idx - 1].get_shape().as_list()[3]])

        with tf.name_scope('output_feedback_conv'):
            self.feedback_filters.append(tf.Variable(
                tf.truncated_normal(
                    [int(self.forward_filters[0].shape[0]),
                     int(self.forward_filters[0].shape[1]),
                     int(self.forward_filters[0].shape[2]),
                     int(self.forward_filters[0].shape[3])], stddev=0.1
                )
            )
            )
            self.feedback_layers.append(
                tf.nn.conv2d_transpose(self.forward_layers[0], self.feedback_filters[-1],
                                       output_shape=[self.batch_size,
                                                     int(picture_target.shape[1]),
                                                     int(picture_target.shape[2]),
                                                     int(picture_target.shape[3])],
                                       strides=self.layer_specs[0]['strides'], padding='VALID')
            )

    def update_forward_network(self):

        for layer_idx in range(len(self.forward_layers)):
            with tf.name_scope('error_value'):
                if layer_idx == 0:
                    error = picture_target - self.feedback_layers[-1]
                else:
                    error = self.forward_layers[layer_idx - 1] - self.feedback_layers[-(layer_idx + 1)]

            with tf.name_scope('updating_forward_conv'):
                self.forward_layers[layer_idx] = \
                    tf.nn.relu(
                        self.forward_layers[layer_idx] +
                        (tf.constant(2.0) * tf.Variable(tf.constant(1.0)) / self.variance(self.forward_layers[layer_idx])) *
                        tf.nn.conv2d(error, self.forward_filters[layer_idx],
                                     strides=self.layer_specs[layer_idx]['strides'],
                                     padding='VALID')
                    )

        with tf.name_scope('output_forward_dense'):
            conv_output = tf.reshape(self.forward_layers[-1], [-1, self.conv_output_size])

            self.output_layer = tf.matmul(conv_output, self.output_weights) + self.output_biases

    def update_feedback_network(self):
        with tf.name_scope('output_prediction'):
            self.output_prediction = tf.matmul(self.output_layer,
                                               self.output_prediction_weights) + self.output_prediction_biases

            output_prediction_reshape = tf.reshape(self.output_prediction,
                                                   [self.batch_size,
                                                    int(self.forward_layers[-1].shape[1]),
                                                    int(self.forward_layers[-1].shape[2]),
                                                    int(self.forward_layers[-1].shape[3])])

        with tf.name_scope('updating_forward_conv'):
            self.forward_layers[-1] = tf.nn.relu(
                (tf.constant(1.0) - tf.constant(2.0) * tf.Variable(tf.constant(0.5)) / self.variance(
                    self.forward_layers[-1])) *
                self.forward_layers[-1] + tf.Variable(tf.constant(0.5)) / self.variance(
                    self.forward_layers[-1]) * output_prediction_reshape
            )

        for layer_idx in range(len(self.forward_layers) - 1, 0, -1):
            with tf.name_scope('updating_feedback_conv'):
                self.feedback_layers[len(self.forward_layers) - layer_idx - 1] = \
                    tf.nn.conv2d_transpose(self.forward_layers[layer_idx],
                                           self.feedback_filters[len(self.forward_layers) - layer_idx - 1],
                                           output_shape=[self.batch_size,
                                                         int(self.forward_layers[layer_idx - 1].shape[1]),
                                                         int(self.forward_layers[layer_idx - 1].shape[2]),
                                                         self.forward_layers[layer_idx - 1].get_shape().as_list()[3]],
                                           strides=self.layer_specs[layer_idx]['strides'], padding='VALID')

            with tf.name_scope('updating_forward_conv'):
                self.forward_layers[layer_idx - 1] = tf.nn.relu(
                    (tf.constant(1.0) - tf.constant(2.0) * tf.Variable(tf.constant(0.5)) / self.variance(
                        self.forward_layers[layer_idx - 1])) *
                    self.forward_layers[layer_idx - 1] + tf.Variable(tf.constant(0.5)) / self.variance(
                        self.forward_layers[layer_idx - 1]) * self.feedback_layers[len(self.forward_layers) - layer_idx - 1]
                )

        with tf.name_scope('output_feedback_conv'):
            self.feedback_layers[-1] = \
                tf.nn.conv2d_transpose(self.forward_layers[0], self.feedback_filters[-1],
                                       output_shape=[self.batch_size,
                                                     int(picture_target.shape[1]),
                                                     int(picture_target.shape[2]),
                                                     int(picture_target.shape[3])],
                                       strides=self.layer_specs[0]['strides'], padding='VALID')


with graph.as_default():
    picture = tf.placeholder(tf.float32, [None, 784])
    normalized_picture = picture / 255.0
    picture_target = tf.reshape(normalized_picture, [-1, 28, 28, 1])
    target_label = tf.placeholder(tf.float32, [None, 10])
    batch_size = tf.shape(picture)[0]

    predNet = ConvolutionalPredictiveNetwork(input_tensor=picture_target, batch_size=batch_size,
                                             convolutional_layer_specs=convolutional_layer_specs,
                                             output_shape=10)
    predNet.build_forward_network()
    predNet.build_feedback_network()
    predNet.update_forward_network()
    predNet.update_feedback_network()
    predNet.update_forward_network()

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=target_label, logits=predNet.output_layer)

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    class_prediction = tf.nn.softmax(predNet.output_layer)
    picture_prediction = predNet.feedback_layers[-1]

with tf.Session(graph=graph) as sess:
    summary_writer = tf.summary.FileWriter(logdir='./graphs', graph=graph)

    sess.run(tf.global_variables_initializer())

    accuracy_prev = 0.0
    accuracy_flag = 0

    for epoch in range(150):

        for batch in range(0, 60000, 30):
            sess.run([optimizer], feed_dict={
                picture: x_train[batch:batch + 30],
                target_label: tf.contrib.keras.utils.to_categorical(y_train[batch:batch + 30], num_classes=10)
            })

            if batch % 5000 == 0:
                print(datetime.datetime.now(), 'finished with 5000 samples')

        print('Predicting...')

        print(
            classification_report(y_test,
                                  np.argmax(sess.run([class_prediction],
                                                     feed_dict={picture: x_test})[0], axis=1))
        )

        accuracy = float(accuracy_score(y_test,
                                        np.argmax(sess.run([class_prediction],
                                                           feed_dict={picture: x_test})[0], axis=1)))

        print('The accuracy of the predictive network is', accuracy,
              'and the highest value is', accuracy_prev,
              'and the flag value is', accuracy_flag,
              'in epoch no.', epoch)

        if accuracy < accuracy_prev:
            accuracy_flag += 1

            if accuracy_flag == 20:
                break
        else:
            accuracy_prev = accuracy
            accuracy_flag = 0

        if (epoch + 1) % 25 == 0:
            import matplotlib.pyplot as plt

            plt.imshow(x_train[0].reshape(28, 28))
            plt.show()

            plt.imshow(sess.run([picture_prediction], feed_dict={
                picture: x_train[0].reshape(1, 784),
            })[0].reshape(28, 28))
            plt.show()

            plt.imshow(x_train[1].reshape(28, 28))
            plt.show()

            plt.imshow(sess.run([picture_prediction], feed_dict={
                picture: x_train[1].reshape(1, 784),
            })[0].reshape(28, 28))
            plt.show()

            plt.imshow(x_train[2].reshape(28, 28))
            plt.show()

            plt.imshow(sess.run([picture_prediction], feed_dict={
                picture: x_train[2].reshape(1, 784),
            })[0].reshape(28, 28))
            plt.show()

            plt.imshow(x_train[3].reshape(28, 28))
            plt.show()

            plt.imshow(sess.run([picture_prediction], feed_dict={
                picture: x_train[3].reshape(1, 784),
            })[0].reshape(28, 28))
            plt.show()

            plt.imshow(x_train[4].reshape(28, 28))
            plt.show()

            plt.imshow(sess.run([picture_prediction], feed_dict={
                picture: x_train[4].reshape(1, 784),
            })[0].reshape(28, 28))
            plt.show()

            plt.imshow(x_train[5].reshape(28, 28))
            plt.show()

            plt.imshow(sess.run([picture_prediction], feed_dict={
                picture: x_train[5].reshape(1, 784),
            })[0].reshape(28, 28))
            plt.show()
