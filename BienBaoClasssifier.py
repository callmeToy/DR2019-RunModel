#!/usr/bin/python3
import rospy,sys,os
try:
	os.chdir(os.path.dirname(__file__))	
	os.system('clear')
	print("\nWait for initial setup, please don't connect anything yet...\n")
	sys.path.remove('/opt/ros/lunar/lib/python2.7/dist-packages')
except: pass
import tensorflow as tf
import cv2
import numpy as np
import os
import getpass

class BienBaoClassifier(object):
    def __init__(self, model_dir = 'model1'):
        self.model_dir = model_dir
        self.weights = {
            'wc1': tf.get_variable('W0', shape=(5, 5, 3, 32), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('W1', shape=(5, 5, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W3', shape=(7 * 7 * 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(64, 3), initializer=tf.contrib.layers.xavier_initializer()),
        }
        self.biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('B4', shape=(3), initializer=tf.contrib.layers.xavier_initializer()),
        }
        self.x = tf.placeholder("float", [None, 28, 28, 3])
        self.y = tf.placeholder("float", [None, 3])
        self.pred = conv_net(self.x, self.weights, self.biases)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        # ssaver.restore(self.sess, "/home/toy/catkin_ws/src/team500/scripts/model1/model.ckpt")
        saver.restore(self.sess, "/home/"+ getpass.getuser()+"/catkin_ws/src/team500/scripts/model1/model.ckpt")

    def detect(self, image):
        image = cv2.resize(image, (28, 28))
        return np.argmax(self.sess.run([self.pred], feed_dict={self.x: np.reshape(image, (1, 28, 28, 3))}))


def conv_net(x, weights, biases, Train=[0]):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    if Train[-1] == 1:
        fc1 = tf.nn.dropout(fc1, 0.45)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def get_image(path, labels_name=['ReTrai', 'RePhai', 'Other']):
    folder = os.listdir(path)
    datas = []
    labels = []
    for fol in folder:
        if len(fol.split('.')) == 1:
            if fol in labels_name:
                Image = os.listdir(path + '/' + fol)
                index = labels_name.index(fol)
                for filImage in Image:
                    if filImage.split('.')[-1] in ['jpg', 'png']:
                        image = cv2.imread(path + '/' + fol + '/' + filImage)
                        image = cv2.resize(image, (28, 28))
                        image_flat = np.reshape(image, (28 * 28 * 3))
                        datas.append(image_flat)
                        labels.append(index)
                        cv2.imshow('image', image)
                        cv2.waitKey(10)
    np.save(path + '/' + 'Datas_Train.npy', datas)
    np.save(path + '/' + 'Labelss_Train.npy', labels)
    cv2.destroyAllWindows()
    return np.asarray(datas, dtype=np.float32), np.asarray(labels, dtype=np.int32)


if __name__ == '__main__':
    BienBaoClassifier()
