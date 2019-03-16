import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
from styletransfer1 import *


num_iterations = 200
STYLE_LAYERS = [('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)]  # 设置各神经层风格损失的权重
CONTENT_LAYERS = [('conv4-1',1.0)]


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    size = n_H * n_W * n_C
    a_C_unrolled = tf.reshape(a_C, [-1, n_C])
    a_G_unrolled = tf.reshape(a_G, [-1, n_C])
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / size
    return J_content

# 计算风格矩阵
# ----------------------------------
def gram_matrix(A):
    GA = tf.matmul(tf.transpose(A),A)
    return GA

def compute_layer_style_cost(a_S, a_G):  # a_S-(1, n_H, n_W, n_C)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    size = n_H * n_W * n_C
    a_S = tf.reshape(a_S, [-1, n_C])
    a_G = tf.reshape(a_G, [-1, n_C])
    GS = gram_matrix(a_S)/size
    GG = gram_matrix(a_G)/size
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / size
    return J_style_layer

def total_cost(content_input,style_input,target,alpha=10, beta=1):
    J_style = 0
    a_C = vgg19(content_input)
    a_S = vgg19(style_input)
    a_G = vgg19(target)
    for layer_name, coeff in STYLE_LAYERS:
        J_style_layer = compute_layer_style_cost(a_S[layer_name], a_G[layer_name])
        J_style += coeff * J_style_layer
    content_layer = 'conv4_2'
    J_content = compute_content_cost(a_C[content_layer], a_G[content_layer])
    J = alpha * J_content + beta * J_style
    return J,J_content,J_style

def generated(content_image,style_image):
    first_image = generate_noise_image(content_image)

    content_input = tf.constant(content_image,dtype=tf.float32)
    style_input = tf.constant(style_image,dtype=tf.float32)
    target = tf.Variable(first_image,dtype=tf.float32)

    J,J_content,J_style = total_cost(content_input,style_input,target,alpha=10, beta=1)
    train_step = tf.train.AdamOptimizer(0.1).minimize(J)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iterations):
            _,loss,generated_image = sess.run([train_step,J,target])
            print('epoch:%d loss:%f' % (i,loss))
            if i % 20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                save_image("generated_images/" + str(i) + ".png", generated_image)
        save_image('generated_images/generated_image.jpg', generated_image)


if __name__ == '__main__':
    content_image = scipy.misc.imread("content_images/content3.jpg")
    content_image = reshape_and_normalize_image(content_image)
    style_image = scipy.misc.imread("style_images/style5.jpg")
    style_image = reshape_and_normalize_image(style_image)
    generated(content_image,style_image)