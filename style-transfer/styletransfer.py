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

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
# print(model)

# content_image = scipy.misc.imread("images/louvre.jpg")
# imshow(content_image)
# style_image = scipy.misc.imread("images/monet_800600.jpg")
# imshow(style_image)
num_iterations=200


def compute_content_cost(a_C, a_G):
#a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
#a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, [-1, n_C])
    a_G_unrolled = tf.reshape(a_G, [-1, n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

    return J_content

#计算风格矩阵
def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
#a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
#a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(a_S, [-1, n_C])
    a_G = tf.reshape(a_G, [-1, n_C])

    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))  # bu yong transpose error

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * (n_C * n_C) * (n_W * n_H) * (n_W * n_W))

    return J_style_layer
#设置各神经层风格损失的权重
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):

    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)  # evaluated

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):

    J = alpha * J_content + beta * J_style
    return J



content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image1 = generate_noise_image(content_image)#随机生成噪声图片
#imshow(generated_image[0])

with tf.Session() as sess:
    # Assign the content image to be the input of the VGG model.
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    J_content = compute_content_cost(a_C, a_G)
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model, STYLE_LAYERS)
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    generated_image = sess.run(model['input'].assign(generated_image1))

    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output/" + str(i) + ".png", generated_image)

    save_image('output/generated_image.jpg', generated_image)
