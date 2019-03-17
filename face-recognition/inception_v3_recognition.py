from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense,BatchNormalization,Lambda
from keras.models import Model
import cv2
import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
'''
next 数据集 训练


# 数据预处理
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)  # process_input (x/255-0.5)*2=(-1,1) zoom 放缩 shear 错切变换_
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(directory='',target_size=(299,299),batch_size=64)
val_generator = val_datagen.flow_from_directory(directory='',target_size=(299,299),batch_size=64)
'''

# 构建模型
base_model = InceptionV3(weights='imagenet',include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,name='dense_layer')(x)
prediction = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
model = Model(inputs=base_model.input,outputs=prediction)
# plot_model(model,'tmodel.png')

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)
    basic_loss = tf.add(pos_dist - neg_dist, alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    return loss

model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])

# 系统存储人员的encoding
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", model)
database["younes"] = img_to_encoding("images/younes.jpg", model)
database["tian"] = img_to_encoding("images/tian.jpg", model)
database["andrew"] = img_to_encoding("images/andrew.jpg", model)
database["kian"] = img_to_encoding("images/kian.jpg", model)
database["dan"] = img_to_encoding("images/dan.jpg", model)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", model)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", model)
database["kevin"] = img_to_encoding("images/kevin.jpg", model)
database["felix"] = img_to_encoding("images/felix.jpg", model)
database["benoit"] = img_to_encoding("images/benoit.jpg", model)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", model)
# 人脸验证
def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])  # ord：范数类型,默认为2
    # Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    return dist, door_open
# 人脸识别
def recognition(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity

