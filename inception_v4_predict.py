# -*- coding: utf-8 -*-
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
# from load_cifar10 import load_cifar10_data

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras import optimizers
import keras
import os
import time
import datetime

# from keras.applications.imagenet_utils import decode_predictions

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), bias=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def block_inception_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, subsample=(2,2), border_mode='valid')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_c(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = merge([branch_10, branch_11], mode='concat', concat_axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = merge([branch_20, branch_21], mode='concat', concat_axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def inception_v4_base(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, subsample=(2,2), border_mode='valid')
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, subsample=(2,2), border_mode='valid')

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, border_mode='valid')

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, border_mode='valid')

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, subsample=(2,2), border_mode='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in xrange(4):
      net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in xrange(7):
      net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in xrange(3):
      net = block_inception_c(net)

    return net


def inception_v4_model(img_rows, img_cols, color_type=1, num_classes=None, dropout_keep_prob=0.2):
    '''
    Inception V4 Model for Keras

    Model Schema is based on
    https://github.com/kentsommer/keras-inceptionV4

    ImageNet Pretrained Weights 
    Theano: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5
    TensorFlow: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 299, 299))
    else:
        inputs = Input((299, 299, 3))

    # Make inception base
    net = inception_v4_base(inputs)

    # Final pooling and prediction

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    net_ft = AveragePooling2D((8,8), border_mode='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(output_dim=num_classes, activation='softmax')(net_ft)
    model = Model(inputs, predictions_ft, name='inception_v4')

    weights_path = './models/InceptionV4_model_fold_01_80train_20val_best.h5'     ## train acc = 0.97
    # weights_path = "./models/InceptionV4_model_fold_01_49_089.h5"    ## train acc = 0.89
    model.load_weights(weights_path, by_name=True)

    return model


def img_processing(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    input_image = image.img_to_array(img)

    # input_image /= 255.
    # input_image -= 0.5
    # input_image *= 2.

    input_image = np.expand_dims(input_image, axis=0)
    # input_image = preprocess_input(input_image)

    return input_image

def backup_txt_rename(txt_path):
    if os.path.exists(txt_path):
        i = datetime.datetime.now()
        date = str(i.year) + str(i.month) + str(i.day) + str(i.hour) + str(i.minute) + str(i.second)
        new_name = txt_path +".bak" + date
        os.rename(txt_path, new_name)
        print("copied and deleted file, new_name = {}".format(new_name))


if __name__ == '__main__':

    # dimensions of our images.
    # ADNI GM
    # X: 121*145
    # Y: 121*121
    # Z: 145*121

    # OASIS GM MRI
    # 176*208
    img_width, img_height = 299, 299
    epochs = 60
    batch_size = 32 #10 #40
    channel = 3
    num_classes = 2

    # Load our model
    model = inception_v4_model(img_height, img_width, channel, num_classes, dropout_keep_prob=0.2)
    label = "NC"
    validation_data_dir = os.path.join(".", "validation", label)
    img_list = os.listdir(validation_data_dir)
    ### added by hcq 20180207
    ### prediction, stored in txt named inception_v4_predict_result.txt
    ### img path, [predict result]
    print("=========================")
    inception_v4_predict_path = "./inception_v4_predict_result_" + label + ".txt"
    backup_txt_rename(inception_v4_predict_path)
    with open(inception_v4_predict_path, "a") as predict_file:
        for img in img_list:
            img_path = os.path.join(validation_data_dir, img)
            input_image = img_processing(img_path)
            predictions = model.predict(input_image)
            # print('Predicted:', decode_predictions(predictions, top=2)[0])
            value1 = round(float(predictions[0][0]), 4)
            value2 = round(float(predictions[0][1]), 4)
            print("img_path = {}".format(img_path))
            print("predictions = {}".format(predictions))
            # print("value1 = {}".format(value1))
            # print("value2 = {}".format(value2))
            predict_file.write(img_path + ", [" + str(value1) + ", " + str(value2) + "]\n")

### CUDA_VISIBLE_DEVICES=0 python inception_v4_predict.py
### CUDA_VISIBLE_DEVICES=0 python inception_v4_predict.py > inception_v4_acc89_predict.txt
### CUDA_VISIBLE_DEVICES=0 python single_subject_inception_v4_fold_01.py > single_subject_inception_v4_fold_01_e100_b32.txt
