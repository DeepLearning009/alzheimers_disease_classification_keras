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

from keras import optimizers
import keras

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

    # 8 x 8 x 1536
    net_old = AveragePooling2D((8,8), border_mode='valid')(net)

    # 1 x 1 x 1536
    net_old = Dropout(dropout_keep_prob)(net_old)
    net_old = Flatten()(net_old)

    # 1536
    predictions = Dense(output_dim=1001, activation='softmax')(net_old)

    model = Model(inputs, predictions, name='inception_v4')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/inception-v4_weights_th_dim_ordering_th_kernels.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'

    # weights_path = './InceptionV4_model_fold_01.h5'
    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    net_ft = AveragePooling2D((8,8), border_mode='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(output_dim=num_classes, activation='softmax')(net_ft)

    model = Model(inputs, predictions_ft, name='inception_v4')

    # Learning rate is changed to 0.001
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    # dimensions of our images.
    # ADNI GM
    # X: 121*145
    # Y: 121*121
    # Z: 145*121

    # OASIS GM MRI
    # 176*208
    ### data_fold_01_train_val_test_entropy_keep_SliceNum_33
    img_width, img_height = 299, 299
    fold_name = "fold_01"  ## data_fold_01_entropy_keep_SliceNum_33
    train_data_dir = 'data_' + fold_name + '_train_val_test_entropy_keep_SliceNum_33/train'
    validation_data_dir = 'data_' + fold_name + '_train_val_test_entropy_keep_SliceNum_33/validation'
    filepath="model_InceptionV4_" + fold_name + "_train_val_test_entropy_keep_SliceNum_33_best.h5"

    # train num (AD+NC) = 14776 + 17077 = 31853
    # validation num (AD+NC) = 3941 + 4555 = 8496
    # test num (AD+NC) = 985 + 1138 = 2123
    # train_samples_AD =  len(os.listdir(path))  
    # train_samples_NC =  # 18216
    # val_samples_AD =  # 3941
    # val_samples_NC =  # 4554
    nb_train_samples = 31853
    nb_validation_samples = 8496
    epochs = 120
    batch_size = 32 #10 #40
    channel = 3
    num_classes = 2
    
    print("=== paramaters info ===")
    print("epochs = {}.".format(epochs))
    print("batch_size = {}.".format(batch_size))
    print("nb_train_samples = {}.".format(nb_train_samples))
    print("nb_validation_samples = {}.".format(nb_validation_samples))

    #if K.image_data_format() == 'channels_first':
    #	input_shape = (3, img_width, img_height)
    #else:
    #	input_shape = (img_width, img_height, 3)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)



    ### class_mode: "categorical", "binary", "sparse"或None之一.
    ### 默认为"categorical: 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,
    ### "binary"返回1D的二值标签.
    ### "sparse"返回1D的整数标签,
    ### 如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # Load our model
    model = inception_v4_model(img_height, img_width, channel, num_classes, dropout_keep_prob=0.2)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath = filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        # save_weights_only=False,
        mode='max',
        period=1
    )
    callbacks_list = [checkpoint]


    ### verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    ### 
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples/batch_size,
        epochs=epochs,
        verbose = 2,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples/batch_size,
        callbacks = callbacks_list)
        #validation_steps=nb_validation_samples // batch_size)


    # model.save('InceptionV4_model_fold_01.h5')


    # Make predictions
    #predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    #score = log_loss(Y_valid, predictions_valid)

### CUDA_VISIBLE_DEVICES=1 python inception_v4_train_val_test_entropy_keep_SliceNum_33_fold_01.py > acc_inception_v4_train_val_test_entropy_keep_SliceNum_33_fold_01.txt
