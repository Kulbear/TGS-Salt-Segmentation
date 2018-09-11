from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

MyActivation = Activation('relu')


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = MyActivation(x)
    return x


def residual_block(block_input, num_filters=16):
    x = MyActivation(block_input)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, block_input])
    return x


# Build model
def build_model(input_layer, start_neurons, dropout_prob=0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding='same')(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = MyActivation(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout_prob / 2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding='same')(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = MyActivation(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout_prob)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding='same')(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = MyActivation(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout_prob)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding='same')(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = MyActivation(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_prob)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding='same')(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16)
    convm = MyActivation(convm)

    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_prob)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding='same')(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = MyActivation(uconv4)

    # 12 -> 25
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding='same')(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding='valid')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_prob)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding='same')(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = MyActivation(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(dropout_prob)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding='same')(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = MyActivation(uconv2)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding='same')(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding='valid')(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(dropout_prob)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding='same')(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = MyActivation(uconv1)

    uconv1 = Dropout(dropout_prob / 1.61803)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(uconv1)

    return output_layer
