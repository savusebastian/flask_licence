from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Dropout, Dense, BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, concatenate
from tensorflow.keras.models import Model
import resnet


# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0.):
    model = Dropout(drop_rate)(inp_layer)
    model = Dense(n)(model)
    model = BatchNormalization(axis=-1)(model)
    model = Activation(activation=activation)(model)
    return model


# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3), strides=(1, 1), zp_flag=False):
    if zp_flag:
        model = ZeroPadding2D((1, 1))(feature_batch)
    else:
        model = feature_batch
        
    model = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(model)
    model = BatchNormalization(axis=3)(model)
    model = LeakyReLU(1 / 10)(model)
    return model


def my_model_1(width=200, height=200, depth=3, classes=9):
    inp_img = Input(shape=(width, height, depth))

    # 51
    model = conv_layer(inp_img, 64, zp_flag=False)
    model = conv_layer(model, 64, zp_flag=False)
    model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model)
    # 23
    model = conv_layer(model, 128, zp_flag=False)
    model = conv_layer(model, 128, zp_flag=False)
    model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model)
    # 9
    model = conv_layer(model, 256, zp_flag=False)
    model = conv_layer(model, 256, zp_flag=False)
    model = conv_layer(model, 256, zp_flag=False)
    model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model)
    # 1
    # dense layers
    model = Flatten()(model)
    model = dense_set(model, 128, activation='tanh')
    out = dense_set(model, classes, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)

    return model


def my_model_2(width=200, height=200, depth=3, classes=9):
    # initialize the model

    inp_img = Input(shape=(width, height, depth))

    # first set of CONV => RELU => POOL layers
    model = Conv2D(32, (3, 3), padding='same', activation='relu')(inp_img)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    model = Conv2D(32, (3, 3), padding='same', activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(0.2)(model)

    # first (and only) set of FC => RELU layers
    model = Flatten()(model)
    model = Dense(128, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.2)(model)
    # softmax classifier
    out = Dense(classes)(model)
    model = Model(inputs=inp_img, outputs=out)
    # return the constructed network architecture
    return model


def my_model_3(width=200, height=200, depth=3, classes=9):
    inp_img = Input(shape=(width, height, depth))
    model = Conv2D(16, (3, 3), activation='relu')(inp_img)
    model = MaxPooling2D((2, 2))(model)
    model = Conv2D(32, (3, 3), activation='relu')(model)
    model = MaxPooling2D((2, 2))(model)
    model = Conv2D(32, (3, 3), activation='relu')(model)
    model = MaxPooling2D((2, 2))(model)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D((2, 2))(model)
    model = Flatten()(model)
    model = Dropout(0.5)(model)
    model = Dense(128, activation='relu')(model)
    out = Dense(classes, activation='softmax')(model)

    model = Model(inputs=inp_img, outputs=out)
    return model


def my_model_4(width=200, height=200, depth=3, classes=9):
    inp_img = Input(shape=(width, height, depth))

    tower_one = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inp_img)
    tower_one = Conv2D(16, (1, 1), activation='relu', border_mode='same')(tower_one)

    tower_two = Conv2D(16, (1, 1), activation='relu', border_mode='same')(inp_img)
    tower_two = Conv2D(16, (3, 3), activation='relu', border_mode='same')(tower_two)

    tower_three = Conv2D(16, (1, 1), activation='relu', border_mode='same')(inp_img)
    tower_three = Conv2D(16, (5, 5), activation='relu', border_mode='same')(tower_three)

    output = concatenate([tower_one, tower_two, tower_three], axis=3)
    output = Flatten()(output)
    output = Dense(64, activation='relu')(output)
    predictions = Dense(classes, activation='softmax')(output)

    model = Model(input=inp_img, outputs=predictions)
    return mode


def my_model_5(width=200, height=200, depth=3, classes=9):
#    kernel_size = (3, 3)
#    pool_size = (2, 2)
#    first_filters = 32
#    second_filters = 64
#    third_filters = 128

#    dropout_conv = 0.3
#    dropout_dense = 0.3

#    inp_img = Input(shape=(width, height, depth))

#    model = Conv2D(first_filters, kernel_size, activation='relu')(inp_img)

#    model = Conv2D(first_filters, kernel_size, activation='relu')(model)
#    model = Conv2D(first_filters, kernel_size, activation='relu')(model)
#    model = MaxPooling2D(pool_size=pool_size)(model)
#    model = Dropout(dropout_conv)(model)

#    model = Conv2D(second_filters, kernel_size, activation='relu')(model)
#    model = Conv2D(second_filters, kernel_size, activation='relu')(model)
#    model = MaxPooling2D(pool_size=pool_size)(model)
#    model = Dropout(dropout_conv)(model)

#    model = Conv2D(third_filters, kernel_size, activation='relu')(model)
#    model = Conv2D(third_filters, kernel_size, activation='relu')(model)
#    model = Conv2D(third_filters, kernel_size, activation='relu')(model)
#    model = MaxPooling2D(pool_size=pool_size)(model)
#    model = Dropout(dropout_conv)(model)

#    model = Flatten()(model)
#    model = Dense(256, activation='relu')(model)
#    model = Dropout(dropout_dense)(model)

    # Pentru 64x64
    inp_img = Input(shape=(width, height, depth))

    model = Conv2D(32, (3, 3), activation='relu')(inp_img)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Flatten()(model)
    model = Dropout(0.5)(model)
    model = Dense(512, activation='relu')(model)
    model = Dense(1, activation='sigmoid')(model)

    out = Dense(classes, activation='softmax')(model)

    model = Model(inputs=inp_img, outputs=out)
    return model

    # Pentru 32x32
    # inp_img = Input(shape=(width, height, depth))
    #
    # model = Conv2D(16, (3, 3), activation='relu')(inp_img)
    # model = MaxPooling2D(pool_size=(2, 2))(model)
    # model = Conv2D(32, (3, 3), activation='relu')(model)
    # model = MaxPooling2D(pool_size=(2, 2))(model)
    # model = Conv2D(64, (3, 3), activation='relu')(model)
    # model = MaxPooling2D(pool_size=(2, 2))(model)
    # model = Conv2D(64, (3, 3), activation='relu')(model)
    # model = MaxPooling2D(pool_size=(2, 2))(model)
    # model = Flatten()(model)
    # model = Dropout(0.5)(model)
    # model = Dense(512, activation='relu')(model)
    # model = Dense(1, activation='sigmoid')(model)
    #
    # out = Dense(classes, activation='softmax')(model)
    #
    # model = Model(inputs=inp_img, outputs=out)
    # return model


def my_model_6(width=200, height=200, depth=3, classes=9):
    inp_img = Input(shape=(width, height, depth))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inp_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    out = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=inp_img, outputs=out)
    return model

def my_model_7(width=200, height=200, depth=3, classes=9):
    return  resnet.ResnetBuilder.build_resnet_50((depth, height, width), classes)

def get_model(index=0, width=200, height=200, depth=3, classes=9):
    if index == 0:
        return my_model_1(width, height, depth, classes)
    elif index == 1:
        return my_model_2(width, height, depth, classes)
    elif index == 2:
        return my_model_3(width, height, depth, classes)
    elif index == 3:
        return my_model_4(width, height, depth, classes)
    elif index == 4:
        return my_model_5(width, height, depth, classes)
    elif index == 5:
        return my_model_6(width, height, depth, classes)
    elif index == 6:
        return my_model_7(width, height, depth, classes)
