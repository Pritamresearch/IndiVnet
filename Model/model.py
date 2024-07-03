from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


#backbone

encoder_ResNet50 = ResNet50(input_shape =  (256, 256,3), include_top = False, weights = 'imagenet')
encoder_ResNet50.trainable = False

#indienco encoder

def indienco(inputs):
    shape = inputs.shape

    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6in = Concatenate()([inputs, y_1])
    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(y_6in)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_7in = Concatenate()([inputs, y_6])
    y_7 = Conv2D(filters=256, kernel_size=3, dilation_rate=7, padding='same', use_bias=False)(y_7in)
    y_7 = BatchNormalization()(y_7)
    y_7 = Activation('relu')(y_7)

    y_8in = Concatenate()([inputs, y_7])
    y_8 = Conv2D(filters=256, kernel_size=3, dilation_rate=8, padding='same', use_bias=False)(y_8in)
    y_8 = BatchNormalization()(y_8)
    y_8 = Activation('relu')(y_8)

    y_9in = Concatenate()([inputs, y_8])
    y_9 = Conv2D(filters=256, kernel_size=3, dilation_rate=9, padding='same', use_bias=False)(y_9in)
    y_9 = BatchNormalization()(y_9)
    y_9 = Activation('relu')(y_9)

    y_10in = Concatenate()([inputs, y_9])
    y_10 = Conv2D(filters=256, kernel_size=3, dilation_rate=10, padding='same', use_bias=False)(y_10in)
    y_10 = BatchNormalization()(y_10)
    y_10 = Activation('relu')(y_10)

    y_11in = Concatenate()([inputs, y_10])
    y_11 = Conv2D(filters=256, kernel_size=3, dilation_rate=11, padding='same', use_bias=False)(y_11in)
    y_11 = BatchNormalization()(y_11)
    y_11 = Activation('relu')(y_11)

    y_12in = Concatenate()([inputs, y_11])
    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(y_12in)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_13in = Concatenate()([inputs, y_12])
    y_13 = Conv2D(filters=256, kernel_size=3, dilation_rate=13, padding='same', use_bias=False)(y_13in)
    y_13 = BatchNormalization()(y_13)
    y_13 = Activation('relu')(y_13)

    y_14in = Concatenate()([inputs, y_13])
    y_14 = Conv2D(filters=256, kernel_size=3, dilation_rate=14, padding='same', use_bias=False)(y_14in)
    y_14 = BatchNormalization()(y_14)
    y_14 = Activation('relu')(y_14)

    y_15in = Concatenate()([inputs, y_14])
    y_15 = Conv2D(filters=256, kernel_size=3, dilation_rate=15, padding='same', use_bias=False)(y_15in)
    y_15 = BatchNormalization()(y_15)
    y_15 = Activation('relu')(y_15)

    y_16in = Concatenate()([inputs, y_15])
    y_16 = Conv2D(filters=256, kernel_size=3, dilation_rate=16, padding='same', use_bias=False)(y_16in)
    y_16 = BatchNormalization()(y_16)
    y_16 = Activation('relu')(y_16)

    y = Concatenate()([y_pool, y_1, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


#indivnet 

def IndiVnet(n_classes=7, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    """ Inputs """

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    """ Pre-trained ResNet50 """

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    """ Pre-trained ResNet50 Output """
    image_features = base_model.get_layer('conv4_block6_out').output #16*16
    x_a = indienco(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    """ Get low-level features """
    x_b = base_model.get_layer('conv2_block2_out').output #64*64
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    x = Concatenate()([x_a, x_b])

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    """ Outputs """
    x = Conv2D(n_classes, (1, 1), activation='softmax')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[x])

    return model


def get_model():
    return IndiVnet(n_classes=8, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)

model = get_model()

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)