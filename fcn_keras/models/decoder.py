from keras.layers import Conv2D, Conv2DTranspose, Add


def decoder_graph_8x(pool_3, pool_4, encoder_out, num_classes):
    # Reduce channels of pool4 to num_classes.
    conv_pool_4 = Conv2D(num_classes, 1)(pool_4)

    # Reduce channels of pool3 to num_classes.
    conv_pool_3 = Conv2D(num_classes, 1)(pool_3)

    # Upsample final layer of encoder by 2x, then merge with pool4.
    out_2x = Conv2DTranspose(num_classes, 4, strides=2,
                             padding='same')(encoder_out)
    merged_2x = Add()([out_2x, conv_pool_4])

    # Upsample merged layer by 2x, then merge with pool3.
    out_4x = Conv2DTranspose(num_classes, 4, strides=2,
                             padding='same')(merged_2x)
    merged_4x = Add()([out_4x, conv_pool_3])

    # Upsample to image shape.
    return Conv2DTranspose(num_classes, 16, strides=8, padding='same',
                           activation='softmax')(merged_4x)


def decoder_graph_16x(pool_4, encoder_out, num_classes):
    # Unpool to 16x
    out_2x = Conv2DTranspose(num_classes, 4, strides=2,
                             padding='same')(encoder_out)
    conv_pool_4 = Conv2D(num_classes, 1, padding='same')(pool_4)
    merged_16x = Add()([out_2x, conv_pool_4])

    # Unpool to image shape
    return Conv2DTranspose(num_classes, 32, strides=16, padding='same',
                           activation='softmax')(merged_16x)


def decoder_graph_32x(encoder_out, num_classes):
    return Conv2DTranspose(num_classes, 64, strides=(32, 32),
                           padding='same', activation='softmax')(encoder_out)
