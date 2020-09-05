from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D


def encoder_graph_vgg16(y_size, x_size, num_channels, num_classes):
    # Retreive VGG16 graph and weights and modify the graph.
    vgg16 = VGG16(weights='imagenet', include_top=True)

    # Get VGG16 layers.
    block1_conv1 = vgg16.get_layer('block1_conv1')
    block1_conv2 = vgg16.get_layer('block1_conv2')
    block1_pool = vgg16.get_layer('block1_pool')

    block2_conv1 = vgg16.get_layer('block2_conv1')
    block2_conv2 = vgg16.get_layer('block2_conv2')
    block2_pool = vgg16.get_layer('block2_pool')

    block3_conv1 = vgg16.get_layer('block3_conv1')
    block3_conv2 = vgg16.get_layer('block3_conv2')
    block3_conv3 = vgg16.get_layer('block3_conv3')
    block3_pool = vgg16.get_layer('block3_pool')

    block4_conv1 = vgg16.get_layer('block4_conv1')
    block4_conv2 = vgg16.get_layer('block4_conv2')
    block4_conv3 = vgg16.get_layer('block4_conv3')
    block4_pool = vgg16.get_layer('block4_pool')

    block5_conv1 = vgg16.get_layer('block5_conv1')
    block5_conv2 = vgg16.get_layer('block5_conv2')
    block5_conv3 = vgg16.get_layer('block5_conv3')
    block5_pool = vgg16.get_layer('block5_pool')

    fc_1 = vgg16.get_layer('fc1')
    fc_2 = vgg16.get_layer('fc2')

    # Convolutionize fc_1 and fc_2.
    fc_1_conv = convolutionize_dense_layer(fc_1, (7, 7, 512, 4096))
    fc_2_conv = convolutionize_dense_layer(fc_2, (1, 1, 4096, 4096))

    # Re-create graph.
    input_graph = Input(shape=(y_size, x_size, num_channels),
                        name='input_image')
    # graph = ZeroPadding2D(padding=100)(input_graph)

    # block_1
    graph = block1_conv1(input_graph)
    graph = block1_conv2(graph)
    graph = block1_pool(graph)

    # block_2
    graph = block2_conv1(graph)
    graph = block2_conv2(graph)
    graph = block2_pool(graph)

    # block_3
    graph = block3_conv1(graph)
    graph = block3_conv2(graph)
    graph = block3_conv3(graph)
    graph = block3_pool(graph)
    pool_3 = graph

    # block_4
    graph = block4_conv1(graph)
    graph = block4_conv2(graph)
    graph = block4_conv3(graph)
    graph = block4_pool(graph)
    pool_4 = graph

    # block_5
    graph = block5_conv1(graph)
    graph = block5_conv2(graph)
    graph = block5_conv3(graph)
    graph = block5_pool(graph)

    # fc_1_convolutionized
    graph = fc_1_conv(graph)

    # fc_2_convolutionized
    graph = fc_2_conv(graph)

    graph = Conv2D(num_classes, 1, name='encoder_graph')(graph)

    return input_graph, pool_3, pool_4, graph


def convolutionize_dense_layer(layer, kernel_shape):
    W, b = layer.get_weights()
    W_reshaped = W.reshape(kernel_shape)

    return Conv2D(kernel_shape[3], (kernel_shape[0], kernel_shape[1]),
                  padding='same', activation='relu', weights=[W_reshaped, b])


def encoder_graph(y_size, x_size, num_channels, num_classes):
    input_graph = Input(shape=(y_size, x_size, num_channels),
                        name='input_image')

    # block_1
    graph = Conv2D(64, 3, padding='same', activation='relu',
                   name='block1_conv1')(input_graph)
    graph = Conv2D(64, 3, padding='same', activation='relu',
                   name='block1_conv2')(graph)
    graph = MaxPooling2D(name='block1_pool')(graph)

    # block_2
    graph = Conv2D(128, 3, padding='same', activation='relu',
                   name='block2_conv1')(graph)
    graph = Conv2D(128, 3, padding='same', activation='relu',
                   name='block2_conv2')(graph)
    graph = MaxPooling2D(name='block2_pool')(graph)

    # block_3
    graph = Conv2D(256, 3, padding='same', activation='relu',
                   name='block3_conv1')(graph)
    graph = Conv2D(256, 3, padding='same', activation='relu',
                   name='block3_conv2')(graph)
    graph = Conv2D(256, 3, padding='same', activation='relu',
                   name='block3_conv3')(graph)
    graph = MaxPooling2D(name='block3_pool')(graph)
    pool_3 = graph

    # block_4
    graph = Conv2D(512, 3, padding='same', activation='relu',
                   name='block4_conv1')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu',
                   name='block4_conv2')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu',
                   name='block4_conv3')(graph)
    graph = MaxPooling2D(name='block4_pool')(graph)
    pool_4 = graph

    # block_5
    graph = Conv2D(512, 3, padding='same', activation='relu',
                   name='block5_conv1')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu',
                   name='block5_conv2')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu',
                   name='block5_conv3')(graph)
    graph = MaxPooling2D(name='block5_pool')(graph)

    # fc_6
    graph = Conv2D(4096, 7, activation='relu', padding='same',
                   name='fc_1')(graph)

    # fc_7
    graph = Conv2D(4096, 1, activation='relu', name='fc_2')(graph)

    graph = Conv2D(num_classes, 1, name='encoder_graph')(graph)

    return input_graph, pool_3, pool_4, graph
