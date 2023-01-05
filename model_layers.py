import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers


def _block(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    """
    creates a bottleneck block containing `repeat + 1` shuffle units
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    channel_map: list
        list containing the number of output channels for a stage
    repeat: int(1)
    .
        number of repetitions for a shuffle unit with stride 1
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    Returns
    -------
    """
    x = _shuffle_unit(x, input_channels=channel_map[stage - 2],
                      out_channels=channel_map[stage - 1], strides=2,
                      groups=groups, bottleneck_ratio=bottleneck_ratio,
                      stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = _shuffle_unit(x, input_channels=channel_map[stage - 1],
                          out_channels=channel_map[stage - 1], strides=1,
                          groups=groups, bottleneck_ratio=bottleneck_ratio,
                          stage=stage, block=(i + 1))

    return x


def _shuffle_unit(inputs, input_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    """
    creates a shuffle-unit
    Parameters
    ----------
    inputs:
        Input tensor of with `channels_last` data format
    input_channels:
        number of input channels
    out_channels:
        number of output channels
    strides:
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
    groups: int(1)
        number of groups per channel
    bottleneck_ratio: float
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    stage: int(1)
        stage number
    block: int(1)
        block number
    Returns
    -------
    """

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    prefix = 'stage%d/block%d' % (stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    # 1x1 GConv
    x = _group_conv(inputs, input_channels, out_channels=bottleneck_channels, groups=groups, name='%s/1x1_gconv_1' % prefix)
    x = layers.BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
    x = layers.Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    # channel shuffle
    x = layers.Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)

    # DepthWise Convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False, strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
    x = layers.BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

    x = _group_conv(x, bottleneck_channels, out_channels=out_channels if strides == 1 else out_channels - input_channels, groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = layers.BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        ret = layers.Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = layers.AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        ret = layers.Concatenate(bn_axis, name='%s/concat' % prefix)([x, avg])

    ret = layers.Activation('relu', name='%s/relu_out' % prefix)(ret)

    return ret


def _group_conv(x, input_channels, out_channels, groups, kernel=1, stride=1, name=''):
    """
    grouped convolution
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    input_channels:
        number of input channels
    out_channels:
        number of output channels
    groups:
        number of groups per channel
    kernel: int(1)
        An integer or tuple/list of 2 integers, specifying the
        width and height of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    stride: int(1)
        An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the width and height.
        Can be a single integer to specify the same value for all spatial dimensions.
    name: str
        A string to specifies the layer name
    Returns
    -------
    """

    if groups == 1:
        return layers.Conv2D(filters=out_channels, kernel_size=kernel, padding='same', use_bias=False, strides=stride, name=name)(x)

    # number of input channels per group
    offset = input_channels // groups
    group_list = []

    for i in range(groups):
        group = layers.Lambda(lambda inputs: inputs[:, :, :, i * offset: offset * (i + 1)], name='%s/g%d_slice' % (name, i))(x)
        group_list.append(layers.Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride, use_bias=False,
                                        padding='same', name='%s_/g%d' % (name, i))(group))

    return layers.Concatenate(name='%s/concat' % name)(group_list)


def channel_shuffle(x, groups):
    """
    Parameters
    ----------
    x:
        Input tensor of with `channels_last` data format
    groups: int
        number of groups per channel
    Returns
    -------
        channel shuffled output tensor
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])

    return x
