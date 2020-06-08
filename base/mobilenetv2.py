from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Input, Activation, Add, DepthwiseConv2D, BatchNormalization
from keras.layers import Reshape
from keras.utils.vis_utils import plot_model

assert K.image_data_format() == 'channels_last', 'backend should be tensorflow'

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    return K.relu(x, max_value=6.0)

def ConvBNRelu(inputs, filters, kernel_size, strides, padding='same'):
    # channels_last(tf): (batch, height, width, channel)
    # channels_first(Caffe/Theano): (batch, channel, height, width)
    # channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel_axis = -1

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='normal')(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)

def DWConvBNRelu(inputs, kernel_size, strides, padding='same'):

    # channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel_axis = -1

    x = DepthwiseConv2D(kernel_size, strides, depth_multiplier=1, padding=padding, kernel_initializer='normal')(inputs) # 深度方向卷积输出通道的总数将等于 filterss_in * depth_multiplier； 如果depth_multiplier=2，则是两个卷积核，卷积后的图征图堆叠起来
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)

def InvertedResidual(inputs, infilters , outfilters, strides, depth_multiplier, padding='same'):
    assert strides in [(1, 1), (2, 2)]
    # channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel_axis = -1

    hidden_dim = int(round(infilters * depth_multiplier))
    use_res_connect = strides == (1, 1) and infilters == outfilters

    x = inputs
    res_x = inputs
    if depth_multiplier != 1:
    # pointwise
        x = ConvBNRelu(x, hidden_dim, kernel_size=(1, 1), strides=(1, 1))
    # depthwise
    x = DWConvBNRelu(x, kernel_size=(3, 3), strides=strides)
    # pointwise-linear
    x = Conv2D(outfilters, kernel_size=(1, 1), strides=(1, 1), padding=padding, kernel_initializer='normal')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if use_res_connect:
        return Add()([x, res_x])
    else:
        return x

def MobileNetV2(inputs, n_class=17, width_mult=1.0, round_nearest=8):
        input_channel = 32
        inverted_residual_setting = [
            # time, output channels, number of InvertedResidual block, strides
            [1, 16, 1, 1], # 0
            [6, 24, 2, 2], # 1
            [6, 32, 3, 2], # 2
            [6, 64, 4, 2], # 3
            [6, 96, 3, 1], # 4
            [6, 160, 3, 2],# 5
            [6, 320, 1, 1],# 6
        ]
        feat_id = [1, 2, 4, 6]
        feat_channel = []

        if len(inverted_residual_setting) == 0 and len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        x = ConvBNRelu(inputs, input_channel, kernel_size=(3, 3), strides=(2, 2))
        x = ConvBNRelu(x, input_channel, kernel_size=(1, 1), strides=(1, 1))

        # building inverted residual blocks
        for id, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                strides = (s, s) if i == 0 else (1, 1)
                x = InvertedResidual(x, input_channel, output_channel, strides, t)
                input_channel = output_channel
            if id in feat_id:
                feat_channel.append(x)
            # last_channel = output_channel
        last_channel = 1280

        x = ConvBNRelu(x, last_channel, kernel_size=(1, 1), strides=(1, 1))
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, last_channel))(x)
        x = Dropout(0.3, name='Dropout')(x)
        x = Conv2D(n_class, kernel_size=(1, 1), padding='same')(x)
        x = Activation('softmax', name='softmax')(x)
        outputs = Reshape((n_class,))(x)
        model = Model(inputs, outputs)
        # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

        return model, feat_channel

if __name__ == '__main__':
    model, feat_channel = MobileNetV2(Input(shape=(224, 224, 3)), 100, 1.0)
    print(model.summary())
