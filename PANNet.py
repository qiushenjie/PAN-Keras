from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Add, Lambda, concatenate
from base.mobilenetv2 import MobileNetV2, ConvBNRelu, DWConvBNRelu
import tensorflow as tf
from keras.utils.vis_utils import plot_model

def PANNET(input_shape, nc=2, result_num=6, scale: int = 1, train=True, pretrained=False):
    conv_out = 128
    inputs = Input(shape=input_shape)
    class_model, out = MobileNetV2(inputs)

    H, W, _ = input_shape

    def smooth(inputs, filter, mode='up'):
        if mode == 'up':
            x = DWConvBNRelu(inputs, (3, 3), (1, 1))
            x = ConvBNRelu(x, filter, (1, 1), (1, 1))
            return x
        elif mode == 'down':
            x = DWConvBNRelu(inputs, (3, 3), (2, 2))
            x = ConvBNRelu(x, filter, (1, 1), (1, 1))
            return x

    def sampling(x, img_w, img_h, method=1):
        """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法
            注：自定义采样方法在调用时须用layers.Lambda层来对操作进行封装,
            否则会出现'Tensor' object has no attribute '_keras_history'的报错
        """
        x = tf.image.resize_images(x, [img_w, img_h], method)
        return x

    def _upsample_add(x, y):
        if K.image_data_format() == 'channels_first':
            x = Lambda(sampling, arguments={'img_w': y.shape[3], 'img_h': y.shape[2]})(x)
            return Add()([x, y])
        else:
            x = Lambda(sampling, arguments={'img_w': y.shape[2], 'img_h': y.shape[1]})(x)
            return Add()([x, y])

    def _upsample_cat(p2, p3, p4, p5):
        if K.image_data_format() == 'channels_first':
            h, w = p2.shape[2], p2.shape[3]
        else:
            h, w = p2.shape[1], p2.shape[2]
        p3 = Lambda(sampling, arguments={'img_w': w, 'img_h': h})(p3)
        p4 = Lambda(sampling, arguments={'img_w': w, 'img_h': h})(p4)
        p5 = Lambda(sampling, arguments={'img_w': w, 'img_h': h})(p5)

        return Add()([p2, p3, p4, p5])

    def FPEM(fpem_inputs: list):
        # Up-Scale Enhancement
        p5 = fpem_inputs[-1]                      # size//32
        p4 = _upsample_add(p5, fpem_inputs[-2])   # size//16
        p4 = smooth(p4, conv_out, mode='up')      # size//16
        p3 = _upsample_add(p4, fpem_inputs[-3])   # size//8
        p3 = smooth(p3, conv_out, mode='up')      # size//8
        p2 = _upsample_add(p3, fpem_inputs[-4])   # size//4
        p2 = smooth(p2, conv_out, mode='up')      # size//4

        # Down-Scale Enhancement
        _p2 = p2                                  # size//4
        _p3 = _upsample_add(p3, _p2)              # size//4
        _p3 = smooth(_p3, conv_out, mode='down')  # size//8
        _p4 = _upsample_add(p4, _p3)              # size//8
        _p4 = smooth(_p4, conv_out, mode='down')  # size//16
        _p5 = _upsample_add(p5, _p4)              # size//16
        _p5 = smooth(_p5, conv_out, mode='down')  # size//32
        return [_p2, _p3, _p4, _p5]

    def FFM(fpems_outputs: list):
        ffm0 = []
        ffm1 = []
        ffm2 = []
        ffm3 = []
        if len(fpems_outputs) == 1:
            return fpems_outputs[0]
        elif len(fpems_outputs) == 2:
            return fpems_outputs[1]
        else:
            for i in range(1, len(fpems_outputs)):
                ffm0.append(fpems_outputs[i][0])
                ffm1.append(fpems_outputs[i][1])
                ffm2.append(fpems_outputs[i][2])
                ffm3.append(fpems_outputs[i][3])
            return Add()(ffm0), Add()(ffm1), Add()(ffm2), Add()(ffm3)

    # Reduce channels
    toplayer0 = ConvBNRelu(out[3], conv_out, kernel_size=(1, 1), strides=(1, 1)) # size//32
    toplayer1 = ConvBNRelu(out[2], conv_out, kernel_size=(1, 1), strides=(1, 1)) # size//16
    toplayer2 = ConvBNRelu(out[1], conv_out, kernel_size=(1, 1), strides=(1, 1)) # size//8
    toplayer3 = ConvBNRelu(out[0], conv_out, kernel_size=(1, 1), strides=(1, 1)) # size//4

    fpem_inputs = [toplayer3, toplayer2, toplayer1, toplayer0]  # size//4,  # size//8,  # size//16,  # size//32

    # FPEM + FFM
    fpems_outputs = [fpem_inputs]
    for i in range(1, nc + 1):
        fpems_outputs.append(FPEM(fpems_outputs[i - 1]))
    ffm0, ffm1, ffm2, ffm3 = FFM(fpems_outputs)

    outputs = _upsample_cat(ffm0, ffm1, ffm2, ffm3)
    outputs = ConvBNRelu(outputs, conv_out, (3, 3), (1, 1))

    if train:
        outputs = Lambda(sampling, arguments={'img_w': W, 'img_h': H})(outputs)
    else:
        outputs = Lambda(sampling, arguments={'img_w': W // scale, 'img_h': H // scale})(outputs)

    texts = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='normal')(outputs)
    texts = Activation('sigmoid', name='texts')(texts)

    kernels = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='normal')(outputs)
    kernels = Activation('sigmoid', name='kernels')(kernels)

    similarity_vectors = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='normal')(outputs)

    outputs = concatenate([texts, kernels, similarity_vectors], axis=3)


    # outputs = Activation('softmax', name='softmax')(outputs)

    model = Model(inputs, outputs)
    # model.summary()
    # plot_model(model, to_file='images/PAN.png', show_shapes=True)
    return model

if __name__ == '__main__':
    model = PANNET((256, 256, 3))
    model.summary()
